# GYRE lane — merge `origin/main` 2026-10-03 (PR 3 preparation)

The third pull request of the NEMO-fidelity campaign takes the GYRE lane
(`fidelity/nemo-testcases-l2-gyre-codex2`) to `main`. This receipt records the
merge that precedes it: what arrived, what conflicted, what was re-anchored,
and every gate measured on the merged tree.

## Tips merged

| side | tip | date |
|---|---|---|
| lane (first parent) | `4646a680e` — round-210 VORTEX 100-day movie | 2026-10-03 |
| `origin/main` (second parent) | `87144e259` — PR #1866, canopy-convergence reporting | 2026-10-03 |
| merge base | `ea12107cb96b9e38935cdb49070ec6a8045402a0` | |
| **merge commit** | **`d8f908a41a615e8c60b514c3ed4550f38353aad2`** | 2026-10-03 |

250 commits arrive from `main` (389 files); the lane carries 1078 commits
`main` does not have. The previous PR, #1842, merged on 2026-09-30; everything
after it is in PR 3.

## Conflicts — none, and why that is a measurable fact rather than luck

`git merge --no-ff origin/main` exited 0 with **zero conflicted paths**.

That is not an assertion about care taken; it follows from the file sets. The
intersection of "files `main` changed since the merge base" and "files the lane
changed since the merge base" has exactly **one** member:

```
$ comm -12 <(git diff --name-only $BASE origin/main | sort) \
           <(git diff --name-only $BASE 4646a680e   | sort)
packages/ocean/legoesm/ocean/eos.py
```

and the two sides touched disjoint regions of it:

| side | hunks in `eos.py` | what |
|---|---|---|
| `main` | `@@ -145,11 +145,16 @@ def wright_eos(` | **docstring only** — rewrites the precision note to say the Wright polynomial runs in the precision policy's `equation_of_state` compute dtype, not the input dtype. No executable line changed. |
| lane | `@@ -746,12 @@`, `@@ -770,12 @@` (`compute_buoyancy_frequency_nemo_bn2`), `@@ -960,6 @@` (`nemo_r3t_stretch` / `nemo_r3t_rk3_stage1_stretch`) | the lane's NEMO `bn2` and stage-one stretch statements |

`git diff 4646a680e HEAD -- packages/ocean/legoesm/ocean/eos.py` is **exactly
`main`'s one prose hunk** and nothing else. No physics line was changed by both
sides, so no `DECISION_NEEDED` arises; nothing was chosen between two versions
because nothing was in contention.

**Citation map — the precise statement, corrected under review.** My first
draft said "the merged map is the set union by construction". That is wrong,
and the reviewer was right to refuse it. What actually happened:

| blob of `nemo_testcase_receipt_citation_gate.py` | sha |
|---|---|
| merge base | `92b112bec` |
| `main` parent | `92b112bec` — **identical; `main` never touched this file** |
| lane parent | `97a8712e5` |
| merge result | `97a8712e5` — **the lane blob, verbatim** |

So there was no union to perform. The property that matters — *this merge
drops no entry from either parent* — holds, and is measured rather than
asserted: 63 keys present in the `main` parent are absent from the merge, and
**all 63 are also absent from the lane parent**, i.e. the lane's own earlier
rounds retired them (re-anchors and renames). Entries dropped **by this
merge: 0**. `main` also added no receipt under `docs/ocean/fidelity` (0 files),
so no citation arrived from that side needing a home. The gate confirms the
live property: 274 citations found across the audited receipts, **0 unmapped**,
0 map entries failing audit.

## What `main` changed on the ocean fidelity paths, named

16 files under `packages/ocean/legoesm/ocean/{dynamics,fidelity,physics}` and
`scripts/validate/ocean_fidelity` moved on the `main` side. **`ocean/fidelity/`
is untouched** — the whole directory has zero `main`-side changes. The rest:

| file | what `main` did | reaches the certified cards? |
|---|---|---|
| `dynamics/barotropic_common.py` (+95) | all hunks at line 1084+, inside `_fixed_iteration_pcg_single_reduce` and `solve_helmholtz_implicit` — the MPAS/SPMD deep-halo Jacobi PCG work (#1847/#1859) | **No.** The certified latlon C-grid path imports **six** symbols from this module: `nemo_literal_after_level_reconcile`, `nemo_reference_depth_reciprocal`, `rk3_stage_barotropic_correction`, `validate_after_reconcile` (top of file) plus `coriolis_at_faces` (function-scope, `ocean_model_latlon_cgrid.py:2485` and `:6833`) and `nemo_auto_substeps` (`fidelity/nemo_testcase_recipe.py:20`). **All six are byte-identical across merge base, `main` and the merge** (each function body extracted and hashed). My first draft said "exactly four", found by a top-of-file line-grep that cannot see deferred imports — the reviewer's AST scan found the other two. The conclusion survived; the method that produced it did not, and the method is what is recorded here. |
| `dynamics/barotropic_implicit_mpas.py`, `barotropic_mpas.py`, `mpas_fill.py`, `ocean_model_mpas.py`, `ocean_pe_mpas.py` | SPMD edge-shard ordering, deep-halo PCG default, land-fill gather, `IntegrationMixin` | **No** — zero references from the certified path. |
| `dynamics/sfno_ocean.py`, `sharded_ocean_step.py`, `spectral_ocean_pe.py` | ponytail refactors (shared `IntegrationMixin`, shared git-SHA helper) | **No** — zero references from the certified path. |
| `physics/tendencies.py` (+4/-4) | `make_none_physics_fn`'s returned callable gains a trailing `dt=None` parameter — a pure addition | **YES, and this is the one `main`-side edit a certified card executes.** Corrected under review: `ocean_model_latlon_cgrid.py:3212` imports the surface-forcing integration factory and line 3222 builds a `scheme="none"` config, and `surface_forcing/integration.py:89-90` returns `make_none_physics_fn()` for that scheme (also `fidelity/nemo_match_recipe.py:353`). The edited `def physics_fn(...)` line therefore runs. It is **inert**: the added parameter is a trailing keyword with a default, the body and return are unchanged, and nothing in `packages/ocean/` introspects a signature (`grep -c 'inspect.signature|getfullargspec|co_varnames'` over the package returns **0**). The gates below are what prove it inert in fact rather than in argument. |
| `physics/tidal_forcing.py` (-5) | deletes `tidal_acceleration_at`, a single-shot convenience wrapper | **No** — `grep -rn tidal_acceleration_at` over `packages src scripts tests` returns **zero** callers; the certified path's only `tidal_forcing` references are `getattr(config, "tidal_forcing", ...)` construction-time validation, not this module. |
| `scripts/validate/ocean_fidelity/{arctic_deep_convection_columns,barotropic_pcg_convergence,compare_three_way_nemo,mle_psi_diagnostics,omip_conservation_closure}.py` | 13 git-SHA helpers now call `legoesm.io.git_provenance` (#13) | Diagnostics, not gates. None is in the lane's push battery or any card gate. |

**Seven more files under `legoesm/ocean/` moved on the `main` side** outside
the three directories the brief named, and are listed here rather than left
silent: `config.py` (its `_deep_merge` swapped for `core.setup_selector.
deep_merge` — the same in-place loop verbatim), `experiments/dino.py` (two
dead helpers removed, `restoring_timescale_T_days` / `_S_days`, zero callers
tree-wide and never touched by the lane), and `bulk_flux_omip.py`,
`simple_ocean.py`, `init_mpas.py`, `mpas_config.py`, `eos.py` (the docstring
above). The first two are reachable and were each cleared by reading the
change; the rest are not on the certified path.

The prediction all of this supports — *the only `main`-side edit any certified
card executes is the inert `dt=None` parameter above* — is then **measured**,
not assumed, by the gates below.

## Re-anchoring — two citations, by symbol

`main`'s `eos.py` docstring is +10/-5, so every line after 145 shifted by +5.
Two citation-map entries anchored past that point. The gate itself named the
symbol and the line it now identifies; both were re-anchored **by symbol**, not
by adding 5:

| citation | anchor symbol | before | after |
|---|---|---:|---:|
| `eos.py:742` | `"raw-mesh e3w_int must contain only finite values > 0",` | 742 | **747** |
| `eos.py:970-1001` | `def nemo_r3t_rk3_stage1_stretch(` … `return jnp.where(wet, nemo_source_round(one + r3_stage), one)` | 970-1001 | **975-1006** |

The 32-line span is preserved (975 + 31 = 1006, which is where the end symbol
now sits — verified, not computed). Two prose mentions follow the map, in the
`pr1802` final-recertification receipt and the ORCA2 round-101 merge-owner
receipt. Landed as `268a88c2b`.

**Non-vacuity.** Before the re-anchor the gate was `FAIL` with those exact two
entries `SYMBOL-NOT-AT-LINE` and 0 unmapped
(`merge_main_2026-10-03/citation_gate.log`); after it, `PASS`
(`citation_gate_after.log`). The gate's own nine planted shifts all fired.

**The merge commit alone is gate-red**, as in round 207: the re-anchor is in
the following commit, so a bisect landing exactly on `d8f908a41` fails the
citation gate. Recorded, not squashed.

## Measurements on the merged tree

Every number below is measured on the merged tree with the scripts the recent
receipts cite, not carried over. All of it ran on the merge plus the
re-anchor commit (`268a88c2b`) with a clean worktree; three of the eight card
gates were stamped at `d8f908a41` (the merge alone) because they completed
before the re-anchor landed, and five at `268a88c2b` — the re-anchor touches
only `docs/` and the gate script, no `packages/` file, so the two stamps name
the same model code.

### GYRE certified `kt=1..10` ladder — UNCHANGED

`nemo_testcase_l2_gyre_phase3_gate.py --max-step 10`, compared against round
208's certified report with `nemo_testcase_offline_compare.py`:

```
OFFLINE_ORACLE_RELATIVE_COMPARE PASS: rows=954 max_worsening_ulps=0
  first_over_bar={'T','S','u','v','ssh'} kt=3 -> {'T','S','u','v','ssh'} kt=3
```

**954 rows, 0 moved, 0 ULP, first-over-bar step unchanged.**

### GYRE from-rest year — BYTE-IDENTICAL TO THE CERTIFIED PIN

A fresh seed-0 member ran the full 360 days
(`nemo_testcase_l2_gyre_year_fromrest.py --member 0 --days 360 --snap-steps 6
--tag pr3a`) and was scored with the committed day-gap scorer on the same
eight days, the same NEMO restarts, the same metric.

| day | measured on the merged tree | certified (note BZ / round 207) | delta |
|---:|---|---|---:|
| 30 | `2.3432465132112266e-06` | same | **0** |
| 60 | `1.4793247973304582e-05` | same | **0** |
| 90 | `1.6332712039638441e-05` | same | **0** |
| 120 | `1.0965907352116351e-04` | same | **0** |
| 180 | `6.1153355393000553e-05` | same | **0** |
| 240 | `6.5817060949447295e-05` | same | **0** |
| 300 | `5.4660498451870513e-05` | same | **0** |
| 360 | `5.4077419367442036e-05` | same | **0** |

**0 of 8 days differ, to all 17 printed digits.** The snapshot digests agree
too — file-level SHA-256 of the member's daily `.npz`:

| day | measured | certified (round 207) |
|---|---|---|
| 030 | `4e36c106403b495e95327213292f0d1655d605fca6b0a75c67cb17833f067cba` | identical |
| 240 | `8b9cd60475626373d9a2fec0baa508f06c1f91aed4c3d7a24fc5d8b3f5878c8a` | identical |
| 360 | `e3e0a068346c7866f0a318bb32141f2fb326cc05c158a1c95bc39b091f585e25` | identical |

Byte-identical daily state, not merely an agreeing score. The certified GYRE
year is **unchanged by this merge**; note BZ stands.

**A cost owned.** My first two day-gap invocations were wrong and I nearly
read their output as a result. `--days` on that scorer is a comma-separated
day LIST, not a day count, and the member lives under `phase3/year_fromrest`,
not the scorer's default root — so `--days 360` scored exactly one day and
the default root refused with a missing-snapshot error. Both are recorded
here rather than quietly re-run; the reported table is the third invocation,
`--days 30,60,90,120,180,240,300,360 --lego-root .../year_fromrest`.

### All eight certified cards — 0/50 EACH

`nemo_testcase_phase3_trajectory_gate.py --case <CARD> --max-step 10
--continue-after-first`, each compared with `nemo_testcase_offline_compare.py`
against its own immutable reference (round 208's ladders for the six VORTEX
cards; round 207's for the two tanks).

| card | rows | moved | largest worsening | first over bar (ref -> candidate) |
|---|---:|---:|---:|---|
| `VORTEX-zco` (flux, 30 km) | 50 | **0** | 0.0 ULP | T,u,v,ssh kt=2 -> unchanged |
| `VORTEX_VEC-zco` (vector, 30 km) | 50 | **0** | 0.0 ULP | u,v,ssh kt=2 -> unchanged |
| `VORTEX-15km-zco` | 50 | **0** | 0.0 ULP | T,u,v,ssh kt=2 -> unchanged |
| `VORTEX_VEC-15km-zco` | 50 | **0** | 0.0 ULP | u,v,ssh kt=2 -> unchanged |
| `VORTEX-10km-zco` | 50 | **0** | 0.0 ULP | T,u,v,ssh kt=2 -> unchanged |
| `VORTEX_VEC-10km-zco` | 50 | **0** | 0.0 ULP | u,v,ssh kt=2 -> unchanged |
| `LOCK_EXCHANGE-zco` | 50 | **0** | 0.0 ULP | u kt=8 -> unchanged |
| `OVERFLOW-zps` | 50 | **0** | 0.0 ULP | T,u kt=2 -> unchanged |

The flux card's step-2 velocities still read `1.216831e-08` / `1.239432e-08`
(u/v) and the vector card's `1.304438e-15` / `1.335683e-15`, i.e. the
numbers notes BY and CA certified.

**Non-vacuity of the card comparator.** The unplanted compare of `VORTEX-zco`
exits 0; the same compare with `--plant worsen-3ulp` exits 1
(`FAIL ... max_worsening_ulps=3`) and with `--plant at-bar-to-debt` exits 1.
A comparator that cannot fail would have reported the same `0 moved`.

### ORCA2 — 0/400

The ORCA2 lane's own scripts, now in the merged tree, run against the ORCA2
records and compared with the ORCA2 lane's own comparator against round 207's
registered ladders (the tightest available base: round 207 already showed
those reproduce round 112's).

```
STATUS PASS_R111_ORCA2_LADDER_COMPARE
rung0: row_count 200, moved_row_count 0, bit_identical_losses 0
rung7: row_count 200, moved_row_count 0, bit_identical_losses 0
```

rung-0 (`--record-root round90/acquisition/orca2_rung0_entry_stage_runoff_
guarded_10step_np2`) returns `PASS_RUNG0_TEN_STEP_LADDER` with its first
non-bit checkpoint still `kt=1 stage1 T`; rung-7 was run in the certified
Decision-52 given-entry mode (`--initial-mode decision52-bridge`), the mode
confound round 207 paid for.

### DINO from-rest month

```
DINO from-rest month day-30 wet 3-D T rms vs NEMO kt=960:
  2.053801168e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS
```

Unchanged from the reference the brief pins.

### Batteries

The lane's push gate — the receipt citation gate, `test_tke_nemo_terms`,
`test_nemo_recipe` (the generic NEMO-GYRE recipe gate), `test_real_freshwater_
closure`, `test_nemo_ws_stage_face_mask_rank`, `test_nemo_prognostic_
barotropic_state`:

```
======================= 136 passed in 1185.34s (0:19:45) =======================
```

Same 136 as round 207.

### Broad ocean battery — and the confound I created, then removed

`tests/ocean/fidelity tests/ocean/unit tests/unit/test_run_omip_cli.py`, `-n 12`.

**First run is DISCARDED as my own confound.** I launched it while the 360-day
GYRE year and the DINO GPU gate were still running. It reported 141 failed /
43 errors, and the log carries **2790 `Cannot allocate memory`, 145 `Failed to
materialize symbols` and 8 crashed xdist workers** — the campaign's documented
collapse mode, not a result. Kept as
`broad_battery_CONFOUNDED_concurrent.log` so the claim is checkable; its
numbers are not reported as findings.

**Second run, on an otherwise idle machine:**

```
= 140 failed, 9435 passed, 179 skipped, 2 xfailed, 88 warnings, 10 errors in 4079.79s (1:07:59) =
```

Seven workers still died with `Fatal Python error: Aborted`, so this run is
also not taken at face value. The two runs' failing sets overlap in only **47
of 184 / 150** IDs — a failing set that is not reproducible between two runs of
the *same tree* is the scheduler, not the code. Those 47 were then re-run
**serially, no `-n`**, on two trees:

| tree | result on the 47 |
|---|---|
| merged (`d8f908a41` + re-anchor) | **25 failed, 22 passed** |
| pre-merge lane tip (`4646a680e`, a detached worktree) | **25 failed, 22 passed** |

**The two failing sets are IDENTICAL** (`diff` of the sorted ID lists is
empty). So:

| classification | count |
|---|---:|
| genuine merge interaction | **0** |
| lane debt, red before the merge and after it | **25** |
| `main`'s own debt arriving with the merge | **0** |
| xdist worker-pressure collateral, not reproducible serially | the remaining 115-159 |

The 25 are the campaign's known-red categories, unchanged by this merge: the
worktree-stamp ratchet and its scope test, four float32 advection-gradient
underflow cases, the three `stpmlf` call-coverage rows, the two step-entry
buoyancy-bundle rows, the SI3 scalar-math v2 provenance gate, the recipe
case-board row gate, and nine more lane items
(`reclass_merged.log` / `reclass_premerge.log` hold both lists).

**Nothing was fixed**, because there was nothing merge-caused to fix.

### One note on the battery-check rule itself

The campaign's one-battery check is
`ps -eo args | grep '[v]env/bin/python -m pytest' | grep -vc '/bin/bash -c'`.
It printed **0 while this PR's own push battery was running**, because
`land.sh` and this receipt's runner both put `.venv/bin` on `PATH` and invoke
plain `python -m pytest`, which `ps` shows as `python -m pytest`. The rule as
written does not see the campaign's own batteries. Reported, not fixed — it is
the brief's rule to change, not mine.

## Evidence

`phase3/merge_main_2026-10-03/`: `merge.log`, `mainside_ocean_commits.txt`,
`mainside_ocean_files.txt`, `overlap_files.txt`, `citation_gate.log`,
`citation_gate_after.log`, `gyre_gates.sh`, `gyre_ladder_pr3.{json,log}`,
`cmp_gyre_ladder.json`, `gyre_year_pr3a.log`, `gyre_day_gap_pr3a.{json,log}`,
`cards.sh`, `cards2.sh`, `cards/{CARD}.{json,stdout}`, `cards/cmp_{CARD}.json`,
`orca2.sh`, `orca2_rung0_ladder.{json,log}`, `orca2_rung7_ladder.{json,log}`,
`orca2_ladder_compare.{json,log}`, `dino_month_gate.log`,
`push_battery.log`, `broad_battery.log`,
`broad_battery_CONFOUNDED_concurrent.log`, `broad_failures.txt`, `stable_fail.txt`,
`classify.sh`, `reclass_merged.log`, `reclass_premerge.log`, `cards/plant_*.log`.
