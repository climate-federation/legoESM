# NEMO SWG double-gyre reproduction package: Round E receipt

Date: 2026-10-14. Status: **implemented, uncommitted** in the work copy
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/claude-swgdgsa-round-e`
(local branch `claude/swgdgsa-round-e`, base `d7d2e21f6`), for the operator to
review and commit. Round E closes the Claude review of Round D (journal
`swg_rounds/reviews/claude_review_swgdgsa_d_journal.jsonl`; every finding
confirmed by a second reviewer). It changes no model physics, deck value,
acceptance bar, or NEMO source, and ran no GPU, `makenemo`, or MPI command.
Evidence logs are in the scratch directory
`/data/abyssal/dbalwada/.claude/jobs/b3690d5d/tmp/claude_dg_e/impl`
(called SCRATCH below); the work copy and the operator's run directories were
only read.

Decisions applied as given: the exactness policy decided on 2026-10-13
(prognostic bar exact; the sign of zero in forcing fields is the one
exception; the per-step replay's last bit is recorded debt), and the
`JAX_PLATFORMS=cuda` mapping of `d7d2e21f6`.

## Review disposition

| Finding | Disposition | Evidence (command → decisive output) | Reversion that makes its test fail |
|---|---|---|---|
| G1 deck guard not exhaustive | **FIXED.** The compiled-read inventory is now derived from the build's preprocessed sources (every namelist-buffer `READ` reachable from `nemo_gcm`, guards ignored): 27 groups, 336 fields, including `nammpp`, `namtsd`, `namsbc_rnf`, `namusr_def` and ten groups this deck switches off. The guard compares the 147 fields with a card counterpart (158 rows) and pins the other 189 to the reference run's values, each with a per-field reason in `swg_dg_nemo_only_namelist_fields.json`; it refuses a field neither compared nor registered, an unresolvable field, an echoed field or group outside the inventory, an echoed group missing a field, a different echoed-group set, an unparsable line or repeated block in a read group, and a card field neither matched nor registered card-only (19). The build runs the inventory on the colleague's own build; the guide gives the command. | Inventory CLI on the fresh rebuild and on the shipped-SWG build → `SWG_DG_NAMELIST_INVENTORY PASS groups=27 fields=336 excluded_reads=2 matches_committed_reference=True`; live test with `SWG_DG_PPSRC` on the reference and fresh builds → `1 passed`. Guard on the reference three-year, fresh three-year and fresh smoke runs → `SWG_DG_DECK_GUARD PASS compiled_reads=336 groups=27 card_compared=147 nemo_only_pinned=189 comparison_rows=347 allowed_differences=2` (reports SCRATCH/`final_guard_*.json`, SHA-256 `20e8ebed…`, `9a535f55…`, `f3c77d9e…`). The review's own plant scripts on copies of the reference run: `nn_hls`, `jpni`, `ln_tsd_init`, `rn_hrnf` changes, `rn_tau`/`rn_dt`/`nn_hls` deletions, and the appended `&nammpp nn_hls = 1` all exit 2; baseline and the allowlisted `nn_stock` change exit 0. Fixture plants in the guards test: all refuse. | R1a drop function-reference edges → `['nama','namb','namd'] == [… 'namf']` fails. R1b drop string masking → `unresolved namelist READ at main.f90:9`. R1c no inventory comparison → `assert 0 == 2`. R2a drop echo completeness and row count → 5 deletion plants `DID NOT RAISE`. R2b stop comparing pinned rows → 5 change plants `DID NOT RAISE`. R2c/d/e/f/g/h (echo-group set, extra field, neither-check, card-only check, unparsed line, repeated block) → each test fails. R9 drop `nn_jctle` from the committed inventory → the card test's frozen-group comparison fails. |
| G4 guard depends on the caller's float64 | **FIXED at the shared bootstrap**: every entry point selects float64 before importing JAX and refuses clearly if JAX was already imported in float32. The same misleading refusal also hit the annual checker, climate tool and movie, which share the bootstrap. | Annual checker without `JAX_ENABLE_X64` on the reference year-1 inputs: before → `REFUSE: SWG-zco grid differs from resolved dx/beta/f0 geometry`; after → `SWG_DG_RESTART_IDENTITY PASS day=365 step=630720 fields=3 wet_words=1227520 unequal=0`. Guard CLI in a shell without the variable → the PASS line above. | R3 remove the selection → the guard subprocess test and both annual/climate tests fail (`JAX x64 is disabled: JAX was imported with float32 …`). |
| G7 constants no namelist carries | **FIXED in the guide** (step 5): density, Earth radius, rotation rate, 500 m layer, gravity, and the 19 card-only fields, and that only the NEMO revision, build manifest and annual identity bind them. The 19 are now an explicit registry the guard checks against the card. | Citations below. | R2f (card-only check) as above. |
| G2/F1/F5 self-test cannot run in the guide's environment | **FIXED.** A package self-test of eleven files with its own pin file `python-selftest-requirements.txt` (pytest 9.0.3, pluggy 1.6.0, iniconfig 2.3.0, pygments 2.20.0, PyYAML 6.0.3), installed by the guide. Measured in a venv holding only the 45 pins, with external evidence hidden, checkout read-only: 279 passed, 1 skipped, 177 s, peak 1.2 GiB (largest process). The gradient matrix stays out (needs `optax`/`xarray`/`pandas`, a row peaked at about 34 GB in the review, and it checks gradient health across many ocean configurations, only one row being the SWG card); it runs in the maintainers' lane battery. | SCRATCH/`selftest_run4.log` (SHA-256 `9dceacfb…`), committed as `swg_dg_selftest_reference.json`; run 4 is on the final files (identical-count runs 2 and 3 are superseded by later edits to a test string and to guide prose). The emulated venv lists exactly 45 distributions; `optax`, `xarray`, `pandas` raise `ModuleNotFoundError` in it. | R10 change one guide row → the runbook test fails against the committed record. |
| F2 repository-only result wrong | **FIXED.** The guide's repository-only result is the measurement above, not derived. The gradient row that read external records without a guard now skips through the ladder's own admission check, and only when the record is missing. | Hidden run of that row → `SKIPPED … external SWG stage evidence absent: missing content-pinned record …/swg_round1_admission.json` (SCRATCH/`matrix_swg_hidden.log`). | R11 remove the skip, same hidden run → `1 failed` (in this uncommitted tree the child first refuses on its clean-tree stamp; the review measured the missing-record failure on a clean commit). |
| G6 environment test not exact | **FIXED.** Every line of both pin files must be `name==version`, the maps must equal the versions recorded for both files in `swg_double_gyre_environment.json`, each file's SHA-256 must equal the recorded one, the running environment must hold exactly those versions, and the closure must be complete. | Guards test passes. | R7a `scipy==1.16.0` → map differs; R7b `scipy>=1.0` → `not an exact pin`; R7c delete `six` → map differs. |
| G3/G5/F11 silent host dependencies | **FIXED.** Guide and environment doc state glibc ≥ 2.34 with `ldd --version` (measured: three lock packages declare `__glibc >=2.34`; the reference executable uses `GLIBC_2.34`; host glibc 2.34), the tested driver 580.126.20 (measured) and the documented CUDA 12.9 Update 1 / minor-compatibility / JAX minimums, `LD_LIBRARY_PATH`, the cuDNN 9.21 generation note (no convolution in the GPU path, by grep), host binutils `ar` (build refuses without it), and `$REPRO` outside the checkout. Identity, run and rank tools take a required `--conda-prefix`, put it first on `PATH` and launch its `mpirun` by default; `--oversubscribe` and `--mpiexec` are documented. | Shell tests: 23 passed. | R8a launcher falls back to `mpirun` on `PATH` → preflight test fails; R8b prefix without `mpirun` accepted → refusal test fails; R8c archiver check removed → `assert 67 == 66`. |
| F3 GPU short ladder always refuses | **FIXED, and a second refusal found and fixed.** The round-7 and composition helpers no longer bootstrap at import (their `main` does). Behind that, the scorer built its case through the certified CPU-only builder, which refuses a GPU backend; the GPU form now scores the long-run driver's own case (shared `build_case`). | Emulation (JAX on CPU reporting `gpu`, as an initialised CUDA backend does): the scorer reaches its first input, `missing single-rank restart`; the CPU control reaches the same point. | R4a restore the import-time bootstrap → `JAX platform mismatch: requested cpu, selected gpu`; R4b restore the CPU builder for GPU → `short restart ladder must run on CPU`. |
| F4 climate PASS is not a verdict | **FIXED by renaming**: the final line is `SWG_DG_CLIMATE_REPORT_WRITTEN window=… agreement_test=none …`; no agreement bar was added. The two fp64 reference reports are committed byte-identical (`swg_dg_climate_year1_reference_report.json` SHA-256 `6919fc38…`, `…years2_3…` `a9ca4157…`), and the guide quotes per-model KE/EKE means and ranges for both windows, with a test that recomputes them from the reports. | Climate tests: 9 passed. | R5 restore `PASS …` → the final-line test fails. |
| F6/G9 policy shown as pending | **FIXED.** Guide section "Exactness policy (decided …)"; the round-7 and composition tools write the ruling. The committed composition JSON keeps its recorded pre-ruling label, as the guide says. N1 (stale PR body) needs the operator; see below. | Round-7 tests: 6 passed. | R6 restore the pending label → the round-7 policy test fails. |
| F7 three counts | **FIXED.** 1,278 is labelled reference-only evidence (comparison (iv) of the round-7 receipt) with the operator-only acquisition command that made it; the receipt/PR wording that called the counts three boundaries is superseded by "PR description must say" below. | Guide text. | Runbook test (required strings). |
| F8 fresh run was not the documented procedure | **FIXED in the guide**: what the 2026-10-13 fresh reproduction covered and did not. | Guide text. | Runbook test. |
| F9 cost spread | **FIXED.** Guide and costs JSON give 22,918 s and 17,404 s (fresh rebuild, same host), the CPU, and the movie's 1,794 s, each with its source log hash. | `WALL_SECONDS=17404` (fresh run.log SHA-256 `145cabbb…`), `MOVIE_WALL_SECONDS=1794` (log SHA-256 `7c154a1b…`). | Runbook test checks the four times against the costs JSON. |
| F10 cannot attribute a departure | **FIXED.** A tile-hash command in step 5 compares the colleague's 96 NEMO restart tiles with the hashes in the three committed annual reports, before GPU time. | On the fresh rebuild → `NEMO_RESTART_TILES_MATCH_REFERENCE tiles=96`; one tile with one byte appended → `FAILED`, exit 1, no match line. | Command, not a test. |
| F12 reference-build flag | **FIXED in the guide**: the flag records byte identity of executable and namelists with the 2026-10-13 run, is expected `false` elsewhere, and is not a pass/fail result. | Guide text. | Runbook test. |
| F13 legoESM checkout unpinned | **FIXED in the guide**: clone URL, branch, detached checkout of the hand-off commit, and a clean-tree check. The commit itself must be supplied at hand-off (a file cannot name the commit that contains it). | Guide text. | Runbook test. |
| G8 environment commands never run | **STATED.** Guide and environment doc say plainly that neither `conda create` nor the `pip install` commands has been run from scratch, and what was checked instead. | Guide text. | Runbook test. |

## Where the code and NEMO say so

The derivation keeps namelist-buffer reads
(`nemo_testcase_l1_swg_namelist_inventory.py:341`) in routines reachable from
the entry (`:278`, `:303`), after masking string literals (`:95`); a build is
compared with the committed inventory (`:388`) and the CLI prints `:437`.
The one excluded read pair is NEMO's self-test on a literal string
(`SWG_DG/BLD/ppsrc/nemo/lib_mpp.f90:2784`); `nambdy` is read without an echo
(`SWG_DG/BLD/ppsrc/nemo/mppini.f90:201`).

The guard's registries and refusals: card-only fields
(`nemo_testcase_l1_swg_dg_deck_guard.py:51`), resolution and echo
completeness (`:223`, `:242`), classification and the neither-check (`:267`,
`:288`), the echoed-group set (`:332`), the row count (`:368`), strict
parsing (`:444`), and the PASS line (`:515`). Float64 is selected and
verified in the shared bootstrap (`nemo_testcase_checkout.py:56`, `:75`).

Gates behind the pinned groups, in the reference build: `sbc_blk_init` only
when `ln_blk` (`SWG_DG/BLD/ppsrc/nemo/sbcmod.f90:365`), `sbc_rnf_init` always
(`:372`) but returning when `ln_rnf` is false
(`SWG_DG/BLD/ppsrc/nemo/sbcrnf.f90:331`), `sbc_wave_init` always
(`SWG_DG/BLD/ppsrc/nemo/sbcmod.f90:378`) but returning before its read when
`ln_wave` is false (`SWG_DG/BLD/ppsrc/nemo/sbcwave.f90:417`), `dta_tsd_init`
always (`SWG_DG/BLD/ppsrc/nemo/istate.f90:102`), `c1d_init` and `dom_clo`
behind `ln_c1d` and `ln_closea` (`SWG_DG/BLD/ppsrc/nemo/domain.f90:192`,
`:196`), `lk_oasis` initialised false (`SWG_DG/BLD/ppsrc/nemo/sbc_oce.f90:52`),
and `nn_hls` raised to at least 2 (`SWG_DG/BLD/ppsrc/nemo/mppini.f90:171`).

Constants no namelist carries: NEMO's density
(`SWG_DG/BLD/ppsrc/nemo/nemogcm.f90:325`), layer depth
(`SWG_DG/BLD/ppsrc/nemo/usrdef_zgr.f90:161`) and closed periodicity
(`SWG_DG/BLD/ppsrc/nemo/usrdef_nam.f90:166`); the card's constants preset
(`constants_config.py:91`, `:94`) and layer thickness
(`nemo_testcase_recipe.py:4245`).

GPU ladder: helpers bind the checkout only in `main`
(`nemo_testcase_l1_swg_dg_round7.py:364`,
`nemo_testcase_l1_swg_dg_composition.py:181`) and fail closed through a local
check (`nemo_testcase_l1_swg_dg_round7.py:19`,
`nemo_testcase_l1_swg_dg_composition.py:20`); the GPU form uses the driver's
case (`nemo_testcase_l1_swg_dg_short_ladder.py:92`,
`run_swg_double_gyre.py:80`, used by the long run at `:333`).

Policy and climate: `nemo_testcase_l1_swg_dg_round7.py:161`,
`nemo_testcase_l1_swg_dg_composition.py:168`,
`nemo_testcase_l1_swg_dg_climate.py:625`. Gradient row skip:
`test_step_gradient_matrix.py:65`, called at `:268`.

Tests named above: `test_swg_dg_reproduction_guards.py:100` (GPU emulation),
`:132` (exhaustive fixture run), `:201` (echo plants), `:261` (registry gap),
`:277` (float64), `:333` (annual and climate float64), `:429` (derivation),
`:477` (live derivation), `:577` (exact pins), `:686` (runbook);
`test_swg_dg_reproduction_shell.py:148` (prefix), `:177` (archiver);
`test_nemo_swg_dg_climate.py:292` (final line);
`test_nemo_swg_card.py:326` (derived groups beyond the frozen thirteen).

## Commands behind the evidence

`py.sh` runs the reference interpreter with this work copy's package roots,
`JAX_PLATFORMS=cpu` and `JAX_ENABLE_X64=1`; `py_nox64.sh` is the same without
`JAX_ENABLE_X64`; `pt.sh` runs pytest on one file that way. All live in SCRATCH.

```bash
# derived inventory on three builds, and the live test on two
"$PY" -I scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_namelist_inventory.py \
  --source-root .../swgdgsa_rounds/round2/nemo_5.0.2_clean_preflight/tests/SWG_DG/BLD/ppsrc/nemo
"$PY" -I scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_namelist_inventory.py \
  --source-root /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/SWG_OMIP_L1/BLD/ppsrc/nemo
SWG_DG_PPSRC=<ppsrc> bash SCRATCH/pt.sh tests/ocean/fidelity/test_swg_dg_reproduction_guards.py -k live_build
# the guard on the reference and both fresh runs
bash SCRATCH/py.sh scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_deck_guard.py \
  --nemo-run-dir .../swg_rounds/maddison_deck/md_dg_dx6p25_dt50_3yr --days 1095 \
  --json SCRATCH/final_guard_reference_3yr.json        # and dg_3yr (1095), dg_smoke (30)
# the review's own plant scripts, unchanged, on copies of the reference run
bash SCRATCH/py.sh .../review_dg_d/guard-and-environment/find/plants.py <work copy> \
  .../md_dg_dx6p25_dt50_3yr SCRATCH/reviewer_plants_on_round_e 1095
python .../review_dg_d/guard-and-environment/verify/plant2.py <work copy> \
  .../md_dg_dx6p25_dt50_3yr SCRATCH/reviewer_plant2_on_round_e 1095
# float64 probe: the annual checker in a shell without JAX_ENABLE_X64
bash SCRATCH/py_nox64.sh scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_restart_identity.py \
  --nemo-root .../md_dg_dx6p25_dt50_3yr \
  --legoesm-npz .../swgdg_rounds/legoesm_runs/dg_gpu_1095d_daily_mean/day0365.npz \
  --day 365 --step 630720 --json SCRATCH/nox64_year1.json
# self-test: 45-pin venv, external evidence hidden, checkout read-only
python SCRATCH/build_restricted_site.py SCRATCH/pinned_site \
  scripts/data/nemo_swg_double_gyre/python-gpu-requirements.txt \
  scripts/data/nemo_swg_double_gyre/python-selftest-requirements.txt
/home/dbalwada/miniconda3/bin/python3.13 -m venv --without-pip SCRATCH/selftest_venv  # + a .pth to pinned_site
unshare -rm --propagation private bash SCRATCH/ns_selftest.sh \
  SCRATCH/selftest_venv/bin/python SCRATCH/selftest_run4.log <the eleven files>
# the SWG gradient row with the evidence hidden, then with its skip reverted
unshare -rm --propagation private bash SCRATCH/ns_pytest.sh <reference python> \
  SCRATCH/matrix_swg_hidden.log tests/ocean/validation/test_step_gradient_matrix.py -k swg_nemo_card_1step
python SCRATCH/revert_matrix_skip_hidden.py
# every other reversion (each edit restored and hash-verified afterwards)
python SCRATCH/reversions.py > SCRATCH/reversions_final2.log
# NEMO tile check on the fresh rebuild, and its one-byte negative control
bash SCRATCH/tile_hash_check.sh .../operator_acquisition/nemo/dg_3yr SCRATCH/tilecheck_fresh
bash SCRATCH/tile_negative_control.sh
# citation gate on this receipt, and a planted shift
LEGOESM_GATE_ALLOW_DIRTY=1 bash SCRATCH/py.sh \
  scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py \
  --receipt docs/ocean/fidelity/testcases/nemo_testcases_l1_swg_dg_reproduction_round_e_receipt.md \
  --from-heading "## Review disposition"
#   ... --plant nemo_testcase_l1_swg_dg_deck_guard.py:242   -> exit 1
```

Decisive outputs: the inventory and guard lines quoted in the table; the
reversion log `reversions_final2.log` (SHA-256 `90757bc1…`) ends every one of
its 24 reversions with `pytest_exit=1` and a restored file; the self-test log
`selftest_run4.log` (SHA-256 `9dceacfb…`); the gate report `PASS` with 61
citations, none unmapped or failing, and all self-test controls firing.

## Package tests, file by file

Run in the reference environment with the external evidence visible
(`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, one pytest process per file):

- tests/ocean/fidelity/test_swg_dg_reproduction_guards.py: 33 passed, 1 skipped in 21.43s (rerun after the last guide and record edits)
- tests/ocean/fidelity/test_swg_dg_reproduction_shell.py: 23 passed in 35.96s
- tests/ocean/fidelity/test_swg_double_gyre_movie.py: 2 passed in 7.78s
- tests/ocean/fidelity/test_run_swg_double_gyre.py: 8 passed in 50.65s
- tests/ocean/fidelity/test_nemo_swg_dg_climate.py: 9 passed in 6.71s
- tests/ocean/fidelity/test_nemo_swg_dg_restart_identity.py: 10 passed in 0.55s
- tests/ocean/fidelity/test_nemo_swg_dg_composition.py: 3 passed in 0.52s
- tests/ocean/fidelity/test_nemo_swg_dg_round6.py: 6 passed in 9.24s
- tests/ocean/fidelity/test_nemo_swg_dg_round7.py: 6 passed in 0.11s
- tests/ocean/fidelity/test_nemo_swg_check_records.py: 22 passed in 0.74s
- tests/ocean/unit/test_nemo_swg_card.py: 157 passed in 8.87s
- tests/ocean/unit/test_nemo_tsunami_card.py: 26 passed in 110.89s (0:01:50)
- tests/ocean/fidelity/test_nemo_tsunami_ladder.py: 13 passed in 129.88s (0:02:09)
- tests/ocean/fidelity/test_nemo_swg_geometry_gate.py: 5 passed in 2.56s
- tests/ocean/fidelity/test_nemo_swg_ladder.py: 26 passed in 49.51s
- tests/ocean/unit/test_pv_flux_ene.py: 12 passed in 19.73s
- tests/ocean/unit/test_swg_swe_rk3_stage.py: 17 passed in 13.96s
- tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py: 15 passed, 2 failed in this uncommitted tree (both failures are the gate's refusal to stamp a dirty worktree); with LEGOESM_GATE_ALLOW_DIRTY=1: 17 passed in 3.85s
- tests/ocean/validation/test_step_gradient_matrix.py: not run in full (its two certified-card rows refuse an uncommitted tree by design); the SWG row with the external evidence hidden: 1 skipped, with the reason quoted above

## Choices made

ASKED means the task or a recorded maintainer decision required it;
UNASKED means the mechanism is this round's choice. Each UNASKED item can be
reverted on request.

- ASKED: fix every review finding; exactness wording per the 2026-10-13
  decision; self-test pins in a separate file; F4 by renaming or a committed
  comparison (renaming chosen, so no agreement bar was invented).
- UNASKED: the derivation rule (static reachability from `nemo_gcm`, every
  call whatever its guard; namelist-buffer units only; the two literal-string
  self-test reads reported as excluded).
- UNASKED: NEMO-only fields are pinned to the reference run's values (any
  change refuses, including in groups this deck switches off).
- UNASKED: new hard refusals that were tolerated before: a different echoed
  group set, an unparsable line or a repeated block in a read group, a build
  whose derived inventory differs, a missing host `ar`, and a missing
  `--conda-prefix` (or one without `mpirun`) on the identity, run and rank
  tools.
- UNASKED: float64 selection moved into the shared bootstrap (all package
  entry points), not only the guard. Consequence, measured on the climate
  tests: a pytest process started without `JAX_ENABLE_X64=1` (the guide and
  the lane battery always set it) now stops at collection with the
  bootstrap's x64 message instead of running those tests in float32.
- UNASKED: the driver's case construction factored into `build_case` (no
  behaviour change; its tests pass) so the GPU ladder scores the same case.
- UNASKED: the eleven-file self-test list; the gradient matrix, TSUNAMI,
  citation-gate, ladder and geometry files stay in the lane battery only.
- UNASKED: committed data: the full reference `namelist_ref` (124 KB) and
  three-year `namelist_cfg` as fixtures (replacing the `nambdy` excerpt), the
  derived inventory, the NEMO-only registry, the two climate reports, the
  self-test record, and all 45 pins plus host requirements in the
  environment JSON.

## UNVERIFIED

- No GPU was used: the GPU short ladder is shown only by CPU emulation, the
  GPU driver check in step 1 was not run, and no reference GPU ladder exists.
- `build_swg_dg.sh --execute` with the new inventory and archiver steps, and
  the identity, rank and run tools with `--conda-prefix`, were exercised only
  in preflight against fake trees; no `makenemo` or `mpirun` ran.
- The launcher has never printed the new guard line after a real NEMO run;
  the guard line was measured by running the guard on finished runs.
- `conda create` and `pip install` from the committed files were not run; the
  self-test environment was assembled from the reference environment's
  installed files.
- The NVIDIA and JAX driver minimums and the cuDNN architecture list are read
  from vendor documentation fetched on 2026-10-14, not measured.
- The derivation over-approximates (guarded calls count) and cannot see calls
  through procedure pointers; it was checked only on this NEMO revision.
- The lane battery was not re-run in full: its two certified gradient rows
  refuse an uncommitted tree by design, so the operator must run it after
  committing.
- The `-rs` skip line quoted in the guide carries a test-file line number that
  moves if that file is edited.

## Still open, and what needs the maintainer

- Name the hand-off commit for colleagues (guide step 1 needs it).
- Update the pull-request body (N1) with the block below.
- After committing: run the lane battery, and on a GPU record a reference GPU
  ladder (`short_ladder.py --backend gpu --gpu-device N --composition
  outer-scan` on the operator's ladder root) so the guide can quote it.
- Optional: run the guide's two `conda create` and two `pip install` commands
  on a clean host.
- Deviation: this round staged the deletion of
  `tests/ocean/fixtures/swg_dg/namelist_ref.nambdy` with `git rm` (an index
  write the session rules did not allow). Nothing was committed.

## PR description must say

- Round E makes the deck guard exhaustive over a derived compiled-read
  inventory: 336 fields in 27 groups; 147 compared with the card (158 rows),
  189 pinned to the reference run with a reason each; it passes the reference
  and both fresh NEMO runs.
- Colleagues get an eleven-file package self-test with its own pin file;
  measured with only the repository: 279 passed, 1 skipped, 177 s, 1.2 GiB.
- The exactness policy is the maintainer's 2026-10-13 decision: exact
  prognostic bar, forcing signed-zero exception, per-step replay last bit as
  recorded debt. The 189 / 1,278 / 414,545 counts are one debt at different
  reductions; 1,278 is reference-only evidence.
- The GPU short ladder now runs past both refusals in emulation; it has never
  run on a GPU.
- The climate tool reports statistics without a verdict; the reference
  statistics of both models are committed.
- Every host dependency found by the review is stated (glibc 2.34, binutils
  `ar`, NVIDIA driver, MPI from the named prefix).
