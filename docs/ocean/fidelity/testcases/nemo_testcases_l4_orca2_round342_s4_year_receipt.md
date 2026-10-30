# ORCA2 round 342 — standalone one-year S4 acquisition tool, legoESM K0 driver, and Decision-185 metric card comparison

Date: 2026-10-28. Base `6b5e8413eb3b`; preregistration `75cc6fec9`.
Disposition: **DECISION_NEEDED** (both one-year arms completed and admitted; full 12-month Decision-160 metric card and 365-day SST RMS difference curve evaluated against Decision-185 reference-only bars; the user decides the bar after seeing this).
This round built and preflighted the observer-free one-year (2,920-step,
365-day) standalone K0 NEMO twin acquisition tool, executed the 20-step
standalone K0 legoESM smoke run, built and smoke-verified the month-by-month
Decision-160 ORCA2X metric card comparison next to the Decision-185 reference
bars, (in Resume 3) fixed the gfortran `stderr` `STOP 0` + `time.step`
completion check and admitted operator-run Arm A, and (in Resume 4) verified the
admitted NEMO one-year twin pair
(`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/orca2_k0_year1_{a,b}_np2`)
and completed 2,920-step legoESM standalone K0 run
(`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_year1_2920step`),
evaluated the Decision-160 metric card month by month across all 12 months
(`M01_Jan..M12_Dec`, `Days 1..365`, `kt = 8..2920`) on both arms (and `NEMO Twin B vs
NEMO Twin A` as the reproducibility control) next to the Decision-185 reference
bars (`REFERENCE ONLY — no pass/fail`), and recorded the day-by-day SST RMS
difference curve and figure data in this receipt.

## Source basis and frozen experiment

### 1. The NEMO arm (observer-free K0 2,920-step twin acquisition)

The standalone K0 NEMO build (`ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y`) reuses the
reviewed Round-339 K0 (`nemo_testcase_l4_orca2_round339_common.py`,
`nemo_testcase_l4_orca2_round339_k0_smoke_gate.py`) and Round-340 S4
(`nemo_testcase_l4_orca2_round340_standalone_gate.py`) helpers with an empty
`MY_SRC` and Decision-175 K0 selectors (`nn_ice = 1`, `ln_drgice_imp = .false.`,
`ln_mxl0 = .true.`, `nn_mxlice = 0`, `ln_icebergs = .false.` at
`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:376-384` and
`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcice_if.f90:149`), reading
the admitted Decision-171/181 yearly-mean ice-cover climatology
`ice_cover_clim.nc` (`sha256 =
2b2d50bbda68bbc934c95f812256834e68631c47f25b5d2d70bf0fade0ba73dd`).

With `nn_it000 = 1`, `nn_itend = 2920`, `nn_stock = 8`, `nn_write = 2920`, and
`ln_rst_list = .false.`, `rst_opn`
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/restart.f90:107-115`) sets
`nitrst = kt + nn_stock - 1` whenever `MOD( kt - 1, nn_stock ) == 0`, and
`rst_write` (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/restart.f90:176-180`),
called from `stp_RK3`
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:241`) after the
Stage-3 time-level swap `Nrhs = Nbb ; Nbb = Naa ; Naa = Nrhs`
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:203`), writes
365 daily two-rank ocean restart pairs
`ORCA2_{kt:08d}_restart_{0000,0001}.nc` at `kt = 8, 16, ..., 2920` containing
the completed end-of-day prognostic state (`sshn, un, vn, tn, sn` from `Kbb`)
required by the Decision-160 ORCA2X metric card, with zero in-loop observers.
In `--run` mode, the launcher runs the two NEMO reproducibility twins
(`orca2_k0_year1_a_np2`, `orca2_k0_year1_b_np2`) and admits their bit-for-bit
reproducibility before any full-year candidate comparison.

### 2. The legoESM arm (production standalone K0 driver and 20-step smoke)

- Driver:
  `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py`.
- Detached launcher:
  `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_legoesm_run.sh`.

The driver runs the landed standalone K0 composed card
(`build_nemo_orca2_k0_ocean_card` and `nemo_orca2_k0_ocean_step` at
`packages/ocean/legoesm/ocean/bulk_flux_omip.py:1006-1030` and
`packages/ocean/legoesm/ocean/bulk_flux_omip.py:1076-1107`) with explicit
`NemoStandaloneK0Config(nn_ice=1, ln_drgice_imp=False, ln_mxl0=True,
nn_mxlice=0, ln_icebergs=False, initial_state='nemo_k0_istate')` verified by
`validate_nemo_standalone_k0` and `_validate_k0_ocean_card`. It extends the
Round-340 S4 `fld_read` schedule across `kt = 1..2920` (all 1,460 6-hourly CORE
records, 365 daily radiation records, 12 cyclic monthly precipitation/snow,
runoff, SSS, chlorophyll, and 3D T/S damping records, and the single-record
Decision-171 `ice_cover_clim.nc` climatology) and writes daily snapshots
`candidate_step{kt:08d}.npz` at `kt = 8, 16, ..., 2920`. The launcher checks
available memory (`>= 150 GiB`), pins the run to the CPU with x64 enabled,
refuses `--steps > 40` unless `--allow-full-year` is passed by the operator,
and launches detached under `nohup`.

**Certified code scope (2026-10-30):** the completed 2,920-step legoESM year
was produced by commit `e1f1b2e7e83088d1e66d0d052383cf36441566a7`, as recorded by its
`run_manifest.json`. The current lane additionally contains commits
`2cc5ff7c89211c0aab2ffd9dcc83773364bedfba` (the Round-341 Stage-2 stored-
thickness change selected by the K0 card in
`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py`, line 1997),
`3178d9a6dc695727ea4ec4e21646c7173ef52917`, and
`e7f884b8e878f57454f05aa8aa0a22bc262a7bdd`. HEAD was not re-run for the
full year. A CPU/x64 exact-mode transfer check at pre-edit HEAD
`fdcd6d6f9d2335e8754cbf64c098bce224cd91f5` ran steps `1..80` into scratch
and compared every saved state (`kt = 8, 10, 16, 24, 32, 40, 48, 56, 64,
72, 80`) with the completed-year directory. All 11 `.npz` files were
byte-identical, so the completed-year result transfers to that HEAD only
through step 80 (10 days); no claim is made for later steps.

The 20-step smoke was executed in
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_smoke_20step`
(`run_manifest.json` SHA-256
`f8b0c0154cfec54f6215c1365f441a0dc2124221df742a957ca8691916db7a65`, wall time
`879.41 s`), producing `candidate_step00000008.npz` (SHA-256
`ce932543cdad48678498811acc2c0625ba668f0043f62c6f8e82a1866ce49daf`),
`candidate_step00000010.npz` (SHA-256
`db59708e0708e24d58ae18ceb8b9d64d2ef0d4f8b1f242b5be9ab074a8337acd`),
`candidate_step00000016.npz` (SHA-256
`cbee45219d6834b7723bc69c6b7c4e44a632a3bf0a879818718fc5a600f4826b`), and
`candidate_step00000020.npz` (SHA-256
`55f6b8fb40a1704b54a1b459f05d7d5b88979e8d60b2dcff49dad6a4ae2f91ae`). Against
the admitted Round-340 S4 NEMO `kt = 9` entry state (`kt = 8` exit) and the
admitted Round-339 K0 NEMO `kt = 10` restart
(`ORCA2_00000010_restart_{0000,0001}.nc`), `candidate_step00000008.npz` and
`candidate_step00000010.npz` match bit-for-bit (`max |diff| = 0.0`, `0 ULP`
across all wet cells on `T, S, u, v, ssh`).

### 3. The comparison (month-by-month ORCA2X metric card next to Decision-185 reference bars)

- Comparison script:
  `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_metrics.py`.

Ported the Decision-160 ORCA2X long-run metric card
(`docs/ocean/fidelity/testcases/orca2x_longrun_design.md`,
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2x_round1_longrun.py`,
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2x_round1_longrun_acquisition.py`,
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2x_round1_pilot_metrics.py`,
and `tests/ocean/fidelity/test_orca2x_round1_pilot_metrics.py`) from the ORCA2X
branch and extended `chlorophyll_at_step` in
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round1_ladder_gate.py`
to `1 <= kt <= 2920`. The round-342 reporter imports the existing
Decision-160 functions directly; it does not compare a copied metric
implementation. The chlorophyll schedule outputs for `kt = 1..10` were
compared and were bit-identical.
`nemo_testcase_l4_orca2_round342_s4_year_metrics.py` reuses the Decision-160
metric card verbatim for both:
1. The NEMO reproducibility twin comparison (`Twin B vs Twin A`), and
2. The legoESM standalone K0 candidate vs NEMO Arm A comparison.

**Sampling schedule and nonlinear metric semantics (Decision 160 card, reused
exactly):**
- **Saved steps (both arms):** The Decision-160 card
  (`docs/ocean/fidelity/testcases/orca2x_longrun_design.md` and
  `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2x_round1_pilot_metrics.py`)
  defines its metrics on the instantaneous completed end-of-day states saved
  every `nn_stock = 8` steps (`kt = 8, 16, 24, ..., 2920` for Days `d = 1..365`
  at `rn_Dt = 10800 s`), without per-timestep online means:
  - NEMO arm (`orca2_k0_year1_{a,b}_np2`): two-rank daily restarts
    `ORCA2_{8*d:08d}_restart_{0000,0001}.nc` containing `tn, sn, un, vn, sshn`
    (`Kbb` after the Stage-3 time-level swap), read via
    `pilot_metrics.read_nemo_day`.
  - legoESM arm: daily snapshots `candidate_step{8*d:08d}.npz` containing
    `T, S, u, v, ssh` (`save_stride = 8`), read via
    `pilot_metrics.read_candidate_day`.
- **Per-saved-state metric evaluation then monthly averaging:** Daily restart or
  snapshot state fields (`T, S, u, v, ssh`) are **never** averaged before
  computing nonlinear metrics (`KE`, `PSI`, `MLD`;
  `state_averaging_before_nonlinear_metrics = False`). Instead, on each saved
  daily state `d` (`kt = 8d`), `pilot_metrics.day_metrics(cand, nemo, geo, mld,
  cover)` computes every linear and nonlinear metric (`SST`, `SSS`, `SSH`,
  depth-band `T`/`S`, surface `KE`, barotropic streamfunction `PSI`, Drake
  Passage transport `DRAKE`, mixed-layer depth `MLD`, and under-ice
  `ICE_SST`/`ICE_SSS`) and its daily difference map, and
  `pilot_metrics.monthly(m_rows, m_maps, geo)` then averages those per-day
  metric values across the saved daily states in each of the 12 no-leap calendar
  months (`[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]` days, summing to
  `365` days), identically for both `Twin B vs Twin A` and `candidate vs NEMO
  Arm A`.

**Known minor (no change):** the metric geometry zeroes the fold-row `vmask`
at 68 faces that are wet in NEMO. Independent recomputation gives a mean-KE
shift of approximately `2e-6`, applied identically on both sides of each
comparison; this round does not alter the metric.

Every monthly metric is reported next to its Decision-185 reference bar:
- Monthly RMS `SST` and depth-band `T` (`0-100m`, `100-1000m`, `>1000m`): `0.01 K`
- Monthly RMS `SSS` and depth-band `S` (`0-100m`, `100-1000m`, `>1000m`): `0.002 psu`
- Monthly RMS `SSH`: `1 mm` (`0.001 m`)
- Surface `KE`: `1%` (`0.01`) of NEMO monthly mean
- Barotropic streamfunction `psi`: RMS `0.1 Sv` and spatial correlation `>= 0.999`
- Drake Passage transport: `0.5 Sv`
- Mixed-layer depth (`MLD`, $\Delta\sigma_0 = 0.01\text{ kg m}^{-3}$): RMS `1 m` (`1.0 m`)

All bars are strictly reference-only (`bar_mode = "DECISION_185_REFERENCE_ONLY"`,
`verdict = None`, `pass_fail_assigned = False`). On the Day-1 (`kt = 8`) smoke
verification in
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/smoke_metrics_comparison/`
(`round342_s4_nemo_twin_b_vs_a_metrics.json` SHA-256
`973a6f1f610eea9f7071fa7753d58671cffb1ef5d90c79f5c7e2769a78e4a23e`,
`round342_s4_candidate_vs_nemo_metrics.json` SHA-256
`2c5246b21fd14204ec1d6f7fb9c5bcbd5ba482ae78f39be177ebc1b8e8ad231b`), both
`Twin B vs Twin A` and `candidate vs NEMO` report `0.0000e+00` RMS (`PSI`
pattern correlation `1.000000`) on `M01_Jan` (`1d` partial month) and
`PENDING_ACQUISITION` on `M02_Feb..M12_Dec`.

## Acquisition checklist disposition (Items 1–16)

1. **Checklist 1–2 (Source basis, observer-free K0 restart path):** NEMO writes
   daily restarts via `rst_opn` and `rst_write`
   (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/restart.f90:107-115` and
   `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/restart.f90:176-180`)
   called from `stp_RK3`
   (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:241`) after
   the Stage-3 `Nbb <-> Naa` swap
   (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:203`). Zero
   in-loop observer files exist (`MY_SRC` is empty).
2. **Checklist 3 (Array extents from `ALLOCATE`):** `verify_allocate_extents`
   verifies `uu(jpi,jpj,jpk,jpt)`, `vv(jpi,jpj,jpk,jpt)`,
   `ts(jpi,jpj,jpk,jpts,jpt)`, `ssh(jpi,jpj,jpt)` in `oce.f90`,
   `tmask/umask/vmask(jpi,jpj,jpk)` in `dom_oce.f90`, and `fr_i(jpi,jpj)` in
   `sbc_oce.f90`. `plant_allocate_mismatch` perturbs `ssh(jpi,jpj,jpt)` to
   `ssh(jpi,jpj,jpk)` and proves refusal.
3. **Checklist 4 & 5 (Bounds-checked Fortran fixture & time-level swap):**
   `verify_restart_fortran_fixture` verifies the literal `MOD( kt - 1, nn_stock ) == 0`,
   `Nbb <-> Naa` swap, and `iom_rstput(..., Kbb)` statements in `restart.f90`
   and `stprk3.f90` and compiles/runs a `-fdefault-real-8 -fcheck=bounds -g -O0`
   Fortran fixture across `kt = 1..2920` with rank-distinct data and `-0.0_wp`
   preservation (`writes=365 first=8 last=2920`). `plant_fixture_wrong_time_level`
   substitutes `Kaa` for `Kbb` after the swap and proves refusal.
4. **Checklist 6 (Real-flag FP instruction census + drift plant):**
   `verify_real_flag_fp_census` parses the real `BLD/Makefile` flags
   (`-fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer -ffree-line-length-none`)
   and audits all `425` compiled `.o` objects (`1,700` functions, `205,193`
   floating-point instructions) plus a real-flag `mpif90` recompile of
   `restart.f90`. `plant_fp_drift` injects `+ 1.0e-30_wp` into `restart.f90`,
   compiles at real `-O3` flags, and proves `compare_fp_census` refuses.
5. **Checklist 7 (Whole-build observer-free equivalence + contamination plant):**
   `verify_same_program_no_observer` checks empty `MY_SRC`, scans all `411`
   preprocessed `ppsrc/nemo/*.f90` files (`413` total `ppsrc` files, `425`
   identical `.o` objects) for observer symbols, and runs
   `common.same_program_no_observer`. `plant_observer_contamination` plants both
   a non-empty `MY_SRC` file and a `CALL r340_s4_surface_write(kstp)` line in
   `stprk3.f90` and proves both refuse.
6. **Checklist 8 (Namelist group validation against `namelist_ref` and `ppsrc`):**
   `verify_namelist_groups_against_ppsrc` validates every staged group and K0 /
   `namrun` variable against both `cfgs/SHARED/namelist_ref` and the `96`
   `NAMELIST /group/` declarations in `BLD/ppsrc/nemo/*.f90`. `plant_namelist_error`
   plants an undeclared `&nam_nonexistent_r342` group and an `E R R O R` block
   in `ocean.output` and proves both refuse.
7. **Checklist 9 (Staged full-year forcing record count check + plant):**
   `verify_staged_full_year_inputs` verifies all 13 NetCDF forcing files have
   full-year record counts (`1460` 6-hourly CORE, `365` daily radiation, `12`
   monthly precip/runoff/SSS/chlorophyll/T/S, `1` yearly `ice_cover_clim.nc`).
   `plant_missing_forcing` replaces `slp.15JUNE2009_fill.nc` with a 5-record
   file and proves refusal.
8. **Checklist 10 (Corrupt-record / ULP / NaN / truncation / Day-1 plants):**
   `run_all_admission_plants` executes six independent plants on a manifested
   twin pair (`1-ULP` twin drift, `NaN` injection, file truncation, Day-1 vs
   admitted R340 `kt = 9` entry mismatch, unproduced `.bin` observer file, and
   manifest SHA-256 mismatch) and proves all six refuse while restoring the runs
   cleanly.
9. **Checklist 11 & 13 (Retention without deletion, rerun-after-refusal
   quarantine, and `refused/ -> --admit-existing`):**
   All `rm -rf` of retained arms are removed from `run.sh`
   (`verify_run_sh_syntax_and_plant` asserts `rm -rf` is absent, and both
   `run_twin_pipeline` and `stage_single_arm` refuse before creating an arm
   directory if `orca2_k0_year1_{a,b}_np2` already exists). Once an arm
   directory is created (`mkdir "${arm_dir}"`), every failure path — including
   an arm-A or arm-B staging failure (`stage_single_arm`), an `mpirun` /
   runtime / timeout failure (`run_single_arm`), or post-run admission failure
   (`--admit-existing`) — routes through `quarantine_arms`, moving any created
   arm directory into
   `<target_root>/refused/<UTC-timestamp>_orca2_k0_year1_{a,b}_np2.failed.<pid>`
   (never deleting any file) so a partial retained arm never blocks a subsequent
   re-run. `admit_existing_with_quarantine` (`--admit-existing <run_root>`)
   supports both in-place admission and the `refused/ -> --admit-existing`
   route: when canonical arm directories are absent and quarantined pairs exist
   in `<run_root>/refused/`, it verifies `producer_manifest.json` inside the
   `refused/` retention trap (`producer_commit` ancestry, `binary_sha256` against
   `./nemo`, and relative file SHA-256 digests) plus full twin admission
   (`admit_s4_year_twin_runs`), promotes the verified arms from `refused/` back
   to canonical `<run_root>/orca2_k0_year1_{a,b}_np2` on pass, keeps them in
   `refused/` on failure, and appends an audit record to
   `<run_root>/admit_existing.log`. Preflight verifies all six routes
   (`in_place_pass_verified`, `existing_arm_overwrite_refused`,
   `stage_failure_quarantine_and_rerun_verified`,
   `refused_quarantine_and_rerun_verified`,
   `refused_to_admit_existing_checksum_plant_verified`, and
   `refused_to_admit_existing_promotion_verified`), including a planted arm-A
   staging failure (`run.sh --stub-run-dir ... --stub-steps 16 --ice-cover
   <missing_ice_cover>`) that proves the partially populated arm-A directory is
   quarantined to `refused/` and a subsequent re-run on the same `<target_root>`
   admits both twins cleanly while preserving the quarantined directory.
10. **Checklist 12 (Stub replaces only `./nemo` and writes only K0 files):**
    Under `--stub-run-dir <stub_root>` (invoked by `--preflight-only`),
    `stage_single_arm` in `run.sh` installs a 2-line wrapper at `${arm_dir}/nemo`
    that `exec`s `nemo_testcase_l4_orca2_round342_s4_year_gate.py
    --act-as-nemo-binary` (with optional `--plant-stub-unproduced-file`,
    `--plant-stub-missing-stop`, `--plant-stub-short-timestep <steps>`, or
    `--plant-stub-sleep <seconds>`), calling `write_stub_s4_year_outputs(Path.cwd(), ...)`
    to read `nn_itend` and `nn_stock` directly from `namelist_cfg` in the staged
    arm directory, write `time.step`
    (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stpctl.f90:106-107` and
    `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stpctl.f90:160-161`),
    emit `"STOP 0"` to `sys.stderr`
    (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/nemogcm.f90:237-238`
    under gfortran), and write only `ocean.output`, `output.namelist.dyn`,
    `time.step`, `mesh_mask_{0000,0001}.nc`, and
    `ORCA2_{kt:08d}_restart_{0000,0001}.nc` (with Day 1 `kt = 8` populated
    from admitted R340 S4 `kt = 9` entry state). `verify_no_unexpected_files`
    refuses any `.bin` or `oracle_*` file.
11. **Checklist 14–16 (No symlinked outputs, disk-space & RAM preflight + plants,
    `timeout --foreground` + plant, clean worktree + plants, gfortran `stderr`
    `STOP 0` + `time.step` completion signal + plants, and `run.sh` syntax
    plant):**
    - All outputs are regular files (`not p.is_symlink()`).
    - `verify_disk_space` checks free space (`351,076.36 GiB` available on
      `/data/abyssal` vs `50.0 GiB` required for the `38.69 GiB` two-twin 365-day
      restart footprint), and `plant_disk_space_exhaustion` plants a `10^9 GiB`
      requirement in both Python (`verify_disk_space(..., required_gib=1.0e9)`)
      and `run.sh` (`R342_REQUIRED_DISK_GIB=1000000000`) and proves both refuse.
    - `verify_available_ram` checks `MemAvailable >= 150.0 GiB` and `free -g`
      column 7 `>= 150 GiB` (`~922 GiB` available, recorded in `ram_check`), and
      `plant_ram_exhaustion` plants a `999,999 GiB` requirement in both Python
      (`verify_available_ram(required_gib=999_999.0)`) and `run.sh`
      (`R342_REQUIRED_RAM_GIB=999999`) and proves both refuse.
    - `run.sh` wraps each arm's `mpirun -np 2 --oversubscribe ./nemo` in
      `timeout --foreground --signal=TERM --kill-after=30s 5400s`, and the
      `timeout_foreground_plant` check in `run_preflight` runs
      `run.sh --stub-run-dir ... --stub-steps 16 --stub-timeout 1
      --plant-stub-sleep 5`, proving `timeout --foreground` terminates the arm
      (`rc = 1` from `run.sh`) and quarantines both staged arms into `refused/`.
    - `verify_clean_worktree` enforces `allow_dirty=False` (`git_sha(allow_dirty=False)`,
      `worktree_stamp(allow_dirty=False)`, and
      `git status --porcelain --untracked-files=all`), and `plant_dirty_worktree`
      plants both an untracked file and a modified tracked file in a temporary
      git repository and proves both Python (`verify_clean_worktree`) and
      `run.sh` (`check_clean_worktree` with `R342_CHECK_REPO_ROOT`) refuse.
    - `verify_nemo_completion_signal` and `run_single_arm` verify NEMO's real
      completion signal where gfortran writes it
      (`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/nemogcm.f90:237-238`,
      `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stpctl.f90:106-107`,
      and `ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stpctl.f90:160-161`):
      `STOP 0` on `stderr` (`mpirun.stderr.log`), `int(time.step) == nn_itend`
      (`2920`), and no `"E R R O R"` or nonzero `STOP` in `ocean.output` (or
      `mpirun.stderr.log` / `mpirun.stdout.log`). `plant_completion_signal_failures`
      and `test_real_arm_a_completion_signal_and_plants` verify the copied real
      Arm-A log files in
      `tests/ocean/fidelity/data/orca2_round342_arm_a_real_logs/` pass and prove
      missing `STOP 0` on `stderr`, `STOP 0` on `stdout` only, `STOP 123`, short
      `time.step` (`2912` / `2919`), missing `time.step`, and `"E R R O R"` in
      `ocean.output` all refuse (plus shell plants `--plant-stub-missing-stop`
      and `--plant-stub-short-timestep 8` in `run.sh`).
    - `verify_run_sh_syntax_and_plant` checks `bash -n` on `run.sh` and plants a
      syntax error (`if then`) in a temporary copy, proving `bash -n` refuses
      (`rc = 2`).

## Resume 3: Kept Arm A admission (`--admit-existing`) and Arm-B-only resume

The operator ran the NEMO 1-year acquisition at `HEAD 5ba77a9d6cd041b929d6982444cd4727c60c791a`
using the compiled observer-free K0 binary
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y/BLD/bin/nemo.exe`
(`sha256 = 835f03b99a3c96e80d23f35c415d099fde292678ae30572e9c5412be4b67e16d`).
Arm A (`orca2_k0_year1_a_np2`) completed all `2,920` steps (`time.step = 2920`,
`365` daily restart pairs `ORCA2_00000008_restart_{0000,0001}.nc` through
`ORCA2_00002920_restart_{0000,0001}.nc`, `mpirun.stderr.log` ending `"STOP 0"`,
and zero `"E R R O R"` lines in `ocean.output`), and was quarantined unmodified
at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/refused/20261027T231902Z_orca2_k0_year1_a_np2.failed.4171609`
because the prior `run.sh` check looked for `STOP 0` in `mpirun.stdout.log`
instead of `mpirun.stderr.log`.

After fixing the completion check in `run.sh` and
`nemo_testcase_l4_orca2_round342_s4_year_gate.py`, kept Arm A was admitted and
promoted (without modifying any file inside the run directory) via:
```bash
bash scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_acquisition/run.sh \
  --admit-existing /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/refused/20261027T231902Z_orca2_k0_year1_a_np2.failed.4171609
```
producing the admission line:
```text
ORCA2_ROUND342_S4_YEAR_ARM_A_ADMITTED /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/orca2_k0_year1_a_np2 producer_commit=5ba77a9d6cd041b929d6982444cd4727c60c791a binary_sha256=835f03b99a3c96e80d23f35c415d099fde292678ae30572e9c5412be4b67e16d daily_restart_pairs=365 time_step=2920
```
- Promoted canonical Arm A path:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/orca2_k0_year1_a_np2`
- Admission evidence JSON:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/round342_s4_year_arm_a_admission.json`
  (and `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/round342_s4_year_arm_a_admission.json`).
- Day 1 (`kt = 8`) restart vs admitted Round-340 S4 `kt = 9` entry state:
  `max_abs_diff = 0.0`, `max_ulp = 0` across all wet cells on `T, S, u, v, ssh`.
- Day 1 (`kt = 8`) restart SHA-256:
  `rank_0000 = 3fbed2281e66f7d51cd34247e799d1b76c8a102d36d5b64c0a3b80aba56dd2e3`,
  `rank_0001 = a5660f4c44b70a7cf7d70500b2df8e2d2145f08a205b266301c7312517413023`.
- Day 365 (`kt = 2920`) restart SHA-256:
  `rank_0000 = c42e8d90ba488b2074f3b0e5651aeefcdb21ffb3ae2791c95c745917cb56a216`,
  `rank_0001 = c362626213814f1738fed7fd6004618e00366bf71b2eeb97061aae150f0980e8`.
- When the operator runs `run.sh --run`, `build_observer_free_k0_nemo` reuses
  the verified `ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y/BLD/bin/nemo.exe` binary and
  `run_twin_pipeline` detects admitted Arm A, prints `R342_RESUMING_ARM_B_ONLY`,
  stages and executes **Arm B only** (`orca2_k0_year1_b_np2`), and admits the
  full twin pair `(orca2_k0_year1_a_np2, orca2_k0_year1_b_np2)`.

## Resume 4: Full 1-year (`2,920`-step, `365`-day) evaluation on both arms (`DECISION_NEEDED`)

### 1. Admitted NEMO one-year twin pair (`orca2_k0_year1_a_np2`, `orca2_k0_year1_b_np2`)

- **Run root:** `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year`
- **Admission evidence JSON:**
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/round342_s4_year_admission.json`
  (mirrored at `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/round342_s4_year_admission.json`;
  SHA-256 `8a9eeb2bd0297d2489a8e8b0787f318e552cac57efca1bcea4c9c1ee4f06d642`).
- **Admission status:** `PASS_R342_S4_YEAR_NEMO_TWINS` (`admitted = true`, `daily_restart_pairs = 365`, `expected_steps = 2920`).
- **Completion signals:** Both `orca2_k0_year1_a_np2` and `orca2_k0_year1_b_np2` completed with `time_step = 2920`, `stderr_stop_0 = true` (`stderr_last_line = "STOP 0"`), `stdout_bytes = 0`, and `ocean_output_error_free = true`.
- **Twin reproducibility (`Twin B vs Twin A`):** Bit-for-bit identical across all `365` daily restart pairs (`kt = 8, 16, ..., 2920`):
  - Day 1 (`kt = 8`): `rank_0000` SHA-256 `3fbed2281e66f7d51cd34247e799d1b76c8a102d36d5b64c0a3b80aba56dd2e3`, `rank_0001` SHA-256 `a5660f4c44b70a7cf7d70500b2df8e2d2145f08a205b266301c7312517413023` (identical in A and B).
  - Day 365 (`kt = 2920`): `rank_0000` SHA-256 `c42e8d90ba488b2074f3b0e5651aeefcdb21ffb3ae2791c95c745917cb56a216`, `rank_0001` SHA-256 `c362626213814f1738fed7fd6004618e00366bf71b2eeb97061aae150f0980e8` (identical in A and B).
  - Every Decision-160 metric on `Twin B vs Twin A` has `0.0000e+00` bias and `0.0000e+00` RMS (`PSI` pattern correlation `1.000000`) in every month `M01_Jan..M12_Dec` and on every day `d = 1..365`.

### 2. Completed legoESM one-year standalone K0 run (`legoesm_year1_2920step`)

- **Run directory:** `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_year1_2920step`
- **Log completion check:** `legoesm_run.log` ended cleanly at `saved kt=2920/2920 (candidate_step00002920.npz) elapsed=70274.4s` with terminal JSON `{"status": "PASS_R342_S4_K0_LEGOESM_RUN", "steps": 2920, "wall_seconds": 70274.35, "jax_default_backend": "cpu", "snapshot_count": 366}`.
- **Manifest and final snapshot digests:**
  - `run_manifest.json` SHA-256: `e94be761e490af86e2df7e2cb18c772406160c02c8593c782e021605a2221bbd`
  - `candidate_step00000008.npz` (Day 1, `kt = 8`) SHA-256: `ce932543cdad48678498811acc2c0625ba668f0043f62c6f8e82a1866ce49daf`
  - `candidate_step00002920.npz` (Day 365, `kt = 2920`) SHA-256: `e5d9237e4cc27b8910eee616d92af5399ffd4319d04ed8b29fe12e9c3ce85597`

### 3. Full-year Decision-160 metric card and figures provenance

- **Output directory:** `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/year1_metrics_comparison`
- **JSON & Markdown artifacts:**
  - `round342_s4_candidate_vs_nemo_metrics.json`: SHA-256 `7ec864d4da7fb28a67671ab895405e43a00124e5ef5ed8a497b1d1ccbb4c85ec`
  - `round342_s4_candidate_vs_nemo_metrics.md`: SHA-256 `3119e7451622538c8d0acadf810b2f67b6a1e4d7e86fa48d07f5c47064d00f81`
  - `round342_s4_nemo_twin_b_vs_a_metrics.json`: SHA-256 `fc41c985995a254b7e36e933ea8477ad69ae53d9f653db1a91c624b5c7d2f5b8`
  - `round342_s4_nemo_twin_b_vs_a_metrics.md`: SHA-256 `9a9d604a16f88dd034740a4a816ce9aa81772f93d336a3ec375244e1a9938f2b`
  - `round342_s4_daily_sst_rms_curve.json`: SHA-256 `2e65e6a175964b6f5e3aabf7465ca3df174772bf5a3422872b3ffa02de8f5f14`
- **Figures:**
  - `sst_rms_daily_curve.png`: SHA-256 `41ec92192d6ef8e2c83fcf5db71e8176a1b4a555aee08c47c4ec8cf4c725ca38`
  - `candidate_vs_nemo_plots/timeseries_bias_rms.png`: SHA-256 `6103a98a44248d012862ad676fc30eadba8d8f3c0c4eebffb471a771d882c2a5`
  - `candidate_vs_nemo_plots/drake_transport.png`: SHA-256 `b43ded92418968bd29e215df05f9d1f8c83a5a31d920b8de736a565cf5b8ff07`
  - `candidate_vs_nemo_plots/profiles_T_S.png`: SHA-256 `b288a278154b73d2a09dec452dfe3d4d1312a634bd8b223e4d135b1ee12e3757`
  - `candidate_vs_nemo_plots/monthly_mean_difference_maps.png`: SHA-256 `ef51d8d887684281d699b1b8189b78ef04649a61a51e9d894882f2023b1949e8`
  - `twin_b_vs_a_plots/timeseries_bias_rms.png`: SHA-256 `8e9bfaffd5c47d297529d65115c35dbcd1f843e67b0c428d7b6479fba59972ff`
  - `twin_b_vs_a_plots/drake_transport.png`: SHA-256 `e0fcad21a75ca9101a7984f48793261a229f275423165d1285b9ba77fe4d09c1`
  - `twin_b_vs_a_plots/profiles_T_S.png`: SHA-256 `9a8cc5b8ca462c2b0710f81933fdd7589767384cb640601272c2dd433ceb1bf0`
  - `twin_b_vs_a_plots/monthly_mean_difference_maps.png`: SHA-256 `3db418d0bb885fb0b3e43b096a06793689bc4e094ea0bf80f9a4b40ba0ab5cc8`

### 4. Trajectory separation milestones (from `round342_s4_daily_sst_rms_curve.json`)

- **Days `1..43` (`kt = 8..344`, entire `M01_Jan` plus first 12 days of `M02_Feb`):** Exact `0.0000e+00 K` `SST RMS` (`last_exact_zero_day = 43`).
- **Day `44` (`kt = 352`):** First nonzero `SST RMS` appears (`1.1151e-11 K`, `bias = -6.7177e-14 K`); the first differing wet SST cell is in the Arctic, with a maximum absolute difference of approximately `5e-9 K` (**CONFIRMED**).
- **Day `112` (`kt = 896`, `M04_Apr`):** Daily `SST RMS` jumps from `3.4579e-10 K` on Day 111 to `5.6067e-07 K` (approximately `1600x`). The event is at Nordic Seas level `k = 12`; its cause is unattributed (**CONFIRMED location and magnitude; attribution unknown**).
- **Days `44..248` (`kt = 352..1984`, `M02_Feb` through mid-`M09_Sep`):** `SST RMS` stays in the `1e-11 K` to `5.6e-07 K` range (`M02_Feb = 5.4642e-11 K`, `M03_Mar = 2.1069e-10 K`, `M04_Apr = 9.2933e-08 K`, `M05_May = 7.2534e-08 K`, `M06_Jun = 3.2197e-08 K`, `M07_Jul = 2.2401e-08 K`, `M08_Aug = 7.2797e-08 K`).
- **Day `249` (`kt = 1992`, `M09_Sep`):** Separation crosses `1e-6 K` (`SST RMS = 1.1999e-06 K`). Interpreting the observed growth as exponential chaotic separation is **PLAUSIBLE**, not demonstrated.
- **Day `279` (`kt = 2232`, `M10_Oct`):** Crosses `1e-5 K` (`SST RMS = 1.0268e-05 K`).
- **Day `297` (`kt = 2376`, `M10_Oct`):** Crosses `1e-4 K` (`SST RMS = 1.6257e-04 K`).
- **Day `324` (`kt = 2592`, `M11_Nov`):** Crosses `1e-3 K` (`1 mK`, `SST RMS = 1.0583e-03 K`).
- **Day `365` (`kt = 2920`, end of `M12_Dec`):** Reaches `3.9232e-03 K` (`3.92 mK`; `M12_Dec` monthly mean of daily `SST RMS = 1.8368e-03 K`), remaining below the `0.01 K` (`10 mK`) Decision-185 reference bar throughout Year 1 (`first_day_ge_0.01_k_ref_bar = null`).

### Month-by-Month Summary (Evaluated & Pending Months 1–12)

| Month | Days (`kt`) | Status | SST RMS (`0.01 K`) | SSS RMS (`0.002 psu`) | SSH RMS (`0.001 m`) | KE rel RMS (`0.01`) | PSI RMS (`0.1 Sv`) | PSI corr (`0.999`) | Drake `|diff|` (`0.5 Sv`) | MLD RMS (`1.0 m`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M01_Jan` | `1..31` (`31d`, `kt=8..248`) | `EVALUATED_COMPLETE_MONTH` | `0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `1.000000` | `0.0000e+00` | `0.0000e+00` |
| `M02_Feb` | `32..59` (`28d`, `kt=256..472`) | `EVALUATED_COMPLETE_MONTH` | `5.4642e-11` | `2.5010e-11` | `2.1439e-12` | `9.5620e-11` | `1.7408e-10` | `1.000000` | `1.3046e-11` | `1.0742e-12` |
| `M03_Mar` | `60..90` (`31d`, `kt=480..720`) | `EVALUATED_COMPLETE_MONTH` | `2.1069e-10` | `8.9433e-11` | `5.7041e-12` | `8.1798e-10` | `3.9657e-10` | `1.000000` | `1.2278e-11` | `2.6923e-12` |
| `M04_Apr` | `91..120` (`30d`, `kt=728..960`) | `EVALUATED_COMPLETE_MONTH` | `9.2933e-08` | `4.5327e-09` | `2.5839e-10` | `3.2995e-09` | `1.6213e-08` | `1.000000` | `3.5183e-10` | `5.6797e-11` |
| `M05_May` | `121..151` (`31d`, `kt=968..1208`) | `EVALUATED_COMPLETE_MONTH` | `7.2534e-08` | `3.8584e-09` | `5.1005e-10` | `2.5481e-09` | `1.2448e-08` | `1.000000` | `2.9840e-10` | `5.2974e-11` |
| `M06_Jun` | `152..181` (`30d`, `kt=1216..1448`) | `EVALUATED_COMPLETE_MONTH` | `3.2197e-08` | `2.1403e-09` | `3.9972e-10` | `9.7334e-09` | `9.2615e-09` | `1.000000` | `1.6408e-10` | `2.8645e-11` |
| `M07_Jul` | `182..212` (`31d`, `kt=1456..1696`) | `EVALUATED_COMPLETE_MONTH` | `2.2401e-08` | `2.0273e-09` | `7.4963e-10` | `1.1447e-07` | `1.3547e-07` | `1.000000` | `1.0764e-08` | `3.7524e-10` |
| `M08_Aug` | `213..243` (`31d`, `kt=1704..1944`) | `EVALUATED_COMPLETE_MONTH` | `7.2797e-08` | `8.8039e-09` | `1.5256e-09` | `7.6776e-07` | `1.9063e-07` | `1.000000` | `2.2698e-10` | `1.1965e-10` |
| `M09_Sep` | `244..273` (`30d`, `kt=1952..2184`) | `EVALUATED_COMPLETE_MONTH` | `1.2176e-06` | `1.7826e-07` | `2.8938e-08` | `2.6068e-05` | `4.3547e-06` | `1.000000` | `5.1503e-08` | `2.1952e-10` |
| `M10_Oct` | `274..304` (`31d`, `kt=2192..2432`) | `EVALUATED_COMPLETE_MONTH` | `7.0776e-05` | `1.4015e-05` | `9.4832e-07` | `1.4613e-03` | `7.1743e-05` | `1.000000` | `2.0399e-06` | `3.5275e-02` |
| `M11_Nov` | `305..334` (`30d`, `kt=2440..2672`) | `EVALUATED_COMPLETE_MONTH` | `8.1962e-04` | `1.0454e-04` | `6.8366e-06` | `1.2823e-02` | `7.5224e-04` | `1.000000` | `1.8753e-06` | `1.2348e-01` |
| `M12_Dec` | `335..365` (`31d`, `kt=2680..2920`) | `EVALUATED_COMPLETE_MONTH` | `1.8368e-03` | `2.0464e-04` | `1.6333e-05` | `3.1441e-02` | `2.6756e-03` | `1.000000` | `3.0384e-05` | `2.0259e-01` |

### Month-by-Month and Metric-by-Metric Table (NEMO Value, legoESM Value, Difference, Reproducibility Control, Decision-185 Reference Bar)

| Month | Metric | NEMO Value | legoESM Value | Difference (`legoESM - NEMO`) | NEMO Twin B vs Twin A Control | Decision-185 Reference Bar (REFERENCE ONLY) |
|---|---|---:|---:|---|---|---|
| `M01_Jan` | `SST` | `18.088300 degC` (mean) | `18.088300 degC` (mean) | bias `+0.0000e+00 K`, RMS `0.0000e+00 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M01_Jan` | `T_0-100m` | `16.947222 degC` (mean) | `16.947222 degC` (mean) | bias `+0.0000e+00 K`, RMS `0.0000e+00 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M01_Jan` | `T_100-1000m` | `7.682240 degC` (mean) | `7.682240 degC` (mean) | bias `+0.0000e+00 K`, RMS `0.0000e+00 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M01_Jan` | `T_>1000m` | `1.939377 degC` (mean) | `1.939377 degC` (mean) | bias `+0.0000e+00 K`, RMS `0.0000e+00 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M01_Jan` | `SSS` | `34.716287 psu` (mean) | `34.716287 psu` (mean) | bias `+0.0000e+00 psu`, RMS `0.0000e+00 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M01_Jan` | `S_0-100m` | `34.855525 psu` (mean) | `34.855525 psu` (mean) | bias `+0.0000e+00 psu`, RMS `0.0000e+00 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M01_Jan` | `S_100-1000m` | `34.727337 psu` (mean) | `34.727337 psu` (mean) | bias `+0.0000e+00 psu`, RMS `0.0000e+00 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M01_Jan` | `S_>1000m` | `34.719706 psu` (mean) | `34.719706 psu` (mean) | bias `+0.0000e+00 psu`, RMS `0.0000e+00 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M01_Jan` | `SSH` | `+3.844649e-04 m` (mean; spatial RMS `0.630730 m`) | `+3.844649e-04 m` (mean; spatial RMS `0.630730 m`) | bias `+0.0000e+00 m`, RMS `0.0000e+00 m` | RMS `0.0000e+00 m` | `0.001 m` (`1 mm` RMS; REFERENCE ONLY) |
| `M01_Jan` | `KE` | `1.429658e-02 m^2 s^-2` (mean) | `1.429658e-02 m^2 s^-2` (mean) | bias `+0.0000e+00`, RMS `0.0000e+00 m^2 s^-2` (rel RMS `0.0000e+00`, rel `|mean diff|` `0.0000e+00`) | rel RMS `0.0000e+00` (RMS `0.0000e+00`) | `0.01` (`1%` of NEMO mean = `1.4297e-04 m^2 s^-2`; REFERENCE ONLY) |
| `M01_Jan` | `PSI` | `152.6070 Sv` (spatial RMS; mean `-136.6060 Sv`) | `152.6070 Sv` (spatial RMS; mean `-136.6060 Sv`) | bias `+0.0000e+00 Sv`, RMS `0.0000e+00 Sv`, corr `1.000000` | RMS `0.0000e+00 Sv`, corr `1.000000` | `0.1 Sv` RMS & `>= 0.999` corr (REFERENCE ONLY) |
| `M01_Jan` | `DRAKE` | `136.728372 Sv` (std `63.2582 Sv`) | `136.728372 Sv` (std `63.2582 Sv`) | `cand - NEMO = +0.0000e+00 Sv`, `|mean diff| = 0.0000e+00 Sv` | `|mean diff| = 0.0000e+00 Sv` | `0.5 Sv` (REFERENCE ONLY) |
| `M01_Jan` | `MLD` | `40.716709 m` (mean) | `40.716709 m` (mean) | bias `+0.0000e+00 m`, RMS `0.0000e+00 m` | RMS `0.0000e+00 m` | `1.0 m` RMS (REFERENCE ONLY) |
| `M01_Jan` | `ICE_SST` | `0.292412 degC` (mean) | `0.292412 degC` (mean) | bias `+0.0000e+00 K`, RMS `0.0000e+00 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M01_Jan` | `ICE_SSS` | `33.265310 psu` (mean) | `33.265310 psu` (mean) | bias `+0.0000e+00 psu`, RMS `0.0000e+00 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M02_Feb` | `SST` | `18.260067 degC` (mean) | `18.260067 degC` (mean) | bias `+1.0261e-13 K`, RMS `5.4642e-11 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M02_Feb` | `T_0-100m` | `17.043571 degC` (mean) | `17.043571 degC` (mean) | bias `+8.6546e-14 K`, RMS `6.7943e-11 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M02_Feb` | `T_100-1000m` | `7.683801 degC` (mean) | `7.683801 degC` (mean) | bias `-4.3988e-15 K`, RMS `3.5125e-11 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M02_Feb` | `T_>1000m` | `1.939463 degC` (mean) | `1.939463 degC` (mean) | bias `+2.2482e-16 K`, RMS `1.9818e-13 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M02_Feb` | `SSS` | `34.729136 psu` (mean) | `34.729136 psu` (mean) | bias `+2.5599e-13 psu`, RMS `2.5010e-11 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M02_Feb` | `S_0-100m` | `34.856438 psu` (mean) | `34.856438 psu` (mean) | bias `+1.8917e-13 psu`, RMS `2.1400e-11 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M02_Feb` | `S_100-1000m` | `34.727043 psu` (mean) | `34.727043 psu` (mean) | bias `-3.2952e-15 psu`, RMS `6.5279e-12 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M02_Feb` | `S_>1000m` | `34.719762 psu` (mean) | `34.719762 psu` (mean) | bias `+2.6518e-17 psu`, RMS `2.3595e-14 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M02_Feb` | `SSH` | `+2.427161e-03 m` (mean; spatial RMS `0.642387 m`) | `+2.427161e-03 m` (mean; spatial RMS `0.642387 m`) | bias `+4.3837e-15 m`, RMS `2.1439e-12 m` | RMS `0.0000e+00 m` | `0.001 m` (`1 mm` RMS; REFERENCE ONLY) |
| `M02_Feb` | `KE` | `1.466139e-02 m^2 s^-2` (mean) | `1.466139e-02 m^2 s^-2` (mean) | bias `+6.5114e-15`, RMS `1.4019e-12 m^2 s^-2` (rel RMS `9.5620e-11`, rel `|mean diff|` `4.4417e-13`) | rel RMS `0.0000e+00` (RMS `0.0000e+00`) | `0.01` (`1%` of NEMO mean = `1.4661e-04 m^2 s^-2`; REFERENCE ONLY) |
| `M02_Feb` | `PSI` | `154.9466 Sv` (spatial RMS; mean `-147.0498 Sv`) | `154.9466 Sv` (spatial RMS; mean `-147.0498 Sv`) | bias `+1.3143e-11 Sv`, RMS `1.7408e-10 Sv`, corr `1.000000` | RMS `0.0000e+00 Sv`, corr `1.000000` | `0.1 Sv` RMS & `>= 0.999` corr (REFERENCE ONLY) |
| `M02_Feb` | `DRAKE` | `159.521905 Sv` (std `3.3459 Sv`) | `159.521905 Sv` (std `3.3459 Sv`) | `cand - NEMO = -1.3046e-11 Sv`, `|mean diff| = 1.3046e-11 Sv` | `|mean diff| = 0.0000e+00 Sv` | `0.5 Sv` (REFERENCE ONLY) |
| `M02_Feb` | `MLD` | `46.578939 m` (mean) | `46.578939 m` (mean) | bias `-2.1109e-14 m`, RMS `1.0742e-12 m` | RMS `0.0000e+00 m` | `1.0 m` RMS (REFERENCE ONLY) |
| `M02_Feb` | `ICE_SST` | `0.205558 degC` (mean) | `0.205558 degC` (mean) | bias `+6.1895e-13 K`, RMS `1.0659e-10 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M02_Feb` | `ICE_SSS` | `33.319260 psu` (mean) | `33.319260 psu` (mean) | bias `-6.3289e-16 psu`, RMS `2.2160e-11 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M03_Mar` | `SST` | `18.314550 degC` (mean) | `18.314550 degC` (mean) | bias `-3.7820e-13 K`, RMS `2.1069e-10 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M03_Mar` | `T_0-100m` | `17.137562 degC` (mean) | `17.137562 degC` (mean) | bias `-5.1288e-13 K`, RMS `2.4682e-10 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M03_Mar` | `T_100-1000m` | `7.684494 degC` (mean) | `7.684494 degC` (mean) | bias `-6.9743e-15 K`, RMS `1.4116e-10 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M03_Mar` | `T_>1000m` | `1.939450 degC` (mean) | `1.939450 degC` (mean) | bias `+5.8307e-15 K`, RMS `1.1338e-12 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M03_Mar` | `SSS` | `34.735066 psu` (mean) | `34.735066 psu` (mean) | bias `-5.1028e-13 psu`, RMS `8.9433e-11 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M03_Mar` | `S_0-100m` | `34.857875 psu` (mean) | `34.857875 psu` (mean) | bias `+1.3242e-13 psu`, RMS `7.3257e-11 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M03_Mar` | `S_100-1000m` | `34.726770 psu` (mean) | `34.726770 psu` (mean) | bias `+2.8940e-14 psu`, RMS `2.4882e-11 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M03_Mar` | `S_>1000m` | `34.719791 psu` (mean) | `34.719791 psu` (mean) | bias `+1.0380e-16 psu`, RMS `1.2945e-13 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M03_Mar` | `SSH` | `+5.462414e-03 m` (mean; spatial RMS `0.643633 m`) | `+5.462414e-03 m` (mean; spatial RMS `0.643633 m`) | bias `+2.0682e-14 m`, RMS `5.7041e-12 m` | RMS `0.0000e+00 m` | `0.001 m` (`1 mm` RMS; REFERENCE ONLY) |
| `M03_Mar` | `KE` | `1.350217e-02 m^2 s^-2` (mean) | `1.350217e-02 m^2 s^-2` (mean) | bias `-6.5259e-14`, RMS `1.1044e-11 m^2 s^-2` (rel RMS `8.1798e-10`, rel `|mean diff|` `4.8332e-12`) | rel RMS `0.0000e+00` (RMS `0.0000e+00`) | `0.01` (`1%` of NEMO mean = `1.3502e-04 m^2 s^-2`; REFERENCE ONLY) |
| `M03_Mar` | `PSI` | `156.4356 Sv` (spatial RMS; mean `-148.4535 Sv`) | `156.4356 Sv` (spatial RMS; mean `-148.4535 Sv`) | bias `+1.2023e-11 Sv`, RMS `3.9657e-10 Sv`, corr `1.000000` | RMS `0.0000e+00 Sv`, corr `1.000000` | `0.1 Sv` RMS & `>= 0.999` corr (REFERENCE ONLY) |
| `M03_Mar` | `DRAKE` | `160.552270 Sv` (std `5.3319 Sv`) | `160.552270 Sv` (std `5.3319 Sv`) | `cand - NEMO = +1.2278e-11 Sv`, `|mean diff| = 1.2278e-11 Sv` | `|mean diff| = 0.0000e+00 Sv` | `0.5 Sv` (REFERENCE ONLY) |
| `M03_Mar` | `MLD` | `48.415007 m` (mean) | `48.415007 m` (mean) | bias `-4.2487e-14 m`, RMS `2.6923e-12 m` | RMS `0.0000e+00 m` | `1.0 m` RMS (REFERENCE ONLY) |
| `M03_Mar` | `ICE_SST` | `0.068083 degC` (mean) | `0.068083 degC` (mean) | bias `+6.7545e-12 K`, RMS `3.9982e-10 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M03_Mar` | `ICE_SSS` | `33.332300 psu` (mean) | `33.332300 psu` (mean) | bias `+1.6748e-12 psu`, RMS `6.9082e-11 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M04_Apr` | `SST` | `18.290390 degC` (mean) | `18.290390 degC` (mean) | bias `+9.3979e-10 K`, RMS `9.2933e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M04_Apr` | `T_0-100m` | `17.162195 degC` (mean) | `17.162195 degC` (mean) | bias `+7.3619e-10 K`, RMS `7.7286e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M04_Apr` | `T_100-1000m` | `7.687595 degC` (mean) | `7.687595 degC` (mean) | bias `-9.0649e-11 K`, RMS `3.9743e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M04_Apr` | `T_>1000m` | `1.939325 degC` (mean) | `1.939325 degC` (mean) | bias `+1.2372e-14 K`, RMS `3.0567e-11 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M04_Apr` | `SSS` | `34.737782 psu` (mean) | `34.737782 psu` (mean) | bias `+4.5646e-11 psu`, RMS `4.5327e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M04_Apr` | `S_0-100m` | `34.860181 psu` (mean) | `34.860181 psu` (mean) | bias `+3.3413e-11 psu`, RMS `3.5933e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M04_Apr` | `S_100-1000m` | `34.726462 psu` (mean) | `34.726462 psu` (mean) | bias `-3.8167e-12 psu`, RMS `1.4380e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M04_Apr` | `S_>1000m` | `34.719796 psu` (mean) | `34.719796 psu` (mean) | bias `+1.5100e-15 psu`, RMS `6.3281e-13 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M04_Apr` | `SSH` | `+7.571103e-03 m` (mean; spatial RMS `0.643695 m`) | `+7.571103e-03 m` (mean; spatial RMS `0.643695 m`) | bias `-1.1002e-12 m`, RMS `2.5839e-10 m` | RMS `0.0000e+00 m` | `0.001 m` (`1 mm` RMS; REFERENCE ONLY) |
| `M04_Apr` | `KE` | `1.270240e-02 m^2 s^-2` (mean) | `1.270240e-02 m^2 s^-2` (mean) | bias `+3.6177e-13`, RMS `4.1912e-11 m^2 s^-2` (rel RMS `3.2995e-09`, rel `|mean diff|` `2.8481e-11`) | rel RMS `0.0000e+00` (RMS `0.0000e+00`) | `0.01` (`1%` of NEMO mean = `1.2702e-04 m^2 s^-2`; REFERENCE ONLY) |
| `M04_Apr` | `PSI` | `153.9012 Sv` (spatial RMS; mean `-145.8031 Sv`) | `153.9012 Sv` (spatial RMS; mean `-145.8031 Sv`) | bias `-7.1651e-10 Sv`, RMS `1.6213e-08 Sv`, corr `1.000000` | RMS `0.0000e+00 Sv`, corr `1.000000` | `0.1 Sv` RMS & `>= 0.999` corr (REFERENCE ONLY) |
| `M04_Apr` | `DRAKE` | `159.237683 Sv` (std `4.9887 Sv`) | `159.237683 Sv` (std `4.9887 Sv`) | `cand - NEMO = +3.5183e-10 Sv`, `|mean diff| = 3.5183e-10 Sv` | `|mean diff| = 0.0000e+00 Sv` | `0.5 Sv` (REFERENCE ONLY) |
| `M04_Apr` | `MLD` | `46.737481 m` (mean) | `46.737481 m` (mean) | bias `-4.4046e-13 m`, RMS `5.6797e-11 m` | RMS `0.0000e+00 m` | `1.0 m` RMS (REFERENCE ONLY) |
| `M04_Apr` | `ICE_SST` | `-0.085517 degC` (mean) | `-0.085517 degC` (mean) | bias `+6.5676e-09 K`, RMS `2.4558e-07 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M04_Apr` | `ICE_SSS` | `33.350249 psu` (mean) | `33.350249 psu` (mean) | bias `+3.1698e-10 psu`, RMS `1.1852e-08 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M05_May` | `SST` | `18.168726 degC` (mean) | `18.168726 degC` (mean) | bias `+9.6730e-10 K`, RMS `7.2534e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M05_May` | `T_0-100m` | `17.100109 degC` (mean) | `17.100109 degC` (mean) | bias `+8.8740e-10 K`, RMS `6.6198e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M05_May` | `T_100-1000m` | `7.690713 degC` (mean) | `7.690713 degC` (mean) | bias `-1.3703e-10 K`, RMS `2.5164e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M05_May` | `T_>1000m` | `1.939239 degC` (mean) | `1.939239 degC` (mean) | bias `-8.8176e-14 K`, RMS `9.7508e-11 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M05_May` | `SSS` | `34.740541 psu` (mean) | `34.740541 psu` (mean) | bias `+4.9838e-11 psu`, RMS `3.8584e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M05_May` | `S_0-100m` | `34.862327 psu` (mean) | `34.862327 psu` (mean) | bias `+4.6300e-11 psu`, RMS `3.5144e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M05_May` | `S_100-1000m` | `34.726057 psu` (mean) | `34.726057 psu` (mean) | bias `-4.8928e-12 psu`, RMS `9.9537e-10 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M05_May` | `S_>1000m` | `34.719799 psu` (mean) | `34.719799 psu` (mean) | bias `-1.8813e-15 psu`, RMS `2.3168e-12 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M05_May` | `SSH` | `+9.509606e-03 m` (mean; spatial RMS `0.643066 m`) | `+9.509606e-03 m` (mean; spatial RMS `0.643066 m`) | bias `-8.8658e-12 m`, RMS `5.1005e-10 m` | RMS `0.0000e+00 m` | `0.001 m` (`1 mm` RMS; REFERENCE ONLY) |
| `M05_May` | `KE` | `1.290203e-02 m^2 s^-2` (mean) | `1.290203e-02 m^2 s^-2` (mean) | bias `+2.7876e-13`, RMS `3.2875e-11 m^2 s^-2` (rel RMS `2.5481e-09`, rel `|mean diff|` `2.1606e-11`) | rel RMS `0.0000e+00` (RMS `0.0000e+00`) | `0.01` (`1%` of NEMO mean = `1.2902e-04 m^2 s^-2`; REFERENCE ONLY) |
| `M05_May` | `PSI` | `156.5630 Sv` (spatial RMS; mean `-148.4289 Sv`) | `156.5630 Sv` (spatial RMS; mean `-148.4289 Sv`) | bias `+2.1178e-09 Sv`, RMS `1.2448e-08 Sv`, corr `1.000000` | RMS `0.0000e+00 Sv`, corr `1.000000` | `0.1 Sv` RMS & `>= 0.999` corr (REFERENCE ONLY) |
| `M05_May` | `DRAKE` | `161.155883 Sv` (std `6.9123 Sv`) | `161.155883 Sv` (std `6.9123 Sv`) | `cand - NEMO = +2.9840e-10 Sv`, `|mean diff| = 2.9840e-10 Sv` | `|mean diff| = 0.0000e+00 Sv` | `0.5 Sv` (REFERENCE ONLY) |
| `M05_May` | `MLD` | `47.807162 m` (mean) | `47.807162 m` (mean) | bias `+2.1435e-12 m`, RMS `5.2974e-11 m` | RMS `0.0000e+00 m` | `1.0 m` RMS (REFERENCE ONLY) |
| `M05_May` | `ICE_SST` | `-0.151222 degC` (mean) | `-0.151222 degC` (mean) | bias `+6.7158e-09 K`, RMS `1.9169e-07 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M05_May` | `ICE_SSS` | `33.350602 psu` (mean) | `33.350602 psu` (mean) | bias `+3.3853e-10 psu`, RMS `1.0183e-08 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M06_Jun` | `SST` | `18.044538 degC` (mean) | `18.044538 degC` (mean) | bias `+6.2131e-10 K`, RMS `3.2197e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M06_Jun` | `T_0-100m` | `16.975724 degC` (mean) | `16.975724 degC` (mean) | bias `+5.7855e-10 K`, RMS `3.1558e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M06_Jun` | `T_100-1000m` | `7.693060 degC` (mean) | `7.693060 degC` (mean) | bias `-1.1422e-10 K`, RMS `1.4221e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M06_Jun` | `T_>1000m` | `1.939163 degC` (mean) | `1.939163 degC` (mean) | bias `-1.1859e-13 K`, RMS `9.3142e-11 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M06_Jun` | `SSS` | `34.741153 psu` (mean) | `34.741153 psu` (mean) | bias `+3.8185e-11 psu`, RMS `2.1403e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M06_Jun` | `S_0-100m` | `34.862609 psu` (mean) | `34.862609 psu` (mean) | bias `+3.4072e-11 psu`, RMS `2.0453e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M06_Jun` | `S_100-1000m` | `34.725838 psu` (mean) | `34.725838 psu` (mean) | bias `-3.4129e-12 psu`, RMS `6.0585e-10 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M06_Jun` | `S_>1000m` | `34.719812 psu` (mean) | `34.719812 psu` (mean) | bias `-1.8047e-14 psu`, RMS `3.5091e-12 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M06_Jun` | `SSH` | `+1.273275e-02 m` (mean; spatial RMS `0.639610 m`) | `+1.273275e-02 m` (mean; spatial RMS `0.639610 m`) | bias `-1.1206e-11 m`, RMS `3.9972e-10 m` | RMS `0.0000e+00 m` | `0.001 m` (`1 mm` RMS; REFERENCE ONLY) |
| `M06_Jun` | `KE` | `1.469879e-02 m^2 s^-2` (mean) | `1.469879e-02 m^2 s^-2` (mean) | bias `+3.8934e-13`, RMS `1.4307e-10 m^2 s^-2` (rel RMS `9.7334e-09`, rel `|mean diff|` `2.6488e-11`) | rel RMS `0.0000e+00` (RMS `0.0000e+00`) | `0.01` (`1%` of NEMO mean = `1.4699e-04 m^2 s^-2`; REFERENCE ONLY) |
| `M06_Jun` | `PSI` | `152.8001 Sv` (spatial RMS; mean `-144.9379 Sv`) | `152.8001 Sv` (spatial RMS; mean `-144.9379 Sv`) | bias `-2.5958e-10 Sv`, RMS `9.2615e-09 Sv`, corr `1.000000` | RMS `0.0000e+00 Sv`, corr `1.000000` | `0.1 Sv` RMS & `>= 0.999` corr (REFERENCE ONLY) |
| `M06_Jun` | `DRAKE` | `156.940419 Sv` (std `4.9245 Sv`) | `156.940419 Sv` (std `4.9245 Sv`) | `cand - NEMO = +1.6408e-10 Sv`, `|mean diff| = 1.6408e-10 Sv` | `|mean diff| = 0.0000e+00 Sv` | `0.5 Sv` (REFERENCE ONLY) |
| `M06_Jun` | `MLD` | `53.338823 m` (mean) | `53.338823 m` (mean) | bias `+1.3468e-12 m`, RMS `2.8645e-11 m` | RMS `0.0000e+00 m` | `1.0 m` RMS (REFERENCE ONLY) |
| `M06_Jun` | `ICE_SST` | `-0.054637 degC` (mean) | `-0.054637 degC` (mean) | bias `+4.2171e-09 K`, RMS `8.4944e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M06_Jun` | `ICE_SSS` | `33.342702 psu` (mean) | `33.342702 psu` (mean) | bias `+2.5875e-10 psu`, RMS `5.6125e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M07_Jul` | `SST` | `18.003188 degC` (mean) | `18.003188 degC` (mean) | bias `+4.3402e-10 K`, RMS `2.2401e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M07_Jul` | `T_0-100m` | `16.854585 degC` (mean) | `16.854585 degC` (mean) | bias `+3.9266e-10 K`, RMS `3.9706e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M07_Jul` | `T_100-1000m` | `7.694367 degC` (mean) | `7.694367 degC` (mean) | bias `-1.0112e-10 K`, RMS `1.5554e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M07_Jul` | `T_>1000m` | `1.939006 degC` (mean) | `1.939006 degC` (mean) | bias `-1.1630e-13 K`, RMS `1.7949e-10 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M07_Jul` | `SSS` | `34.735541 psu` (mean) | `34.735541 psu` (mean) | bias `+2.5218e-11 psu`, RMS `2.0273e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M07_Jul` | `S_0-100m` | `34.859498 psu` (mean) | `34.859498 psu` (mean) | bias `+2.1694e-11 psu`, RMS `3.4018e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M07_Jul` | `S_100-1000m` | `34.725582 psu` (mean) | `34.725582 psu` (mean) | bias `-2.1353e-12 psu`, RMS `1.0347e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M07_Jul` | `S_>1000m` | `34.719805 psu` (mean) | `34.719805 psu` (mean) | bias `-2.7968e-14 psu`, RMS `6.2572e-12 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M07_Jul` | `SSH` | `+1.721762e-02 m` (mean; spatial RMS `0.638437 m`) | `+1.721762e-02 m` (mean; spatial RMS `0.638437 m`) | bias `-1.1790e-11 m`, RMS `7.4963e-10 m` | RMS `0.0000e+00 m` | `0.001 m` (`1 mm` RMS; REFERENCE ONLY) |
| `M07_Jul` | `KE` | `1.730372e-02 m^2 s^-2` (mean) | `1.730372e-02 m^2 s^-2` (mean) | bias `+3.7162e-11`, RMS `1.9807e-09 m^2 s^-2` (rel RMS `1.1447e-07`, rel `|mean diff|` `2.1477e-09`) | rel RMS `0.0000e+00` (RMS `0.0000e+00`) | `0.01` (`1%` of NEMO mean = `1.7304e-04 m^2 s^-2`; REFERENCE ONLY) |
| `M07_Jul` | `PSI` | `149.1837 Sv` (spatial RMS; mean `-141.0761 Sv`) | `149.1837 Sv` (spatial RMS; mean `-141.0761 Sv`) | bias `+3.5799e-09 Sv`, RMS `1.3547e-07 Sv`, corr `1.000000` | RMS `0.0000e+00 Sv`, corr `1.000000` | `0.1 Sv` RMS & `>= 0.999` corr (REFERENCE ONLY) |
| `M07_Jul` | `DRAKE` | `153.282793 Sv` (std `3.5750 Sv`) | `153.282793 Sv` (std `3.5750 Sv`) | `cand - NEMO = -1.0764e-08 Sv`, `|mean diff| = 1.0764e-08 Sv` | `|mean diff| = 0.0000e+00 Sv` | `0.5 Sv` (REFERENCE ONLY) |
| `M07_Jul` | `MLD` | `59.095930 m` (mean) | `59.095930 m` (mean) | bias `+9.0339e-13 m`, RMS `3.7524e-10 m` | RMS `0.0000e+00 m` | `1.0 m` RMS (REFERENCE ONLY) |
| `M07_Jul` | `ICE_SST` | `0.184782 degC` (mean) | `0.184782 degC` (mean) | bias `+3.1304e-09 K`, RMS `5.4173e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M07_Jul` | `ICE_SSS` | `33.317771 psu` (mean) | `33.317771 psu` (mean) | bias `+2.3179e-10 psu`, RMS `4.3725e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M08_Aug` | `SST` | `17.970099 degC` (mean) | `17.970099 degC` (mean) | bias `+1.0218e-09 K`, RMS `7.2797e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M08_Aug` | `T_0-100m` | `16.781931 degC` (mean) | `16.781931 degC` (mean) | bias `+8.8100e-10 K`, RMS `2.7153e-07 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M08_Aug` | `T_100-1000m` | `7.695486 degC` (mean) | `7.695486 degC` (mean) | bias `-1.6743e-10 K`, RMS `7.2004e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M08_Aug` | `T_>1000m` | `1.938944 degC` (mean) | `1.938944 degC` (mean) | bias `-2.9934e-13 K`, RMS `9.0244e-10 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M08_Aug` | `SSS` | `34.725931 psu` (mean) | `34.725931 psu` (mean) | bias `+1.2076e-12 psu`, RMS `8.8039e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M08_Aug` | `S_0-100m` | `34.856004 psu` (mean) | `34.856004 psu` (mean) | bias `+4.7029e-11 psu`, RMS `2.2045e-08 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M08_Aug` | `S_100-1000m` | `34.726005 psu` (mean) | `34.726005 psu` (mean) | bias `-4.9173e-12 psu`, RMS `6.0199e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M08_Aug` | `S_>1000m` | `34.719816 psu` (mean) | `34.719816 psu` (mean) | bias `-3.4752e-14 psu`, RMS `2.8761e-11 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M08_Aug` | `SSH` | `+2.123374e-02 m` (mean; spatial RMS `0.643890 m`) | `+2.123374e-02 m` (mean; spatial RMS `0.643890 m`) | bias `-1.6050e-11 m`, RMS `1.5256e-09 m` | RMS `0.0000e+00 m` | `0.001 m` (`1 mm` RMS; REFERENCE ONLY) |
| `M08_Aug` | `KE` | `1.991276e-02 m^2 s^-2` (mean) | `1.991276e-02 m^2 s^-2` (mean) | bias `+1.0066e-10`, RMS `1.5288e-08 m^2 s^-2` (rel RMS `7.6776e-07`, rel `|mean diff|` `5.0550e-09`) | rel RMS `0.0000e+00` (RMS `0.0000e+00`) | `0.01` (`1%` of NEMO mean = `1.9913e-04 m^2 s^-2`; REFERENCE ONLY) |
| `M08_Aug` | `PSI` | `155.8946 Sv` (spatial RMS; mean `-147.5864 Sv`) | `155.8946 Sv` (spatial RMS; mean `-147.5864 Sv`) | bias `-4.0978e-09 Sv`, RMS `1.9063e-07 Sv`, corr `1.000000` | RMS `0.0000e+00 Sv`, corr `1.000000` | `0.1 Sv` RMS & `>= 0.999` corr (REFERENCE ONLY) |
| `M08_Aug` | `DRAKE` | `160.379823 Sv` (std `8.1899 Sv`) | `160.379823 Sv` (std `8.1899 Sv`) | `cand - NEMO = -2.2698e-10 Sv`, `|mean diff| = 2.2698e-10 Sv` | `|mean diff| = 0.0000e+00 Sv` | `0.5 Sv` (REFERENCE ONLY) |
| `M08_Aug` | `MLD` | `61.365100 m` (mean) | `61.365100 m` (mean) | bias `+4.9519e-14 m`, RMS `1.1965e-10 m` | RMS `0.0000e+00 m` | `1.0 m` RMS (REFERENCE ONLY) |
| `M08_Aug` | `ICE_SST` | `0.348383 degC` (mean) | `0.348383 degC` (mean) | bias `+2.4249e-09 K`, RMS `3.8933e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M08_Aug` | `ICE_SSS` | `33.288288 psu` (mean) | `33.288288 psu` (mean) | bias `+2.0200e-10 psu`, RMS `3.5386e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M09_Sep` | `SST` | `17.946859 degC` (mean) | `17.946859 degC` (mean) | bias `-2.7557e-09 K`, RMS `1.2176e-06 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M09_Sep` | `T_0-100m` | `16.797226 degC` (mean) | `16.797226 degC` (mean) | bias `-4.4277e-09 K`, RMS `2.5010e-06 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M09_Sep` | `T_100-1000m` | `7.697631 degC` (mean) | `7.697631 degC` (mean) | bias `+4.2540e-10 K`, RMS `1.0038e-06 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M09_Sep` | `T_>1000m` | `1.938920 degC` (mean) | `1.938920 degC` (mean) | bias `+3.5934e-13 K`, RMS `1.7161e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M09_Sep` | `SSS` | `34.719794 psu` (mean) | `34.719794 psu` (mean) | bias `-1.3163e-09 psu`, RMS `1.7826e-07 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M09_Sep` | `S_0-100m` | `34.853339 psu` (mean) | `34.853339 psu` (mean) | bias `-1.3774e-09 psu`, RMS `3.3050e-07 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M09_Sep` | `S_100-1000m` | `34.726284 psu` (mean) | `34.726284 psu` (mean) | bias `+1.6704e-10 psu`, RMS `8.1153e-08 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M09_Sep` | `S_>1000m` | `34.719807 psu` (mean) | `34.719807 psu` (mean) | bias `+9.2976e-15 psu`, RMS `5.8044e-10 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M09_Sep` | `SSH` | `+2.817954e-02 m` (mean; spatial RMS `0.645764 m`) | `+2.817954e-02 m` (mean; spatial RMS `0.645764 m`) | bias `-1.1807e-10 m`, RMS `2.8938e-08 m` | RMS `0.0000e+00 m` | `0.001 m` (`1 mm` RMS; REFERENCE ONLY) |
| `M09_Sep` | `KE` | `2.172986e-02 m^2 s^-2` (mean) | `2.172986e-02 m^2 s^-2` (mean) | bias `+1.2871e-11`, RMS `5.6644e-07 m^2 s^-2` (rel RMS `2.6068e-05`, rel `|mean diff|` `5.9234e-10`) | rel RMS `0.0000e+00` (RMS `0.0000e+00`) | `0.01` (`1%` of NEMO mean = `2.1730e-04 m^2 s^-2`; REFERENCE ONLY) |
| `M09_Sep` | `PSI` | `160.1059 Sv` (spatial RMS; mean `-152.0506 Sv`) | `160.1059 Sv` (spatial RMS; mean `-152.0506 Sv`) | bias `+7.6509e-08 Sv`, RMS `4.3547e-06 Sv`, corr `1.000000` | RMS `0.0000e+00 Sv`, corr `1.000000` | `0.1 Sv` RMS & `>= 0.999` corr (REFERENCE ONLY) |
| `M09_Sep` | `DRAKE` | `164.521310 Sv` (std `4.6497 Sv`) | `164.521310 Sv` (std `4.6497 Sv`) | `cand - NEMO = -5.1503e-08 Sv`, `|mean diff| = 5.1503e-08 Sv` | `|mean diff| = 0.0000e+00 Sv` | `0.5 Sv` (REFERENCE ONLY) |
| `M09_Sep` | `MLD` | `55.979986 m` (mean) | `55.979986 m` (mean) | bias `-5.4447e-12 m`, RMS `2.1952e-10 m` | RMS `0.0000e+00 m` | `1.0 m` RMS (REFERENCE ONLY) |
| `M09_Sep` | `ICE_SST` | `0.283965 degC` (mean) | `0.283965 degC` (mean) | bias `+1.8182e-09 K`, RMS `2.9185e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M09_Sep` | `ICE_SSS` | `33.269552 psu` (mean) | `33.269552 psu` (mean) | bias `+1.6752e-10 psu`, RMS `2.9167e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M10_Oct` | `SST` | `17.901981 degC` (mean) | `17.901982 degC` (mean) | bias `+5.2638e-07 K`, RMS `7.0776e-05 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M10_Oct` | `T_0-100m` | `16.859976 degC` (mean) | `16.859976 degC` (mean) | bias `-2.3127e-07 K`, RMS `1.1113e-04 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M10_Oct` | `T_100-1000m` | `7.701138 degC` (mean) | `7.701138 degC` (mean) | bias `+2.1672e-08 K`, RMS `3.8344e-05 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M10_Oct` | `T_>1000m` | `1.938986 degC` (mean) | `1.938986 degC` (mean) | bias `+2.5388e-11 K`, RMS `6.7040e-07 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M10_Oct` | `SSS` | `34.722751 psu` (mean) | `34.722751 psu` (mean) | bias `-3.7656e-08 psu`, RMS `1.4015e-05 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M10_Oct` | `S_0-100m` | `34.852622 psu` (mean) | `34.852622 psu` (mean) | bias `+1.7328e-09 psu`, RMS `2.1845e-05 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M10_Oct` | `S_100-1000m` | `34.726079 psu` (mean) | `34.726079 psu` (mean) | bias `+3.1527e-11 psu`, RMS `3.5027e-06 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M10_Oct` | `S_>1000m` | `34.719826 psu` (mean) | `34.719826 psu` (mean) | bias `-8.4341e-13 psu`, RMS `2.1885e-08 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M10_Oct` | `SSH` | `+3.534739e-02 m` (mean; spatial RMS `0.645526 m`) | `+3.534739e-02 m` (mean; spatial RMS `0.645526 m`) | bias `-5.6347e-09 m`, RMS `9.4832e-07 m` | RMS `0.0000e+00 m` | `0.001 m` (`1 mm` RMS; REFERENCE ONLY) |
| `M10_Oct` | `KE` | `1.981617e-02 m^2 s^-2` (mean) | `1.981619e-02 m^2 s^-2` (mean) | bias `+2.1040e-08`, RMS `2.8958e-05 m^2 s^-2` (rel RMS `1.4613e-03`, rel `|mean diff|` `1.0618e-06`) | rel RMS `0.0000e+00` (RMS `0.0000e+00`) | `0.01` (`1%` of NEMO mean = `1.9816e-04 m^2 s^-2`; REFERENCE ONLY) |
| `M10_Oct` | `PSI` | `156.1773 Sv` (spatial RMS; mean `-148.3407 Sv`) | `156.1773 Sv` (spatial RMS; mean `-148.3407 Sv`) | bias `+6.2205e-07 Sv`, RMS `7.1743e-05 Sv`, corr `1.000000` | RMS `0.0000e+00 Sv`, corr `1.000000` | `0.1 Sv` RMS & `>= 0.999` corr (REFERENCE ONLY) |
| `M10_Oct` | `DRAKE` | `161.007185 Sv` (std `4.2374 Sv`) | `161.007183 Sv` (std `4.2374 Sv`) | `cand - NEMO = -2.0399e-06 Sv`, `|mean diff| = 2.0399e-06 Sv` | `|mean diff| = 0.0000e+00 Sv` | `0.5 Sv` (REFERENCE ONLY) |
| `M10_Oct` | `MLD` | `45.488221 m` (mean) | `45.488251 m` (mean) | bias `+2.9474e-05 m`, RMS `3.5275e-02 m` | RMS `0.0000e+00 m` | `1.0 m` RMS (REFERENCE ONLY) |
| `M10_Oct` | `ICE_SST` | `0.140651 degC` (mean) | `0.140651 degC` (mean) | bias `+1.3174e-09 K`, RMS `2.3596e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M10_Oct` | `ICE_SSS` | `33.272243 psu` (mean) | `33.272243 psu` (mean) | bias `+1.2351e-10 psu`, RMS `2.4950e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M11_Nov` | `SST` | `17.896163 degC` (mean) | `17.896163 degC` (mean) | bias `-3.4725e-07 K`, RMS `8.1962e-04 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M11_Nov` | `T_0-100m` | `16.906486 degC` (mean) | `16.906485 degC` (mean) | bias `-1.3237e-06 K`, RMS `7.4306e-04 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M11_Nov` | `T_100-1000m` | `7.707193 degC` (mean) | `7.707193 degC` (mean) | bias `+1.2890e-07 K`, RMS `2.3959e-04 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M11_Nov` | `T_>1000m` | `1.939027 degC` (mean) | `1.939027 degC` (mean) | bias `-1.3679e-10 K`, RMS `5.2469e-06 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M11_Nov` | `SSS` | `34.729905 psu` (mean) | `34.729904 psu` (mean) | bias `-4.3958e-07 psu`, RMS `1.0454e-04 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M11_Nov` | `S_0-100m` | `34.853678 psu` (mean) | `34.853677 psu` (mean) | bias `-9.9120e-08 psu`, RMS `9.2015e-05 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M11_Nov` | `S_100-1000m` | `34.725767 psu` (mean) | `34.725767 psu` (mean) | bias `+1.2789e-08 psu`, RMS `2.1122e-05 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M11_Nov` | `S_>1000m` | `34.719852 psu` (mean) | `34.719852 psu` (mean) | bias `-5.4800e-11 psu`, RMS `1.7708e-07 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M11_Nov` | `SSH` | `+4.023049e-02 m` (mean; spatial RMS `0.645753 m`) | `+4.023047e-02 m` (mean; spatial RMS `0.645753 m`) | bias `-2.2818e-08 m`, RMS `6.8366e-06 m` | RMS `0.0000e+00 m` | `0.001 m` (`1 mm` RMS; REFERENCE ONLY) |
| `M11_Nov` | `KE` | `2.136755e-02 m^2 s^-2` (mean) | `2.136762e-02 m^2 s^-2` (mean) | bias `+7.3471e-08`, RMS `2.7400e-04 m^2 s^-2` (rel RMS `1.2823e-02`, rel `|mean diff|` `3.4385e-06`) | rel RMS `0.0000e+00` (RMS `0.0000e+00`) | `0.01` (`1%` of NEMO mean = `2.1368e-04 m^2 s^-2`; REFERENCE ONLY) |
| `M11_Nov` | `PSI` | `154.7315 Sv` (spatial RMS; mean `-146.8335 Sv`) | `154.7314 Sv` (spatial RMS; mean `-146.8335 Sv`) | bias `+2.0766e-05 Sv`, RMS `7.5224e-04 Sv`, corr `1.000000` | RMS `0.0000e+00 Sv`, corr `1.000000` | `0.1 Sv` RMS & `>= 0.999` corr (REFERENCE ONLY) |
| `M11_Nov` | `DRAKE` | `159.530418 Sv` (std `4.4095 Sv`) | `159.530416 Sv` (std `4.4095 Sv`) | `cand - NEMO = -1.8753e-06 Sv`, `|mean diff| = 1.8753e-06 Sv` | `|mean diff| = 0.0000e+00 Sv` | `0.5 Sv` (REFERENCE ONLY) |
| `M11_Nov` | `MLD` | `41.916909 m` (mean) | `41.916874 m` (mean) | bias `-3.4927e-05 m`, RMS `1.2348e-01 m` | RMS `0.0000e+00 m` | `1.0 m` RMS (REFERENCE ONLY) |
| `M11_Nov` | `ICE_SST` | `0.032519 degC` (mean) | `0.032519 degC` (mean) | bias `+1.1211e-09 K`, RMS `2.5511e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M11_Nov` | `ICE_SSS` | `33.284745 psu` (mean) | `33.284745 psu` (mean) | bias `+5.9532e-11 psu`, RMS `4.9511e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M12_Dec` | `SST` | `18.029888 degC` (mean) | `18.029882 degC` (mean) | bias `-5.7258e-06 K`, RMS `1.8368e-03 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M12_Dec` | `T_0-100m` | `16.954759 degC` (mean) | `16.954760 degC` (mean) | bias `+7.5153e-07 K`, RMS `1.7365e-03 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M12_Dec` | `T_100-1000m` | `7.713733 degC` (mean) | `7.713733 degC` (mean) | bias `-1.1593e-08 K`, RMS `5.7897e-04 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M12_Dec` | `T_>1000m` | `1.939138 degC` (mean) | `1.939138 degC` (mean) | bias `+7.1997e-11 K`, RMS `1.2812e-05 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M12_Dec` | `SSS` | `34.731761 psu` (mean) | `34.731760 psu` (mean) | bias `-9.5668e-07 psu`, RMS `2.0464e-04 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M12_Dec` | `S_0-100m` | `34.853870 psu` (mean) | `34.853870 psu` (mean) | bias `-1.1468e-07 psu`, RMS `1.8289e-04 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M12_Dec` | `S_100-1000m` | `34.725351 psu` (mean) | `34.725351 psu` (mean) | bias `+1.2137e-08 psu`, RMS `4.0078e-05 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M12_Dec` | `S_>1000m` | `34.719882 psu` (mean) | `34.719882 psu` (mean) | bias `-1.4476e-10 psu`, RMS `4.3713e-07 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `M12_Dec` | `SSH` | `+4.511908e-02 m` (mean; spatial RMS `0.643913 m`) | `+4.511912e-02 m` (mean; spatial RMS `0.643913 m`) | bias `+4.4875e-08 m`, RMS `1.6333e-05 m` | RMS `0.0000e+00 m` | `0.001 m` (`1 mm` RMS; REFERENCE ONLY) |
| `M12_Dec` | `KE` | `1.827398e-02 m^2 s^-2` (mean) | `1.827265e-02 m^2 s^-2` (mean) | bias `-1.3226e-06`, RMS `5.7455e-04 m^2 s^-2` (rel RMS `3.1441e-02`, rel `|mean diff|` `7.2378e-05`) | rel RMS `0.0000e+00` (RMS `0.0000e+00`) | `0.01` (`1%` of NEMO mean = `1.8274e-04 m^2 s^-2`; REFERENCE ONLY) |
| `M12_Dec` | `PSI` | `155.2194 Sv` (spatial RMS; mean `-147.6010 Sv`) | `155.2194 Sv` (spatial RMS; mean `-147.6010 Sv`) | bias `-2.8916e-05 Sv`, RMS `2.6756e-03 Sv`, corr `1.000000` | RMS `0.0000e+00 Sv`, corr `1.000000` | `0.1 Sv` RMS & `>= 0.999` corr (REFERENCE ONLY) |
| `M12_Dec` | `DRAKE` | `160.624041 Sv` (std `4.6610 Sv`) | `160.624011 Sv` (std `4.6610 Sv`) | `cand - NEMO = -3.0384e-05 Sv`, `|mean diff| = 3.0384e-05 Sv` | `|mean diff| = 0.0000e+00 Sv` | `0.5 Sv` (REFERENCE ONLY) |
| `M12_Dec` | `MLD` | `40.522659 m` (mean) | `40.521610 m` (mean) | bias `-1.0490e-03 m`, RMS `2.0259e-01 m` | RMS `0.0000e+00 m` | `1.0 m` RMS (REFERENCE ONLY) |
| `M12_Dec` | `ICE_SST` | `0.099291 degC` (mean) | `0.099291 degC` (mean) | bias `-1.3658e-12 K`, RMS `8.5339e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `M12_Dec` | `ICE_SSS` | `33.296466 psu` (mean) | `33.296466 psu` (mean) | bias `-1.6617e-10 psu`, RMS `2.1410e-08 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `ALL_365d` | `SST` | `18.075066 degC` (mean) | `18.075065 degC` (mean) | bias `-4.7003e-07 K`, RMS `2.2950e-04 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `ALL_365d` | `T_0-100m` | `16.959423 degC` (mean) | `16.959423 degC` (mean) | bias `-6.4686e-08 K`, RMS `2.1824e-04 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `ALL_365d` | `T_100-1000m` | `7.694351 degC` (mean) | `7.694351 degC` (mean) | bias `+1.1434e-08 K`, RMS `7.2219e-05 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `ALL_365d` | `T_>1000m` | `1.939168 degC` (mean) | `1.939168 degC` (mean) | bias `-2.9933e-12 K`, RMS `1.5778e-06 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `ALL_365d` | `SSS` | `34.730463 psu` (mean) | `34.730463 psu` (mean) | bias `-1.2067e-07 psu`, RMS `2.7180e-05 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `ALL_365d` | `S_0-100m` | `34.856997 psu` (mean) | `34.856997 psu` (mean) | bias `-1.7838e-08 psu`, RMS `2.4982e-05 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `ALL_365d` | `S_100-1000m` | `34.726209 psu` (mean) | `34.726209 psu` (mean) | bias `+2.0968e-09 psu`, RMS `5.4449e-06 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `ALL_365d` | `S_>1000m` | `34.719805 psu` (mean) | `34.719805 psu` (mean) | bias `-1.6876e-11 psu`, RMS `5.3591e-08 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |
| `ALL_365d` | `SSH` | `+1.888187e-02 m` (mean; spatial RMS `0.642182 m`) | `+1.888187e-02 m` (mean; spatial RMS `0.642182 m`) | bias `+1.4435e-09 m`, RMS `2.0323e-06 m` | RMS `0.0000e+00 m` | `0.001 m` (`1 mm` RMS; REFERENCE ONLY) |
| `ALL_365d` | `KE` | `1.677180e-02 m^2 s^-2` (mean) | `1.677169e-02 m^2 s^-2` (mean) | bias `-1.0449e-07`, RMS `7.3826e-05 m^2 s^-2` (rel RMS `4.4018e-03`, rel `|mean diff|` `6.2304e-06`) | rel RMS `0.0000e+00` (RMS `0.0000e+00`) | `0.01` (`1%` of NEMO mean = `1.6772e-04 m^2 s^-2`; REFERENCE ONLY) |
| `ALL_365d` | `PSI` | `154.8744 Sv` (spatial RMS; mean `-146.2110 Sv`) | `154.8744 Sv` (spatial RMS; mean `-146.2110 Sv`) | bias `-6.8995e-07 Sv`, RMS `2.9555e-04 Sv`, corr `1.000000` | RMS `0.0000e+00 Sv`, corr `1.000000` | `0.1 Sv` RMS & `>= 0.999` corr (REFERENCE ONLY) |
| `ALL_365d` | `DRAKE` | `157.751095 Sv` (std `20.3060 Sv`) | `157.751092 Sv` (std `20.3060 Sv`) | `cand - NEMO = -2.9131e-06 Sv`, `|mean diff| = 2.9131e-06 Sv` | `|mean diff| = 0.0000e+00 Sv` | `0.5 Sv` (REFERENCE ONLY) |
| `ALL_365d` | `MLD` | `49.011344 m` (mean) | `49.011255 m` (mean) | bias `-8.9459e-05 m`, RMS `3.0352e-02 m` | RMS `0.0000e+00 m` | `1.0 m` RMS (REFERENCE ONLY) |
| `ALL_365d` | `ICE_SST` | `0.113697 degC` (mean) | `0.113697 degC` (mean) | bias `+2.2826e-09 K`, RMS `6.5145e-08 K` | RMS `0.0000e+00 K` | `0.01 K` RMS (REFERENCE ONLY) |
| `ALL_365d` | `ICE_SSS` | `33.307313 psu` (mean) | `33.307313 psu` (mean) | bias `+1.2810e-10 psu`, RMS `5.6566e-09 psu` | RMS `0.0000e+00 psu` | `0.002 psu` RMS (REFERENCE ONLY) |

### Day-by-Day SST RMS Difference Curve

| Day | `kt` | NEMO Mean SST (`degC`) | legoESM Mean SST (`degC`) | SST Bias (`K`) | SST RMS (`K`) | NEMO Twin B vs A SST RMS (`K`) | Decision-185 Reference Bar (`K`, Reference Only) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `8` | `18.055454` | `18.055454` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `2` | `16` | `18.042659` | `18.042659` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `3` | `24` | `18.036184` | `18.036184` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `4` | `32` | `18.035525` | `18.035525` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `5` | `40` | `18.037099` | `18.037099` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `6` | `48` | `18.032160` | `18.032160` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `7` | `56` | `18.028847` | `18.028847` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `8` | `64` | `18.035050` | `18.035050` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `9` | `72` | `18.038679` | `18.038679` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `10` | `80` | `18.045208` | `18.045208` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `11` | `88` | `18.053770` | `18.053770` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `12` | `96` | `18.063861` | `18.063861` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `13` | `104` | `18.072721` | `18.072721` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `14` | `112` | `18.081292` | `18.081292` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `15` | `120` | `18.088632` | `18.088632` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `16` | `128` | `18.092398` | `18.092398` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `17` | `136` | `18.096098` | `18.096098` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `18` | `144` | `18.099716` | `18.099716` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `19` | `152` | `18.106047` | `18.106047` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `20` | `160` | `18.105162` | `18.105162` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `21` | `168` | `18.101407` | `18.101407` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `22` | `176` | `18.103963` | `18.103963` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `23` | `184` | `18.111052` | `18.111052` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `24` | `192` | `18.118596` | `18.118596` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `25` | `200` | `18.125827` | `18.125827` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `26` | `208` | `18.136065` | `18.136065` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `27` | `216` | `18.145776` | `18.145776` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `28` | `224` | `18.151335` | `18.151335` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `29` | `232` | `18.156791` | `18.156791` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `30` | `240` | `18.166547` | `18.166547` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `31` | `248` | `18.173372` | `18.173372` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `32` | `256` | `18.181947` | `18.181947` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `33` | `264` | `18.188891` | `18.188891` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `34` | `272` | `18.194522` | `18.194522` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `35` | `280` | `18.197319` | `18.197319` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `36` | `288` | `18.199555` | `18.199555` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `37` | `296` | `18.206892` | `18.206892` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `38` | `304` | `18.218955` | `18.218955` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `39` | `312` | `18.231993` | `18.231993` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `40` | `320` | `18.235477` | `18.235477` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `41` | `328` | `18.239641` | `18.239641` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `42` | `336` | `18.249642` | `18.249642` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `43` | `344` | `18.261073` | `18.261073` | `+0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `0.01` |
| `44` | `352` | `18.265574` | `18.265574` | `-6.7177e-14` | `1.1151e-11` | `0.0000e+00` | `0.01` |
| `45` | `360` | `18.267175` | `18.267175` | `-4.0211e-13` | `3.4607e-11` | `0.0000e+00` | `0.01` |
| `46` | `368` | `18.268658` | `18.268658` | `-8.6777e-15` | `8.3187e-11` | `0.0000e+00` | `0.01` |
| `47` | `376` | `18.273136` | `18.273136` | `+3.3217e-14` | `8.0413e-11` | `0.0000e+00` | `0.01` |
| `48` | `384` | `18.274433` | `18.274433` | `+3.1850e-14` | `5.7924e-11` | `0.0000e+00` | `0.01` |
| `49` | `392` | `18.279430` | `18.279430` | `+2.8678e-13` | `7.7317e-11` | `0.0000e+00` | `0.01` |
| `50` | `400` | `18.287747` | `18.287747` | `+1.6580e-12` | `1.1966e-10` | `0.0000e+00` | `0.01` |
| `51` | `408` | `18.294564` | `18.294564` | `+1.0417e-12` | `9.8792e-11` | `0.0000e+00` | `0.01` |
| `52` | `416` | `18.299114` | `18.299114` | `+1.7359e-12` | `1.3255e-10` | `0.0000e+00` | `0.01` |
| `53` | `424` | `18.303612` | `18.303612` | `-1.6754e-12` | `1.6807e-10` | `0.0000e+00` | `0.01` |
| `54` | `432` | `18.305446` | `18.305446` | `-1.8878e-13` | `1.3983e-10` | `0.0000e+00` | `0.01` |
| `55` | `440` | `18.308692` | `18.308692` | `+3.3486e-13` | `1.0296e-10` | `0.0000e+00` | `0.01` |
| `56` | `448` | `18.311217` | `18.311217` | `+1.2661e-13` | `9.7103e-11` | `0.0000e+00` | `0.01` |
| `57` | `456` | `18.312852` | `18.312852` | `+3.3042e-13` | `1.0988e-10` | `0.0000e+00` | `0.01` |
| `58` | `464` | `18.312033` | `18.312033` | `-8.4534e-14` | `1.2713e-10` | `0.0000e+00` | `0.01` |
| `59` | `472` | `18.312281` | `18.312281` | `-2.7948e-13` | `8.9412e-11` | `0.0000e+00` | `0.01` |
| `60` | `480` | `18.313510` | `18.313510` | `+2.3033e-13` | `1.3706e-10` | `0.0000e+00` | `0.01` |
| `61` | `488` | `18.314987` | `18.314987` | `+6.3599e-13` | `1.1903e-10` | `0.0000e+00` | `0.01` |
| `62` | `496` | `18.315505` | `18.315505` | `+2.0221e-12` | `1.8144e-10` | `0.0000e+00` | `0.01` |
| `63` | `504` | `18.310442` | `18.310442` | `+3.3661e-14` | `1.0844e-10` | `0.0000e+00` | `0.01` |
| `64` | `512` | `18.305482` | `18.305482` | `+2.0292e-12` | `2.0365e-10` | `0.0000e+00` | `0.01` |
| `65` | `520` | `18.303240` | `18.303240` | `+3.3249e-13` | `1.9609e-10` | `0.0000e+00` | `0.01` |
| `66` | `528` | `18.302976` | `18.302976` | `+5.1685e-13` | `1.6846e-10` | `0.0000e+00` | `0.01` |
| `67` | `536` | `18.301185` | `18.301185` | `-4.8096e-13` | `1.1256e-10` | `0.0000e+00` | `0.01` |
| `68` | `544` | `18.295218` | `18.295218` | `-3.3696e-13` | `1.0907e-10` | `0.0000e+00` | `0.01` |
| `69` | `552` | `18.292291` | `18.292291` | `-2.4726e-12` | `2.7326e-10` | `0.0000e+00` | `0.01` |
| `70` | `560` | `18.292949` | `18.292949` | `-1.4400e-12` | `2.9590e-10` | `0.0000e+00` | `0.01` |
| `71` | `568` | `18.296486` | `18.296486` | `-3.0498e-12` | `5.3579e-10` | `0.0000e+00` | `0.01` |
| `72` | `576` | `18.302854` | `18.302854` | `-1.8283e-12` | `2.6701e-10` | `0.0000e+00` | `0.01` |
| `73` | `584` | `18.306536` | `18.306536` | `-1.5568e-12` | `2.5186e-10` | `0.0000e+00` | `0.01` |
| `74` | `592` | `18.309478` | `18.309478` | `-1.8172e-12` | `2.3867e-10` | `0.0000e+00` | `0.01` |
| `75` | `600` | `18.311494` | `18.311494` | `-1.6375e-12` | `2.0626e-10` | `0.0000e+00` | `0.01` |
| `76` | `608` | `18.311062` | `18.311062` | `-4.0999e-14` | `2.0336e-10` | `0.0000e+00` | `0.01` |
| `77` | `616` | `18.309289` | `18.309289` | `-2.5384e-12` | `3.4265e-10` | `0.0000e+00` | `0.01` |
| `78` | `624` | `18.306043` | `18.306043` | `-1.4445e-13` | `1.7871e-10` | `0.0000e+00` | `0.01` |
| `79` | `632` | `18.308611` | `18.308611` | `+1.1914e-12` | `2.7130e-10` | `0.0000e+00` | `0.01` |
| `80` | `640` | `18.316229` | `18.316229` | `-1.0839e-13` | `1.4434e-10` | `0.0000e+00` | `0.01` |
| `81` | `648` | `18.323276` | `18.323276` | `+3.6053e-13` | `1.5061e-10` | `0.0000e+00` | `0.01` |
| `82` | `656` | `18.328222` | `18.328222` | `-2.0944e-13` | `1.5034e-10` | `0.0000e+00` | `0.01` |
| `83` | `664` | `18.334433` | `18.334433` | `-5.7448e-13` | `1.4957e-10` | `0.0000e+00` | `0.01` |
| `84` | `672` | `18.338665` | `18.338665` | `-4.9149e-13` | `1.4846e-10` | `0.0000e+00` | `0.01` |
| `85` | `680` | `18.337976` | `18.337976` | `+5.0192e-13` | `1.9515e-10` | `0.0000e+00` | `0.01` |
| `86` | `688` | `18.336901` | `18.336901` | `+6.9067e-13` | `1.9704e-10` | `0.0000e+00` | `0.01` |
| `87` | `696` | `18.335361` | `18.335361` | `-1.0320e-13` | `1.8035e-10` | `0.0000e+00` | `0.01` |
| `88` | `704` | `18.333687` | `18.333687` | `-8.5487e-13` | `1.7111e-10` | `0.0000e+00` | `0.01` |
| `89` | `712` | `18.330891` | `18.330891` | `-1.9986e-12` | `2.8751e-10` | `0.0000e+00` | `0.01` |
| `90` | `720` | `18.325764` | `18.325764` | `+1.4151e-12` | `3.5625e-10` | `0.0000e+00` | `0.01` |
| `91` | `728` | `18.324798` | `18.324798` | `-1.3538e-12` | `2.1615e-10` | `0.0000e+00` | `0.01` |
| `92` | `736` | `18.322213` | `18.322213` | `-2.2088e-13` | `2.1680e-10` | `0.0000e+00` | `0.01` |
| `93` | `744` | `18.315828` | `18.315828` | `-1.5551e-12` | `2.4844e-10` | `0.0000e+00` | `0.01` |
| `94` | `752` | `18.310599` | `18.310599` | `-8.1639e-13` | `2.1536e-10` | `0.0000e+00` | `0.01` |
| `95` | `760` | `18.311634` | `18.311634` | `-6.5822e-13` | `2.4423e-10` | `0.0000e+00` | `0.01` |
| `96` | `768` | `18.314491` | `18.314491` | `-1.2810e-13` | `3.0092e-10` | `0.0000e+00` | `0.01` |
| `97` | `776` | `18.315138` | `18.315138` | `-2.6676e-12` | `3.9002e-10` | `0.0000e+00` | `0.01` |
| `98` | `784` | `18.310536` | `18.310536` | `-3.3144e-12` | `4.0828e-10` | `0.0000e+00` | `0.01` |
| `99` | `792` | `18.305186` | `18.305186` | `-3.0694e-12` | `4.0083e-10` | `0.0000e+00` | `0.01` |
| `100` | `800` | `18.304541` | `18.304541` | `-3.9094e-12` | `3.9344e-10` | `0.0000e+00` | `0.01` |
| `101` | `808` | `18.305647` | `18.305647` | `-3.9617e-12` | `4.9456e-10` | `0.0000e+00` | `0.01` |
| `102` | `816` | `18.304149` | `18.304149` | `-1.7919e-12` | `4.6641e-10` | `0.0000e+00` | `0.01` |
| `103` | `824` | `18.298464` | `18.298464` | `-1.4555e-12` | `4.2052e-10` | `0.0000e+00` | `0.01` |
| `104` | `832` | `18.295125` | `18.295125` | `-1.6061e-12` | `3.8815e-10` | `0.0000e+00` | `0.01` |
| `105` | `840` | `18.294404` | `18.294404` | `-6.5967e-13` | `3.8881e-10` | `0.0000e+00` | `0.01` |
| `106` | `848` | `18.295245` | `18.295245` | `-1.6865e-12` | `4.6068e-10` | `0.0000e+00` | `0.01` |
| `107` | `856` | `18.293759` | `18.293759` | `-1.4321e-12` | `4.0633e-10` | `0.0000e+00` | `0.01` |
| `108` | `864` | `18.294629` | `18.294629` | `-7.3871e-13` | `3.8915e-10` | `0.0000e+00` | `0.01` |
| `109` | `872` | `18.290876` | `18.290876` | `-1.8316e-12` | `3.7531e-10` | `0.0000e+00` | `0.01` |
| `110` | `880` | `18.283980` | `18.283980` | `-1.5243e-12` | `3.6582e-10` | `0.0000e+00` | `0.01` |
| `111` | `888` | `18.274888` | `18.274888` | `-4.8696e-13` | `3.4579e-10` | `0.0000e+00` | `0.01` |
| `112` | `896` | `18.267954` | `18.267954` | `+5.6133e-09` | `5.6067e-07` | `0.0000e+00` | `0.01` |
| `113` | `904` | `18.267171` | `18.267171` | `+4.8476e-09` | `4.7059e-07` | `0.0000e+00` | `0.01` |
| `114` | `912` | `18.268871` | `18.268871` | `+3.7607e-09` | `3.7984e-07` | `0.0000e+00` | `0.01` |
| `115` | `920` | `18.265784` | `18.265784` | `+3.2129e-09` | `3.2409e-07` | `0.0000e+00` | `0.01` |
| `116` | `928` | `18.260755` | `18.260755` | `+2.8654e-09` | `2.8479e-07` | `0.0000e+00` | `0.01` |
| `117` | `936` | `18.255572` | `18.255572` | `+2.3476e-09` | `2.3103e-07` | `0.0000e+00` | `0.01` |
| `118` | `944` | `18.253415` | `18.253415` | `+2.0296e-09` | `1.9646e-07` | `0.0000e+00` | `0.01` |
| `119` | `952` | `18.252823` | `18.252823` | `+1.8889e-09` | `1.7886e-07` | `0.0000e+00` | `0.01` |
| `120` | `960` | `18.253212` | `18.253212` | `+1.6627e-09` | `1.5413e-07` | `0.0000e+00` | `0.01` |
| `121` | `968` | `18.252365` | `18.252365` | `+1.5386e-09` | `1.4152e-07` | `0.0000e+00` | `0.01` |
| `122` | `976` | `18.253078` | `18.253078` | `+1.4720e-09` | `1.3418e-07` | `0.0000e+00` | `0.01` |
| `123` | `984` | `18.252043` | `18.252043` | `+1.4278e-09` | `1.2876e-07` | `0.0000e+00` | `0.01` |
| `124` | `992` | `18.243642` | `18.243642` | `+1.3816e-09` | `1.2219e-07` | `0.0000e+00` | `0.01` |
| `125` | `1000` | `18.233134` | `18.233134` | `+1.3311e-09` | `1.1522e-07` | `0.0000e+00` | `0.01` |
| `126` | `1008` | `18.223780` | `18.223780` | `+1.2833e-09` | `1.0757e-07` | `0.0000e+00` | `0.01` |
| `127` | `1016` | `18.215780` | `18.215780` | `+1.2247e-09` | `1.0031e-07` | `0.0000e+00` | `0.01` |
| `128` | `1024` | `18.205585` | `18.205585` | `+1.1571e-09` | `9.1920e-08` | `0.0000e+00` | `0.01` |
| `129` | `1032` | `18.195800` | `18.195800` | `+1.0984e-09` | `8.6519e-08` | `0.0000e+00` | `0.01` |
| `130` | `1040` | `18.184670` | `18.184670` | `+1.0726e-09` | `8.3285e-08` | `0.0000e+00` | `0.01` |
| `131` | `1048` | `18.176864` | `18.176864` | `+1.0417e-09` | `7.8197e-08` | `0.0000e+00` | `0.01` |
| `132` | `1056` | `18.171082` | `18.171082` | `+9.9547e-10` | `7.3543e-08` | `0.0000e+00` | `0.01` |
| `133` | `1064` | `18.166311` | `18.166311` | `+9.4470e-10` | `6.8716e-08` | `0.0000e+00` | `0.01` |
| `134` | `1072` | `18.163369` | `18.163369` | `+9.1078e-10` | `6.5245e-08` | `0.0000e+00` | `0.01` |
| `135` | `1080` | `18.158935` | `18.158935` | `+8.8349e-10` | `6.2296e-08` | `0.0000e+00` | `0.01` |
| `136` | `1088` | `18.153059` | `18.153059` | `+8.5950e-10` | `6.0154e-08` | `0.0000e+00` | `0.01` |
| `137` | `1096` | `18.148715` | `18.148715` | `+8.3573e-10` | `5.8282e-08` | `0.0000e+00` | `0.01` |
| `138` | `1104` | `18.147475` | `18.147475` | `+8.0658e-10` | `5.6217e-08` | `0.0000e+00` | `0.01` |
| `139` | `1112` | `18.148456` | `18.148456` | `+7.8373e-10` | `5.3983e-08` | `0.0000e+00` | `0.01` |
| `140` | `1120` | `18.146604` | `18.146604` | `+7.6718e-10` | `5.2052e-08` | `0.0000e+00` | `0.01` |
| `141` | `1128` | `18.145327` | `18.145327` | `+7.6860e-10` | `5.0912e-08` | `0.0000e+00` | `0.01` |
| `142` | `1136` | `18.140237` | `18.140237` | `+7.6811e-10` | `5.0036e-08` | `0.0000e+00` | `0.01` |
| `143` | `1144` | `18.135923` | `18.135923` | `+7.5498e-10` | `4.8664e-08` | `0.0000e+00` | `0.01` |
| `144` | `1152` | `18.133240` | `18.133240` | `+7.5366e-10` | `4.7966e-08` | `0.0000e+00` | `0.01` |
| `145` | `1160` | `18.132094` | `18.132094` | `+7.5394e-10` | `4.7075e-08` | `0.0000e+00` | `0.01` |
| `146` | `1168` | `18.128035` | `18.128035` | `+7.4127e-10` | `4.5861e-08` | `0.0000e+00` | `0.01` |
| `147` | `1176` | `18.123900` | `18.123900` | `+7.3231e-10` | `4.4609e-08` | `0.0000e+00` | `0.01` |
| `148` | `1184` | `18.118964` | `18.118964` | `+7.3004e-10` | `4.4118e-08` | `0.0000e+00` | `0.01` |
| `149` | `1192` | `18.112350` | `18.112350` | `+7.2039e-10` | `4.3674e-08` | `0.0000e+00` | `0.01` |
| `150` | `1200` | `18.110128` | `18.110128` | `+7.2520e-10` | `4.3081e-08` | `0.0000e+00` | `0.01` |
| `151` | `1208` | `18.109549` | `18.109549` | `+7.2195e-10` | `4.2376e-08` | `0.0000e+00` | `0.01` |
| `152` | `1216` | `18.108346` | `18.108346` | `+7.1477e-10` | `4.0934e-08` | `0.0000e+00` | `0.01` |
| `153` | `1224` | `18.103576` | `18.103576` | `+7.0552e-10` | `4.0139e-08` | `0.0000e+00` | `0.01` |
| `154` | `1232` | `18.094762` | `18.094762` | `+6.9626e-10` | `3.9495e-08` | `0.0000e+00` | `0.01` |
| `155` | `1240` | `18.085545` | `18.085545` | `+6.7897e-10` | `3.8405e-08` | `0.0000e+00` | `0.01` |
| `156` | `1248` | `18.078899` | `18.078899` | `+6.8067e-10` | `3.7689e-08` | `0.0000e+00` | `0.01` |
| `157` | `1256` | `18.073471` | `18.073471` | `+6.7911e-10` | `3.7242e-08` | `0.0000e+00` | `0.01` |
| `158` | `1264` | `18.069258` | `18.069258` | `+6.6888e-10` | `3.6202e-08` | `0.0000e+00` | `0.01` |
| `159` | `1272` | `18.062734` | `18.062734` | `+6.6069e-10` | `3.5696e-08` | `0.0000e+00` | `0.01` |
| `160` | `1280` | `18.054083` | `18.054083` | `+6.5160e-10` | `3.5097e-08` | `0.0000e+00` | `0.01` |
| `161` | `1288` | `18.048227` | `18.048227` | `+6.5016e-10` | `3.4569e-08` | `0.0000e+00` | `0.01` |
| `162` | `1296` | `18.045780` | `18.045780` | `+6.4975e-10` | `3.4033e-08` | `0.0000e+00` | `0.01` |
| `163` | `1304` | `18.044901` | `18.044901` | `+6.4436e-10` | `3.3500e-08` | `0.0000e+00` | `0.01` |
| `164` | `1312` | `18.044230` | `18.044230` | `+6.3970e-10` | `3.2951e-08` | `0.0000e+00` | `0.01` |
| `165` | `1320` | `18.042759` | `18.042759` | `+6.2613e-10` | `3.2372e-08` | `0.0000e+00` | `0.01` |
| `166` | `1328` | `18.041490` | `18.041490` | `+6.1865e-10` | `3.1863e-08` | `0.0000e+00` | `0.01` |
| `167` | `1336` | `18.039882` | `18.039882` | `+6.0828e-10` | `3.1246e-08` | `0.0000e+00` | `0.01` |
| `168` | `1344` | `18.037300` | `18.037300` | `+6.0698e-10` | `3.1118e-08` | `0.0000e+00` | `0.01` |
| `169` | `1352` | `18.031988` | `18.031988` | `+5.9336e-10` | `3.0412e-08` | `0.0000e+00` | `0.01` |
| `170` | `1360` | `18.026504` | `18.026504` | `+5.7944e-10` | `2.9981e-08` | `0.0000e+00` | `0.01` |
| `171` | `1368` | `18.021380` | `18.021380` | `+5.7582e-10` | `2.9761e-08` | `0.0000e+00` | `0.01` |
| `172` | `1376` | `18.016945` | `18.016945` | `+5.6632e-10` | `2.9161e-08` | `0.0000e+00` | `0.01` |
| `173` | `1384` | `18.014496` | `18.014496` | `+5.5938e-10` | `2.8628e-08` | `0.0000e+00` | `0.01` |
| `174` | `1392` | `18.014004` | `18.014004` | `+5.5105e-10` | `2.8190e-08` | `0.0000e+00` | `0.01` |
| `175` | `1400` | `18.014046` | `18.014046` | `+5.6773e-10` | `2.8012e-08` | `0.0000e+00` | `0.01` |
| `176` | `1408` | `18.012825` | `18.012825` | `+5.8427e-10` | `2.7785e-08` | `0.0000e+00` | `0.01` |
| `177` | `1416` | `18.015178` | `18.015178` | `+5.7035e-10` | `2.7021e-08` | `0.0000e+00` | `0.01` |
| `178` | `1424` | `18.017212` | `18.017212` | `+5.6197e-10` | `2.6715e-08` | `0.0000e+00` | `0.01` |
| `179` | `1432` | `18.020575` | `18.020575` | `+5.6990e-10` | `2.6242e-08` | `0.0000e+00` | `0.01` |
| `180` | `1440` | `18.026452` | `18.026452` | `+5.8129e-10` | `2.5894e-08` | `0.0000e+00` | `0.01` |
| `181` | `1448` | `18.029281` | `18.029281` | `+5.9808e-10` | `2.5576e-08` | `0.0000e+00` | `0.01` |
| `182` | `1456` | `18.028411` | `18.028411` | `+5.3570e-10` | `2.5424e-08` | `0.0000e+00` | `0.01` |
| `183` | `1464` | `18.025137` | `18.025137` | `+4.9733e-10` | `2.4940e-08` | `0.0000e+00` | `0.01` |
| `184` | `1472` | `18.025506` | `18.025506` | `+5.3229e-10` | `2.4510e-08` | `0.0000e+00` | `0.01` |
| `185` | `1480` | `18.024070` | `18.024070` | `+5.2365e-10` | `2.4146e-08` | `0.0000e+00` | `0.01` |
| `186` | `1488` | `18.020611` | `18.020611` | `+5.1130e-10` | `2.3762e-08` | `0.0000e+00` | `0.01` |
| `187` | `1496` | `18.018527` | `18.018527` | `+5.1718e-10` | `2.3539e-08` | `0.0000e+00` | `0.01` |
| `188` | `1504` | `18.016931` | `18.016931` | `+5.0873e-10` | `2.2985e-08` | `0.0000e+00` | `0.01` |
| `189` | `1512` | `18.015895` | `18.015895` | `+5.1087e-10` | `2.2704e-08` | `0.0000e+00` | `0.01` |
| `190` | `1520` | `18.013681` | `18.013681` | `+5.2419e-10` | `2.2531e-08` | `0.0000e+00` | `0.01` |
| `191` | `1528` | `18.012296` | `18.012296` | `+5.1323e-10` | `2.2121e-08` | `0.0000e+00` | `0.01` |
| `192` | `1536` | `18.013216` | `18.013216` | `+4.9547e-10` | `2.1717e-08` | `0.0000e+00` | `0.01` |
| `193` | `1544` | `18.018526` | `18.018526` | `+5.0844e-10` | `2.1771e-08` | `0.0000e+00` | `0.01` |
| `194` | `1552` | `18.020617` | `18.020617` | `+5.1688e-10` | `2.1573e-08` | `0.0000e+00` | `0.01` |
| `195` | `1560` | `18.019946` | `18.019946` | `+5.2379e-10` | `2.1464e-08` | `0.0000e+00` | `0.01` |
| `196` | `1568` | `18.020440` | `18.020440` | `+4.9993e-10` | `2.1409e-08` | `0.0000e+00` | `0.01` |
| `197` | `1576` | `18.015544` | `18.015544` | `+4.9287e-10` | `2.1207e-08` | `0.0000e+00` | `0.01` |
| `198` | `1584` | `18.005835` | `18.005835` | `+5.8822e-10` | `2.1391e-08` | `0.0000e+00` | `0.01` |
| `199` | `1592` | `17.998851` | `17.998851` | `+5.5878e-10` | `2.1205e-08` | `0.0000e+00` | `0.01` |
| `200` | `1600` | `17.996179` | `17.996179` | `+5.0228e-10` | `2.1075e-08` | `0.0000e+00` | `0.01` |
| `201` | `1608` | `17.995580` | `17.995580` | `+4.9063e-10` | `2.1100e-08` | `0.0000e+00` | `0.01` |
| `202` | `1616` | `17.992849` | `17.992849` | `+4.2816e-10` | `2.1088e-08` | `0.0000e+00` | `0.01` |
| `203` | `1624` | `17.990532` | `17.990532` | `+4.3384e-10` | `2.1164e-08` | `0.0000e+00` | `0.01` |
| `204` | `1632` | `17.988028` | `17.988028` | `+4.1125e-10` | `2.1323e-08` | `0.0000e+00` | `0.01` |
| `205` | `1640` | `17.985362` | `17.985362` | `+3.7334e-10` | `2.1957e-08` | `0.0000e+00` | `0.01` |
| `206` | `1648` | `17.983486` | `17.983486` | `+3.1412e-10` | `2.2217e-08` | `0.0000e+00` | `0.01` |
| `207` | `1656` | `17.979253` | `17.979253` | `+2.4249e-10` | `2.2134e-08` | `0.0000e+00` | `0.01` |
| `208` | `1664` | `17.976831` | `17.976831` | `+2.8009e-10` | `2.4794e-08` | `0.0000e+00` | `0.01` |
| `209` | `1672` | `17.975882` | `17.975882` | `+2.2547e-10` | `2.2257e-08` | `0.0000e+00` | `0.01` |
| `210` | `1680` | `17.973814` | `17.973814` | `+2.1142e-10` | `2.1311e-08` | `0.0000e+00` | `0.01` |
| `211` | `1688` | `17.973512` | `17.973512` | `+8.4767e-11` | `2.2508e-08` | `0.0000e+00` | `0.01` |
| `212` | `1696` | `17.973466` | `17.973466` | `+9.7944e-11` | `2.3091e-08` | `0.0000e+00` | `0.01` |
| `213` | `1704` | `17.973054` | `17.973054` | `+2.6704e-11` | `2.4057e-08` | `0.0000e+00` | `0.01` |
| `214` | `1712` | `17.973621` | `17.973621` | `+1.4104e-10` | `2.5426e-08` | `0.0000e+00` | `0.01` |
| `215` | `1720` | `17.972746` | `17.972746` | `+1.0934e-10` | `2.5517e-08` | `0.0000e+00` | `0.01` |
| `216` | `1728` | `17.974148` | `17.974148` | `+2.1026e-10` | `2.4904e-08` | `0.0000e+00` | `0.01` |
| `217` | `1736` | `17.973338` | `17.973338` | `+1.9557e-10` | `2.7939e-08` | `0.0000e+00` | `0.01` |
| `218` | `1744` | `17.973029` | `17.973029` | `+2.3955e-10` | `2.6404e-08` | `0.0000e+00` | `0.01` |
| `219` | `1752` | `17.969595` | `17.969595` | `+3.8802e-10` | `2.7582e-08` | `0.0000e+00` | `0.01` |
| `220` | `1760` | `17.967440` | `17.967440` | `+4.1638e-10` | `2.8800e-08` | `0.0000e+00` | `0.01` |
| `221` | `1768` | `17.967328` | `17.967328` | `+4.7275e-10` | `2.8283e-08` | `0.0000e+00` | `0.01` |
| `222` | `1776` | `17.969979` | `17.969979` | `+5.2930e-10` | `2.8821e-08` | `0.0000e+00` | `0.01` |
| `223` | `1784` | `17.970576` | `17.970576` | `+5.9010e-10` | `3.0081e-08` | `0.0000e+00` | `0.01` |
| `224` | `1792` | `17.972651` | `17.972651` | `+6.5489e-10` | `3.2164e-08` | `0.0000e+00` | `0.01` |
| `225` | `1800` | `17.973878` | `17.973878` | `+6.5058e-10` | `3.2442e-08` | `0.0000e+00` | `0.01` |
| `226` | `1808` | `17.971436` | `17.971436` | `+7.4576e-10` | `3.3444e-08` | `0.0000e+00` | `0.01` |
| `227` | `1816` | `17.966474` | `17.966474` | `+8.4230e-10` | `3.5134e-08` | `0.0000e+00` | `0.01` |
| `228` | `1824` | `17.962043` | `17.962043` | `+7.4664e-10` | `3.7241e-08` | `0.0000e+00` | `0.01` |
| `229` | `1832` | `17.963136` | `17.963136` | `+9.3954e-10` | `6.0234e-08` | `0.0000e+00` | `0.01` |
| `230` | `1840` | `17.967316` | `17.967316` | `-5.0050e-10` | `8.1178e-08` | `0.0000e+00` | `0.01` |
| `231` | `1848` | `17.969462` | `17.969462` | `-4.5280e-12` | `5.1377e-08` | `0.0000e+00` | `0.01` |
| `232` | `1856` | `17.970259` | `17.970259` | `+5.6709e-10` | `7.0952e-08` | `0.0000e+00` | `0.01` |
| `233` | `1864` | `17.971316` | `17.971316` | `+2.2522e-09` | `1.1919e-07` | `0.0000e+00` | `0.01` |
| `234` | `1872` | `17.970493` | `17.970493` | `+2.3035e-09` | `1.3826e-07` | `0.0000e+00` | `0.01` |
| `235` | `1880` | `17.969771` | `17.969771` | `+2.9157e-09` | `1.4694e-07` | `0.0000e+00` | `0.01` |
| `236` | `1888` | `17.973151` | `17.973151` | `+3.2667e-09` | `1.5974e-07` | `0.0000e+00` | `0.01` |
| `237` | `1896` | `17.975350` | `17.975350` | `+2.4829e-09` | `1.5127e-07` | `0.0000e+00` | `0.01` |
| `238` | `1904` | `17.974597` | `17.974597` | `+2.1281e-09` | `1.3601e-07` | `0.0000e+00` | `0.01` |
| `239` | `1912` | `17.972446` | `17.972446` | `+1.4721e-09` | `1.2778e-07` | `0.0000e+00` | `0.01` |
| `240` | `1920` | `17.967375` | `17.967375` | `+1.8500e-09` | `1.3084e-07` | `0.0000e+00` | `0.01` |
| `241` | `1928` | `17.965337` | `17.965337` | `+1.6735e-09` | `1.2383e-07` | `0.0000e+00` | `0.01` |
| `242` | `1936` | `17.966067` | `17.966067` | `+1.7639e-09` | `1.4800e-07` | `0.0000e+00` | `0.01` |
| `243` | `1944` | `17.965669` | `17.965669` | `+1.6056e-09` | `1.4287e-07` | `0.0000e+00` | `0.01` |
| `244` | `1952` | `17.968593` | `17.968593` | `+1.0377e-09` | `1.2752e-07` | `0.0000e+00` | `0.01` |
| `245` | `1960` | `17.972054` | `17.972054` | `+1.1173e-10` | `1.2556e-07` | `0.0000e+00` | `0.01` |
| `246` | `1968` | `17.969237` | `17.969237` | `-1.6415e-09` | `1.4335e-07` | `0.0000e+00` | `0.01` |
| `247` | `1976` | `17.971105` | `17.971105` | `-4.2858e-09` | `2.8487e-07` | `0.0000e+00` | `0.01` |
| `248` | `1984` | `17.974549` | `17.974549` | `-9.8683e-09` | `4.5791e-07` | `0.0000e+00` | `0.01` |
| `249` | `1992` | `17.973293` | `17.973293` | `-2.0102e-08` | `1.1999e-06` | `0.0000e+00` | `0.01` |
| `250` | `2000` | `17.970692` | `17.970693` | `+8.7662e-09` | `1.0573e-06` | `0.0000e+00` | `0.01` |
| `251` | `2008` | `17.967459` | `17.967459` | `+2.8352e-08` | `1.6465e-06` | `0.0000e+00` | `0.01` |
| `252` | `2016` | `17.965091` | `17.965091` | `+1.9284e-08` | `1.5078e-06` | `0.0000e+00` | `0.01` |
| `253` | `2024` | `17.963680` | `17.963680` | `+1.0927e-08` | `1.3528e-06` | `0.0000e+00` | `0.01` |
| `254` | `2032` | `17.957481` | `17.957481` | `+7.9083e-09` | `1.2832e-06` | `0.0000e+00` | `0.01` |
| `255` | `2040` | `17.952046` | `17.952046` | `+2.7090e-09` | `1.5383e-06` | `0.0000e+00` | `0.01` |
| `256` | `2048` | `17.945691` | `17.945691` | `+6.5641e-09` | `1.2730e-06` | `0.0000e+00` | `0.01` |
| `257` | `2056` | `17.938626` | `17.938626` | `+9.2790e-09` | `1.0706e-06` | `0.0000e+00` | `0.01` |
| `258` | `2064` | `17.938223` | `17.938223` | `+5.3296e-09` | `9.7178e-07` | `0.0000e+00` | `0.01` |
| `259` | `2072` | `17.935200` | `17.935200` | `+2.9701e-09` | `8.9499e-07` | `0.0000e+00` | `0.01` |
| `260` | `2080` | `17.929646` | `17.929646` | `+5.9151e-09` | `1.4239e-06` | `0.0000e+00` | `0.01` |
| `261` | `2088` | `17.924267` | `17.924267` | `-1.6396e-08` | `1.6048e-06` | `0.0000e+00` | `0.01` |
| `262` | `2096` | `17.916910` | `17.916910` | `-2.3272e-08` | `2.0963e-06` | `0.0000e+00` | `0.01` |
| `263` | `2104` | `17.915529` | `17.915529` | `-1.3755e-08` | `1.9129e-06` | `0.0000e+00` | `0.01` |
| `264` | `2112` | `17.923005` | `17.923005` | `-1.3023e-08` | `1.8774e-06` | `0.0000e+00` | `0.01` |
| `265` | `2120` | `17.932099` | `17.932099` | `-1.9149e-08` | `1.8599e-06` | `0.0000e+00` | `0.01` |
| `266` | `2128` | `17.940500` | `17.940500` | `-2.0463e-08` | `1.8051e-06` | `0.0000e+00` | `0.01` |
| `267` | `2136` | `17.948171` | `17.948171` | `-1.5242e-08` | `1.6000e-06` | `0.0000e+00` | `0.01` |
| `268` | `2144` | `17.951254` | `17.951254` | `-9.5674e-09` | `1.4136e-06` | `0.0000e+00` | `0.01` |
| `269` | `2152` | `17.947728` | `17.947728` | `-7.0195e-09` | `1.3134e-06` | `0.0000e+00` | `0.01` |
| `270` | `2160` | `17.940933` | `17.940932` | `-8.1164e-09` | `1.2388e-06` | `0.0000e+00` | `0.01` |
| `271` | `2168` | `17.930672` | `17.930672` | `-7.8493e-09` | `1.1996e-06` | `0.0000e+00` | `0.01` |
| `272` | `2176` | `17.923466` | `17.923466` | `-2.4720e-09` | `1.1313e-06` | `0.0000e+00` | `0.01` |
| `273` | `2184` | `17.918578` | `17.918578` | `+3.9856e-10` | `1.1148e-06` | `0.0000e+00` | `0.01` |
| `274` | `2192` | `17.917880` | `17.917880` | `+9.8001e-09` | `1.1045e-06` | `0.0000e+00` | `0.01` |
| `275` | `2200` | `17.918215` | `17.918215` | `+4.8048e-09` | `1.0974e-06` | `0.0000e+00` | `0.01` |
| `276` | `2208` | `17.916209` | `17.916209` | `-5.7055e-09` | `1.1005e-06` | `0.0000e+00` | `0.01` |
| `277` | `2216` | `17.914081` | `17.914081` | `-1.3904e-08` | `1.1125e-06` | `0.0000e+00` | `0.01` |
| `278` | `2224` | `17.912889` | `17.912889` | `-1.4060e-08` | `1.1053e-06` | `0.0000e+00` | `0.01` |
| `279` | `2232` | `17.908041` | `17.908041` | `-1.0918e-07` | `1.0268e-05` | `0.0000e+00` | `0.01` |
| `280` | `2240` | `17.906442` | `17.906442` | `+1.5385e-07` | `1.5666e-05` | `0.0000e+00` | `0.01` |
| `281` | `2248` | `17.904992` | `17.904993` | `+2.1972e-07` | `3.5574e-05` | `0.0000e+00` | `0.01` |
| `282` | `2256` | `17.904631` | `17.904631` | `+1.4712e-07` | `2.0138e-05` | `0.0000e+00` | `0.01` |
| `283` | `2264` | `17.902794` | `17.902794` | `+6.8807e-08` | `1.0854e-05` | `0.0000e+00` | `0.01` |
| `284` | `2272` | `17.898794` | `17.898794` | `-3.1808e-07` | `6.4908e-05` | `0.0000e+00` | `0.01` |
| `285` | `2280` | `17.893755` | `17.893755` | `+4.4260e-09` | `5.9508e-05` | `0.0000e+00` | `0.01` |
| `286` | `2288` | `17.891493` | `17.891494` | `+4.4048e-07` | `3.4889e-05` | `0.0000e+00` | `0.01` |
| `287` | `2296` | `17.888563` | `17.888563` | `+4.2186e-07` | `2.7241e-05` | `0.0000e+00` | `0.01` |
| `288` | `2304` | `17.886963` | `17.886964` | `+9.6547e-07` | `5.1516e-05` | `0.0000e+00` | `0.01` |
| `289` | `2312` | `17.882914` | `17.882915` | `+8.5707e-07` | `3.3647e-05` | `0.0000e+00` | `0.01` |
| `290` | `2320` | `17.883446` | `17.883446` | `+1.9702e-07` | `4.3499e-05` | `0.0000e+00` | `0.01` |
| `291` | `2328` | `17.884883` | `17.884883` | `+4.4312e-07` | `3.6782e-05` | `0.0000e+00` | `0.01` |
| `292` | `2336` | `17.885402` | `17.885403` | `+6.0649e-07` | `4.8360e-05` | `0.0000e+00` | `0.01` |
| `293` | `2344` | `17.885956` | `17.885956` | `+3.3716e-07` | `4.6637e-05` | `0.0000e+00` | `0.01` |
| `294` | `2352` | `17.888465` | `17.888465` | `+3.1032e-07` | `4.2533e-05` | `0.0000e+00` | `0.01` |
| `295` | `2360` | `17.892814` | `17.892815` | `+9.1174e-07` | `4.3499e-05` | `0.0000e+00` | `0.01` |
| `296` | `2368` | `17.901231` | `17.901232` | `+1.1861e-06` | `5.0403e-05` | `0.0000e+00` | `0.01` |
| `297` | `2376` | `17.907183` | `17.907183` | `+1.4039e-07` | `1.6257e-04` | `0.0000e+00` | `0.01` |
| `298` | `2384` | `17.910313` | `17.910312` | `-6.2759e-07` | `1.4835e-04` | `0.0000e+00` | `0.01` |
| `299` | `2392` | `17.913242` | `17.913245` | `+3.5195e-06` | `2.2979e-04` | `0.0000e+00` | `0.01` |
| `300` | `2400` | `17.917057` | `17.917060` | `+2.8231e-06` | `2.1308e-04` | `0.0000e+00` | `0.01` |
| `301` | `2408` | `17.916102` | `17.916102` | `-1.0053e-07` | `1.9117e-04` | `0.0000e+00` | `0.01` |
| `302` | `2416` | `17.913480` | `17.913480` | `+4.8266e-07` | `2.1259e-04` | `0.0000e+00` | `0.01` |
| `303` | `2424` | `17.908247` | `17.908249` | `+1.7845e-06` | `1.9438e-04` | `0.0000e+00` | `0.01` |
| `304` | `2432` | `17.904940` | `17.904941` | `+1.4712e-06` | `1.6069e-04` | `0.0000e+00` | `0.01` |
| `305` | `2440` | `17.897629` | `17.897628` | `-4.4660e-08` | `1.6243e-04` | `0.0000e+00` | `0.01` |
| `306` | `2448` | `17.889147` | `17.889147` | `+3.0387e-07` | `1.9435e-04` | `0.0000e+00` | `0.01` |
| `307` | `2456` | `17.886107` | `17.886108` | `+1.7076e-06` | `2.0010e-04` | `0.0000e+00` | `0.01` |
| `308` | `2464` | `17.885854` | `17.885858` | `+3.1740e-06` | `2.6907e-04` | `0.0000e+00` | `0.01` |
| `309` | `2472` | `17.888925` | `17.888929` | `+3.3992e-06` | `2.4374e-04` | `0.0000e+00` | `0.01` |
| `310` | `2480` | `17.891811` | `17.891812` | `+5.3581e-07` | `2.2897e-04` | `0.0000e+00` | `0.01` |
| `311` | `2488` | `17.888134` | `17.888133` | `-1.1774e-06` | `3.2294e-04` | `0.0000e+00` | `0.01` |
| `312` | `2496` | `17.881730` | `17.881730` | `-3.6548e-10` | `6.6135e-04` | `0.0000e+00` | `0.01` |
| `313` | `2504` | `17.878502` | `17.878506` | `+4.0160e-06` | `4.9504e-04` | `0.0000e+00` | `0.01` |
| `314` | `2512` | `17.878248` | `17.878251` | `+2.5420e-06` | `4.7420e-04` | `0.0000e+00` | `0.01` |
| `315` | `2520` | `17.880192` | `17.880192` | `+5.0557e-08` | `4.9633e-04` | `0.0000e+00` | `0.01` |
| `316` | `2528` | `17.882079` | `17.882077` | `-1.3723e-06` | `5.8867e-04` | `0.0000e+00` | `0.01` |
| `317` | `2536` | `17.883652` | `17.883655` | `+2.4457e-06` | `5.7662e-04` | `0.0000e+00` | `0.01` |
| `318` | `2544` | `17.890977` | `17.890983` | `+6.0035e-06` | `5.1337e-04` | `0.0000e+00` | `0.01` |
| `319` | `2552` | `17.893954` | `17.893955` | `+1.2876e-06` | `4.7333e-04` | `0.0000e+00` | `0.01` |
| `320` | `2560` | `17.893499` | `17.893504` | `+4.3887e-06` | `4.2274e-04` | `0.0000e+00` | `0.01` |
| `321` | `2568` | `17.892794` | `17.892800` | `+5.5468e-06` | `6.9728e-04` | `0.0000e+00` | `0.01` |
| `322` | `2576` | `17.890681` | `17.890686` | `+5.8099e-06` | `7.0280e-04` | `0.0000e+00` | `0.01` |
| `323` | `2584` | `17.892124` | `17.892127` | `+3.6271e-06` | `5.0553e-04` | `0.0000e+00` | `0.01` |
| `324` | `2592` | `17.896049` | `17.896043` | `-6.3561e-06` | `1.0583e-03` | `0.0000e+00` | `0.01` |
| `325` | `2600` | `17.901496` | `17.901485` | `-1.0153e-05` | `1.4786e-03` | `0.0000e+00` | `0.01` |
| `326` | `2608` | `17.905750` | `17.905747` | `-2.7588e-06` | `1.0291e-03` | `0.0000e+00` | `0.01` |
| `327` | `2616` | `17.908501` | `17.908507` | `+6.2093e-06` | `1.0128e-03` | `0.0000e+00` | `0.01` |
| `328` | `2624` | `17.909699` | `17.909706` | `+7.4770e-06` | `1.0387e-03` | `0.0000e+00` | `0.01` |
| `329` | `2632` | `17.906568` | `17.906562` | `-5.5356e-06` | `1.3170e-03` | `0.0000e+00` | `0.01` |
| `330` | `2640` | `17.909071` | `17.909061` | `-9.6195e-06` | `1.4376e-03` | `0.0000e+00` | `0.01` |
| `331` | `2648` | `17.913529` | `17.913514` | `-1.4402e-05` | `1.5722e-03` | `0.0000e+00` | `0.01` |
| `332` | `2656` | `17.919833` | `17.919822` | `-1.1432e-05` | `1.4118e-03` | `0.0000e+00` | `0.01` |
| `333` | `2664` | `17.922590` | `17.922588` | `-2.2556e-06` | `2.1676e-03` | `0.0000e+00` | `0.01` |
| `334` | `2672` | `17.925779` | `17.925776` | `-3.8345e-06` | `2.8362e-03` | `0.0000e+00` | `0.01` |
| `335` | `2680` | `17.932590` | `17.932576` | `-1.4193e-05` | `2.4346e-03` | `0.0000e+00` | `0.01` |
| `336` | `2688` | `17.941097` | `17.941088` | `-9.5468e-06` | `2.4298e-03` | `0.0000e+00` | `0.01` |
| `337` | `2696` | `17.952319` | `17.952317` | `-1.1728e-06` | `2.3215e-03` | `0.0000e+00` | `0.01` |
| `338` | `2704` | `17.961423` | `17.961427` | `+4.7365e-06` | `2.2537e-03` | `0.0000e+00` | `0.01` |
| `339` | `2712` | `17.967672` | `17.967664` | `-8.6932e-06` | `1.4705e-03` | `0.0000e+00` | `0.01` |
| `340` | `2720` | `17.973649` | `17.973631` | `-1.7476e-05` | `1.8045e-03` | `0.0000e+00` | `0.01` |
| `341` | `2728` | `17.972950` | `17.972947` | `-3.1846e-06` | `1.6901e-03` | `0.0000e+00` | `0.01` |
| `342` | `2736` | `17.968491` | `17.968494` | `+2.6175e-06` | `1.2258e-03` | `0.0000e+00` | `0.01` |
| `343` | `2744` | `17.962424` | `17.962418` | `-5.8335e-06` | `1.3038e-03` | `0.0000e+00` | `0.01` |
| `344` | `2752` | `17.960071` | `17.960075` | `+3.8246e-06` | `1.1433e-03` | `0.0000e+00` | `0.01` |
| `345` | `2760` | `17.959123` | `17.959113` | `-9.6233e-06` | `1.5203e-03` | `0.0000e+00` | `0.01` |
| `346` | `2768` | `17.960636` | `17.960636` | `+1.2560e-07` | `1.2367e-03` | `0.0000e+00` | `0.01` |
| `347` | `2776` | `17.967193` | `17.967206` | `+1.2718e-05` | `1.6060e-03` | `0.0000e+00` | `0.01` |
| `348` | `2784` | `17.968191` | `17.968201` | `+1.0007e-05` | `1.0496e-03` | `0.0000e+00` | `0.01` |
| `349` | `2792` | `17.968766` | `17.968765` | `-7.1178e-07` | `1.7122e-03` | `0.0000e+00` | `0.01` |
| `350` | `2800` | `17.973265` | `17.973287` | `+2.1528e-05` | `1.3935e-03` | `0.0000e+00` | `0.01` |
| `351` | `2808` | `17.985179` | `17.985183` | `+4.0901e-06` | `1.4045e-03` | `0.0000e+00` | `0.01` |
| `352` | `2816` | `17.999960` | `17.999959` | `-1.2975e-06` | `1.5586e-03` | `0.0000e+00` | `0.01` |
| `353` | `2824` | `18.016870` | `18.016869` | `-4.7503e-07` | `1.7473e-03` | `0.0000e+00` | `0.01` |
| `354` | `2832` | `18.035366` | `18.035369` | `+2.7091e-06` | `1.8365e-03` | `0.0000e+00` | `0.01` |
| `355` | `2840` | `18.052653` | `18.052660` | `+7.0369e-06` | `1.4206e-03` | `0.0000e+00` | `0.01` |
| `356` | `2848` | `18.074246` | `18.074239` | `-6.5217e-06` | `1.9990e-03` | `0.0000e+00` | `0.01` |
| `357` | `2856` | `18.095158` | `18.095136` | `-2.2085e-05` | `1.5669e-03` | `0.0000e+00` | `0.01` |
| `358` | `2864` | `18.116304` | `18.116292` | `-1.2099e-05` | `1.8527e-03` | `0.0000e+00` | `0.01` |
| `359` | `2872` | `18.136988` | `18.136965` | `-2.2290e-05` | `1.5833e-03` | `0.0000e+00` | `0.01` |
| `360` | `2880` | `18.152543` | `18.152525` | `-1.7711e-05` | `1.3892e-03` | `0.0000e+00` | `0.01` |
| `361` | `2888` | `18.162481` | `18.162471` | `-1.0038e-05` | `1.4208e-03` | `0.0000e+00` | `0.01` |
| `362` | `2896` | `18.170531` | `18.170519` | `-1.2852e-05` | `2.2223e-03` | `0.0000e+00` | `0.01` |
| `363` | `2904` | `18.175370` | `18.175341` | `-2.9296e-05` | `3.1607e-03` | `0.0000e+00` | `0.01` |
| `364` | `2912` | `18.181176` | `18.181166` | `-1.0374e-05` | `3.2592e-03` | `0.0000e+00` | `0.01` |
| `365` | `2920` | `18.181837` | `18.181805` | `-3.1419e-05` | `3.9232e-03` | `0.0000e+00` | `0.01` |


## Preflight result

Executed
`bash scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_acquisition/run.sh --preflight-only`:
- Status: **`ORCA2_ROUND342_S4_YEAR_PREFLIGHT_READY`**
- Evidence JSON:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/round342_s4_year_preflight.json`.
- All Checklist 1–16 controls and all 28 planted failure modes (including
  disk-space `10^9 GiB`, RAM `999,999 GiB`, `timeout --foreground`, dirty-tree
  untracked + modified-tracked, `run.sh` syntax, real-Arm-A log completion
  plants + shell missing-`STOP` / short-`time.step` plants, arm-A
  staging-failure quarantine + re-run, failed-arm quarantine + re-run,
  `refused/ -> --admit-existing` binary-checksum plant + clean promotion, and
  single-Arm-A promotion + Arm-B-only resume) fired and refused as expected.

## Choices and review

- ASKED: Decision 171/181 yearly-mean ice-cover climatology, Decision 175 K0
  selectors, Decision 160 ORCA2X long-run metric card reused month by month on
  both one-year arms and on `NEMO Twin B vs NEMO Twin A` as the reproducibility
  control, Decision 185 reference-only bars (`verdict = None`, no pass/fail),
  month-by-month and metric-by-metric table (`NEMO value`, `legoESM value`,
  `difference`, `Decision-185 REFERENCE bar`), day-by-day SST RMS difference
  curve across `Days 1..365` (`kt = 8..2920`), no NEMO execution, and
  `DECISION_NEEDED` for the user to decide the bar after seeing this.
- UNASKED: none.
- independent reviews (2026-10-30): Codex read-only review **LAND**; Claude
  Sonnet review **LAND** with receipt edits. Both reviews covered the 42
  round-342 commits carried by this round; all cited numbers were independently
  recomputed and **CONFIRMED**.

## Verification

- Focused Round-342 and ORCA2X metric card tests
  (`tests/ocean/fidelity/test_orca2_round342_s4_year.py` and
  `tests/ocean/fidelity/test_orca2x_round1_pilot_metrics.py`): **14 / 14 PASS**.
- Card-fingerprint gate (`nemo_testcase_l4_orca2_card_fingerprint.py`): **PASS**
  (`0` changed arrays across all 6 certified cards).
- Default ORCA2 ladder gate (`nemo_testcase_l4_orca2_round1_ladder_gate.py`):
  **PASS**.
- Per-receipt citation gate on
  `docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_round342_s4_year_receipt.md`
  and `docs/ocean/fidelity/testcases/orca2x_longrun_design.md`: **PASS** (`0`
  failures, `0` unmapped citations, `0` map-audit failures, all 9 self-tests
  fired).
- Campaign-default citation gate (`nemo_testcase_receipt_citation_gate.py`):
  **PASS** (`0` failures, `0` unmapped citations, `0` map-audit failures, all 9
  self-tests fired).
- Required standalone citation-gate pytest
  (`tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py` with
  `JAX_ENABLE_X64=1`): **17 / 17 PASS**.
