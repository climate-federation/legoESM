# ORCA2 round 342 — standalone one-year S4 acquisition tool, legoESM K0 driver, and Decision-185 metric card comparison

Date: 2026-10-27. Base `6b5e8413eb3b`; preregistration `75cc6fec9`.
Disposition: **ACQUISITION_NEEDED** (Arm B only; Arm A completed and admitted).
This round built and preflighted the observer-free one-year (2,920-step,
365-day) standalone K0 NEMO twin acquisition tool, executed the 20-step
standalone K0 legoESM smoke run, built and smoke-verified the month-by-month
Decision-160 ORCA2X metric card comparison next to the Decision-185 reference
bars, and (in Resume 3) fixed the gfortran `stderr` `STOP 0` + `time.step`
completion check, admitted the completed operator-run Arm A
(`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/orca2_k0_year1_a_np2`),
and stopped for operator execution of reproducibility Twin B (`arm B only`).

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
available memory (`>= 150 GiB`) and idle GPU status (`memory.used < 500 MiB`,
`utilization.gpu == 0`), refuses `--steps > 40` unless `--allow-full-year` is
passed by the operator, and launches detached under `nohup`.

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
branch verbatim (`100%` byte-identical) and extended `chlorophyll_at_step` in
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round1_ladder_gate.py`
to `1 <= kt <= 2920` (bit-identical for `kt = 1..10`).
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
  selectors, Decision 160 ORCA2X long-run metric card reused month by month,
  Decision 185 reference-only bars (`verdict = None`, no pass/fail), 20-step
  standalone K0 legoESM smoke run, gfortran `stderr` `STOP 0` + `time.step == nn_itend`
  completion check with real-run log test and plants, kept Arm A admission via
  `--admit-existing`, no NEMO execution, and stop for operator execution of
  Arm B (`arm B only`).
- UNASKED: none.
- independent review: operator

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
