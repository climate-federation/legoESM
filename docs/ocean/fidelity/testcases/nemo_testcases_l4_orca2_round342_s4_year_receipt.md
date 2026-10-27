# ORCA2 round 342 — standalone one-year S4 acquisition tool, legoESM K0 driver, and Decision-185 metric card comparison

Date: 2026-10-27. Base `6b5e8413eb3b`; preregistration `75cc6fec9`.
Disposition: **ACQUISITION_NEEDED**. This round built and preflighted the
observer-free one-year (2,920-step, 365-day) standalone K0 NEMO twin
acquisition tool, executed the 20-step standalone K0 legoESM smoke run, and
built and smoke-verified the month-by-month Decision-160 ORCA2X metric card
comparison next to the Decision-185 reference bars. It did not build or run
NEMO (`--run` is operator-only) and did not start the full 2,920-step legoESM
year.

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
`nemo_testcase_l4_orca2_round342_s4_year_metrics.py` imports and calls
`pilot_metrics.static_geometry`, `pilot_metrics.read_nemo_day`,
`pilot_metrics.read_candidate_day`, `pilot_metrics.day_metrics`,
`pilot_metrics.monthly`, and `pilot_metrics.planted_controls` across each of
the 12 no-leap calendar months (`[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30,
31]` days, summing to `365` days) for both:
1. The NEMO reproducibility twin comparison (`Twin B vs Twin A`), and
2. The legoESM standalone K0 candidate vs NEMO Arm A comparison.

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
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/smoke_metrics_comparison/`,
both `Twin B vs Twin A` and `candidate vs NEMO` report `0.0000e+00` RMS (`PSI`
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
   and audits all `425` compiled `.o` objects (`2,530` functions, `445,274`
   floating-point instructions) plus a real-flag `mpif90` recompile of
   `restart.f90`. `plant_fp_drift` injects `+ 1.0e-30_wp` into `restart.f90`,
   compiles at real `-O3` flags, and proves `compare_fp_census` refuses.
5. **Checklist 7 (Whole-build observer-free equivalence + contamination plant):**
   `verify_same_program_no_observer` checks empty `MY_SRC`, scans all `424`
   preprocessed `.f90` files for observer symbols, and runs
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
9. **Checklist 11 (Rerun-after-refusal quarantine + `--admit-existing`):**
   `admit_existing_with_quarantine` admits a valid twin pair in place and, on
   failure, moves `orca2_k0_year1_{a,b}_np2` into `refused/<UTC-timestamp>_*`
   without deleting any file. Verified in preflight for both pass and fail
   paths.
10. **Checklist 12 (Stub replaces only `./nemo` and writes only K0 files):**
    In `--preflight-only`, `run.sh` installs a 3-line wrapper at `$run_dir/nemo`
    that calls `nemo_testcase_l4_orca2_round342_s4_year_gate.py --stub-nemo
    "$run_dir"`, reading `nn_itend` and `nn_stock` directly from
    `$run_dir/namelist_cfg` and writing only `ocean.output`,
    `output.namelist.dyn`, `mesh_mask_{0000,0001}.nc`, and
    `ORCA2_{kt:08d}_restart_{0000,0001}.nc` (with Day 1 `kt = 8` populated from
    admitted R340 S4 `kt = 9` entry state). `verify_no_unexpected_files` refuses
    any `.bin` or `oracle_*` file.
11. **Checklist 13–16 (No symlinked outputs, disk-space & RAM preflight, timeout,
    clean tree):** All outputs are regular files (`not p.is_symlink()`); disk
    guard checks free space (`361,329.85 GiB` available vs `25.0 GiB` required)
    with a `10^9 GiB` plant that refuses; RAM guard checks `>= 150 GiB` with a
    `999,999 GiB` plant that refuses; per-arm `timeout --foreground 5400s` wraps
    `mpirun -np 2 ./nemo`; dirty-tree guard refuses untracked or modified files.

## Preflight result

Executed
`bash scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_acquisition/run.sh --preflight-only`:
- Status: **`ORCA2_ROUND342_S4_YEAR_PREFLIGHT_READY`**
- Evidence JSON:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/round342_s4_year_preflight.json`
  (SHA-256 `4ee05995817ff0795ef63c04f1cf1220f023cf26b3650caee2c5677f70b88bb9`).
- All Checklist 1–16 controls and all 13 planted failure modes fired and refused
  as expected.

## Choices and review

- ASKED: Decision 171/181 yearly-mean ice-cover climatology, Decision 175 K0
  selectors, Decision 160 ORCA2X long-run metric card reused month by month,
  Decision 185 reference-only bars (`verdict = None`, no pass/fail), 20-step
  standalone K0 legoESM smoke run, no NEMO execution, and no full-year legoESM
  launch until operator instruction.
- UNASKED: none.
- independent review: operator

## Verification

- Focused Round-342 and ORCA2X metric card tests
  (`tests/ocean/fidelity/test_orca2_round342_s4_year.py` and
  `tests/ocean/fidelity/test_orca2x_round1_pilot_metrics.py`): **13 / 13 PASS**.
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
