# Preregistration: ORCA2 Round 342 — Standalone ORCA2 One-Year Comparison (S4 Statistical Arm, Observer-Free K0 Twin Acquisition + legoESM K0 Driver + Month-by-Month ORCA2X Metric Card)

**Date:** 2026-04-20
**Round:** `orca2_r342` (Argon)
**Branch:** `fidelity/nemo-testcases-orca2-rounds` (base `6b5e8413eb3b`)
**Governing documents:**
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_orca2/standing_note.md` (Decisions 160, 171, 175, 181, 185; Acquisition Tool Checklist 1–16; Memory Rule; Kill Rule)
- `docs/ocean/fidelity/testcases/orca2_standalone_design.md` (Section 6 `S4` standalone composed card; one year = 2,920 steps, ~55 min stepping, reserve 90 min + build per NEMO arm, reproducibility twin before the candidate)
- `docs/ocean/fidelity/testcases/orca2x_longrun_design.md` (Decision 160 ORCA2X long-run metric card — reused directly, never re-implemented)

---

## 1. Scope & Hard Rules

This round builds the three deliverables for the **standalone ORCA2 one-year comparison (S4 statistical arm)** without running NEMO:

1. **The NEMO arm (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_gate.py` + `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_acquisition/run.sh`):**
   - Extends the reviewed round-339 K0 (`nemo_testcase_l4_orca2_round339_common.py`, `nemo_testcase_l4_orca2_round339_k0_smoke_gate.py`) and round-340 S4 (`nemo_testcase_l4_orca2_round340_standalone_gate.py`) tools by importing their helpers.
   - Configures an **observer-free K0 2,920-step run (`nn_itend = 2920`, `nn_stock = 8`, `nn_write = 2920`)** plus its **reproducibility twin before the candidate** (`orca2_k0_year1_a_np2` and `orca2_k0_year1_b_np2` under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year`, target build `ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y`, 90-minute = 5,400 s per-arm timeout).
   - Writes exactly what the ORCA2X metric card needs: 365 daily ocean restart pairs `ORCA2_{kt:08d}_restart_{0000,0001}.nc` for `kt = 8, 16, ..., 2920` (providing daily `tn, sn, un, vn, sshn` at end of day `d = 1..365`, plus `tb, sb, ub, vb, sshb, rhop, avt, avm, sbc_hc_b, sbc_sc_b, qsr_hc_b, fraqsr_1lev, utau_b, vtau_b, qns_b, emp_b, sfx_b, en, hmlp`), `ocean.output`, and `output.namelist.dyn`.
   - Zero in-loop observers (`MY_SRC` empty; `same_program_no_observer = true` verified on real `BLD/ppsrc/nemo/*.f90` and `BLD/obj/*.o`, with a non-blank plant that fails).
   - Satisfies all Acquisition Tool Checklist items 1–16, including a stub (`--stub-run-dir`) that replaces ONLY the `./nemo` executable inside the real `run.sh` pipeline and writes ONLY files this observer-free K0 build writes (plus a `--plant-stub-unproduced-file` control proving any unproduced file fails admission).
   - Stops after `--preflight-only` with `ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/orca2_r342/repo/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_acquisition/run.sh`.

2. **The legoESM arm (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py` + `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_legoesm_run.sh`):**
   - Runs the landed standalone K0 composed card (`build_nemo_orca2_k0_ocean_card`, `nemo_orca2_k0_ocean_step`, explicit `NemoStandaloneK0Config(nn_ice=1, ln_drgice_imp=False, ln_mxl0=True, nn_mxlice=0, ln_icebergs=False, initial_state="nemo_k0_istate")`) on the production path across steps `1..2920` (with full-year 12-month cyclic `fld_read` time interpolation for `sorunoff`, `sss`, `CHLA`, 3D T/S damping, and `ice_cover`, plus 6-hourly/daily/monthly `ln_tint=.false.` CORE forcing).
   - Saves daily `candidate_step{kt:08d}.npz` snapshots at `kt = 8, 16, ..., 2920` (and terminal `kt = steps` when `steps % 8 != 0`) containing `T, S, u, v, ssh, fr_i, kt, day`.
   - Detached launcher checks `free -g` (`>= 150 GB` available) and `nvidia-smi` for an idle GPU (`memory.used < 500 MiB`, `utilization.gpu == 0`), launches via `nohup`, records PID in `legoesm_run.pid`, and refuses `--steps 2920` unless `--allow-full-year` is passed by the operator.
   - Runs a **20-step smoke (`--steps 20`)** detached on an idle GPU now to prove it runs and matches Round 340 `S4` (`kt = 8` day-1 and `kt = 10` terminal state).

3. **The comparison (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_metrics.py`):**
   - Reuses `nemo_testcase_l4_orca2x_round1_pilot_metrics.py` and `nemo_testcase_l4_orca2x_round1_longrun.py` directly (Decision 160; zero re-implementation of metric physics).
   - Applies the ORCA2X metric card month by month across the 12 calendar months of the 365-day no-leap year (`[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]` days, plus partial-month support for smoke/30-day runs) for both arms (NEMO twin B vs twin A, and legoESM candidate vs NEMO arm A).
   - Reports every metric next to the **Decision-185 REFERENCE bars**:
     - Monthly RMS SST and band T (`global, tropical_20S_20N, north_40N_90N, south_90S_40S, depth_0_100m, depth_100_500m, depth_500_2000m, depth_2000m_bottom`): `0.01 K`
     - Monthly RMS SSS and band S (same 8 regions/layers): `0.002 psu`
     - Monthly RMS SSH: `1 mm` (`0.001 m`)
     - Surface KE: `1%` (`0.01` relative error of NEMO monthly mean)
     - Barotropic streamfunction $\psi$: RMS `0.1 Sv` and spatial correlation `>= 0.999`
     - Drake Passage transport: `0.5 Sv`
     - Mixed-layer depth (MLD $\Delta\sigma_0 = 0.01\text{ kg m}^{-3}$): RMS `1 m`
   - **REFERENCE ONLY (`verdict: None`, `bar_status: "REFERENCE_ONLY"` — no pass/fail).**
