# ORCA2 Round 342 — Standalone ORCA2 One-Year (2,920-Step, S4 Statistical Arm) NEMO Acquisition Tool

This directory contains the operator acquisition runner for the standalone ORCA2 one-year (`2,920`-step, `365`-day) observer-free `K0` NEMO reference run and its reproducibility twin (`orca2_k0_year1_a_np2`, `orca2_k0_year1_b_np2`).

## Configuration

- **Target NEMO build:** `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y`
  - Built from `ORCA2_ICE_PISCES` with `del_key "key_si3 key_top"` and **empty `MY_SRC/`** (`0` files — observer-free).
- **Run root:** `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year`
  - Reproducibility Twin A (run first): `orca2_k0_year1_a_np2`
  - Candidate Twin B: `orca2_k0_year1_b_np2`
- **Namelist switches (`K0` + 1-year daily restarts):**
  - `nn_it000 = 1`, `nn_itend = 2920`, `nn_stock = 8`, `nn_write = 2920`, `ln_rst_list = .false.`
  - `nn_ice = 1`, `ln_drgice_imp = .false.`, `ln_mxl0 = .true.`, `nn_mxlice = 0`, `ln_icebergs = .false.`
  - `cn_fwb = 'none'`, `sn_apr = 'slp'`, `sn_ice = 'ice_cover_clim', -12., 'ice_cover', .true., .true., 'yearly', '', '', ''`
- **Outputs per twin arm:**
  - 365 daily ocean restart pairs `ORCA2_{kt:08d}_restart_{0000,0001}.nc` for `kt = 8, 16, ..., 2920` (`730` `.nc` files per twin)
  - `ocean.output` and `output.namelist.dyn`
  - `producer_manifest.json` and `round342_s4_year_admission.json`
- **Wall-time budget:** ~55 min stepping per arm; 90 min (`5,400 s`) timeout per arm.

## Usage

```bash
# Read-only preflight + Checklist 1-16 controls + stub pipeline (never runs NEMO)
bash scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_acquisition/run.sh --preflight-only

# Operator NEMO build + Twin A + Twin B + admission
bash scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_acquisition/run.sh --run

# Admit existing Twin A + Twin B in place (or quarantine to refused/ on failure)
bash scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_acquisition/run.sh \
  --admit-existing /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year
```
