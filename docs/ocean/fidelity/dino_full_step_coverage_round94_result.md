# DINO full-step registry harvest, round 94

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Gate provenance and verdict

The master full-step gate from `fidelity/dino-full-step-coverage-codex` was
fetched at `97d3d7188eb5`. That commit is already an ancestor of this branch,
so no duplicate cherry-pick was needed. The gate now SHA-admits all 34
committed split-explicit/momentum-chain result documents plus the committed
ZDF chain-end receipt before applying any promotion.

| quantity | result |
|---|---:|
| NEMO `stp_MLF` calls enumerated | 116 |
| measured `COVERED` calls | 34 |
| active rows still `COVERED (row UNMEASURED)` | 3 |
| explicit `WAIVED` calls | 79 |
| `UNCOVERED` calls | 0 |
| **strict active coverage** | **34/37 = 91.891892%** |
| registry accounting, waivers included | 116/116 = 100% |

`COVERED` means a committed lane receipt measures and dispositions the call;
it does not silently upgrade a conditional or Rule-1b result to bit-exact.
The Redi/tracer-tail promotion therefore retains round 93's explicit Rule-1b
qualification for S `zfw`.

## Residual active and waived table

The complete active residual is:

| NEMO line | routine | disposition | reason |
|---:|---|---|---|
| 204 | `ldf_dyn` | UNMEASURED | its coefficient is only folded into the measured `dyn_ldf` consumer; no coefficient-isolated row |
| 387 | `tra_sbc` | UNMEASURED | the surface forcing fields are covered, but their application into the tracer RHS is not isolated |
| 394 | `tra_qsr` | UNMEASURED | surface `qsr` is covered, but its two-band vertical redistribution is not measured |

The 79 waivers remain explicit, machine-readable per call in the harvest JSON.
Their non-overlapping accounting is:

| waiver class | count | exact members |
|---|---:|---|
| I/O, diagnostics, calendar, or non-feedback control | 29 | `iom_init`, `dia_mlr_iom_init`, `iom_init_closedef` (2), `dia_hth_init`, `dia_ptr_init`, `dia_ar5_init`, `dia_hsb_init`, `dia_25h_init`, `mlf_dia`, `iom_swap`, `iom_setkt` (5), `day`, `dia_cfl`, `dia_dct`, `dia_hth`, `dia_ar5`, `dia_ptr`, `dia_wri` (2), `dia_detide`, `dia_mlr`, `dia_hsb`, `rst_write`, `stp_ctl` |
| namelist-off or compiled-out physics | 30 | `tide_update`, `sbc_apr`, `bdy_dta`, `isf_stp`, `sto_par`, `sto_pts`, `bbl`, `dom_qco_r3c` spg-exp variant, `wAimp` (2), `dyn_dmp`, `dyn_asm_inc`, `asm_bkg_wri`, `bdy_dyn3d_dmp`, `dyn_osm`, `diurnal_layers`, `ldf_eke`, `trc_stp`, `tra_asm_inc`, `tra_isf`, `tra_bbc`, `tra_bbl`, `tra_dmp`, `bdy_tra_dmp`, `tra_mfc`, `tra_osm`, `tra_npc`, `sto_rst_write`, `dia_obs`, `sbc_cpl_snd` |
| upstream value covered through its measured consumer | 2 | `eos` feeding `ldf_slp`; `eos` feeding `dyn_hpg` |
| physics-inert campaign instrumentation | 18 | `OPEN/WRITE r3c_dump`, four `stp_dump_krhs`, two `stp_dump_state_and_bt`, five `stp_dump_ts_krhs`, `trddump_acc_baro`, `trddump_acc_plant`, and three `trddump_acc_state` calls |
| **total waived** | **79** | full line/routine/reason rows are in the JSON artifact |

The gate's synthetic unaccounted-call plant passes: adding one call with no
disposition makes the gate fail. Focused unit tests also pin the 34/37
fraction and exact three-row residual.

## Imported lane promotions

The harvested receipts promote the two WZV calls, `dyn_zdf`, `ssh_atf`,
`traldf_iso_lap`, and `tra_zdf` from registry-visible-but-unmeasured to
measured. All other lane findings were already represented by covered call
rows; their receipts are still hash-admitted so omission or later editing
fails closed.
