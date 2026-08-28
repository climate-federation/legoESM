# DINO full-step coverage master registry

This lane enumerates the selected DINO MLF timestep; it changes no physics.
The source is `cfgs/DINO/MY_SRC/stpmlf.F90`, compiled with
`cpp_DINO.fcm` and resolved against
`RUN_SEQDUMP_D180_1R/{namelist_cfg,ocean.output}`. The gate parses all 148
direct `CALL` statements in source order and replaces dispatch wrappers with
the selected concrete routine plus the selectable alternatives marked
`INACTIVE`. Its 208 committed JSON rows are the
human- and machine-readable enumeration (`full_step_coverage.py --json`). An
inactive row quotes the cpp, namelist, output, or timestep condition that
excludes it.

The initial receipt audit admits only six ZDF-prefix rows from the committed
round-6 ZDF artifact. Existing campaign prose and gates point to candidate
evidence but do not contain an artifact checksum plus measured value, bar,
precision, and producer SHA, so they make no `VERIFIED` claim here. Coverage is
therefore **6/42 = 14.3%** among active physical/state rows; 36 are
`UNMEASURED`. `WAIVED` diagnostics/I/O and `INACTIVE` calls are excluded from
that denominator.

## Future receipt lanes, in NEMO execution order

| order | chain lane | rough registry rows | boundary of the lane |
|---:|---|---:|---|
| 1 | surface boundary forcing | 1 | `usrdef_sbc_oce` |
| 2 | NOW-level thermodynamic seam | 1 | isolate the second `eos_rab`; the row-3 receipt verifies only its downstream `bn2` |
| 3 | ZDF continuation | 3 | resume at the failing `zdf_tke` Prandtl operand, then `zdf_evd` and `lbc_lnk` |
| 4 | lateral coefficients and slopes | 4 | in-situ `eos` → `ldf_slp` → `ldf_tra`/`ldf_dyn` |
| 5 | pre-RHS free surface and continuity | 4 | `ssh_nxt`, first `dom_qco_r3c`, first `wzv_MLF`, and HPG `eos` |
| 6 | 3-D momentum RHS | 6 | `dyn_keg` → `dyn_zad` → two `vor_een` calls → `dynldf_lev_lap` → `hpg_sco` |
| 7 | split-explicit/free-surface commit | 5 | `dyn_spg_ts` → `div_hor` → `dom_qco_r3c` → `dyn_zdf` → second `wzv_MLF` |
| 8 | free-surface filter | 2 | `ssh_atf` → filtered `dom_qco_r3c` |
| 9 | tracer surface forcing | 2 | `tra_sbc` → `qsr_2BD` |
| 10 | tracer transport and diffusion | 4 | `ldf_eiv_trp_MLF` → `tra_adv_fct` → `traldf_iso_lap` → `tra_zdf_imp` |
| 11 | MLF reconciliation, boundaries, and filters | 4 | `mlf_baro_corr` → `finalize_lbc` → `tra_atf_qco` → `dyn_atf_qco` |

Counts are planning estimates over registry rows, not costs or measured climate
importance. The gate's ranked list separately labels its leverage ordering as
planning judgment. Each future lane must add a committed, checksum-validated
receipt at its pre-registered bar; adding prose or merely adding a checklist
row cannot change a disposition.

Run:

```bash
python scripts/validate/ocean_fidelity/dino_1226/full_step_coverage.py
python scripts/validate/ocean_fidelity/dino_1226/full_step_coverage.py --self-test
python scripts/validate/ocean_fidelity/dino_1226/full_step_coverage.py --json
```
