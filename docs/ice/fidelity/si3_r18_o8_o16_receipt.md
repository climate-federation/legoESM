# SI3 round 18 receipt — O8–O16 master-record acquisition tool

## Outcome

The operator-only acquisition tool is ready and its read-only preflight passes.
No real NEMO integration was run. It produces one shared, one-rank, 100-step
record under the unmodified production ice namelist. The O7 log verifier now
accepts an explicit expected step count, preserving its two-step default while
admitting this preregistered 100-step window. The stage order is taken from
NEMO's own thermodynamics call chain
(`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:174-229`).

## Complete boundary map

Every row names the exact record kinds and the NEMO-owned state recorded on
both sides. Common category state includes `nptidx`, surface/ocean forcing,
ice/snow temperatures and enthalpies, thickness/salinity, all `dh_*`
diagnostics and all thermodynamic heat/water/salt flux accumulators. Common
global state includes category prognostics, aggregate/equivalent fields, pond
state, fluxes, `h_i_b`, `a_i_b`, `t_bo`, `ht_i_new`, `fraz_frac`, `sss_m`,
`rn_amax_2d`, `hi_max`, `hi_mean`, precipitation and category surface fluxes.

| Rung | Recorded boundaries | Additional inputs/constants mapped to recorded names |
|---|---|---|
| O8 | 1 frazil entry; 2 frazil exit | `qlead`, `ht_i_new`, `fraz_frac`; scalar `rn_hinew` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:393-477`) |
| O9 | 3 category entry; 4 ZDF exit; 5 `dh` exit; 6 first-temperature exit; 17 final Picard state at steps 1, 2, 100 | `t_su`, `qns_ice`, `t_i`, `t_s`; `zkappa_i/s`, `zradab_i/s`, `zradtr_i/s`; `ztrid`, `zindterm`, `zdiagbis`, `zindtbis`; scalar `rcnd_s`, `rn_kappa_i/s`, `rn_snwblow`, `rn_sinew`, `nn_icesal` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:112-139`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:387-577`) |
| O10 | 6 salinity entry; 7 salinity exit; 8 second-temperature exit | complete category state and fluxes; `rn_icesal`, `rn_sal_gd`, `rn_time_gd`, `rn_sal_fl`, `rn_time_fl`, `ln_drainage`, `ln_flushing`, `nn_drainage`, `nn_flushing`, `rn_sal_himin`, `rn_flushrate`, `rn_vbrc`, `nn_sal_scheme`, `rn_alpha_RJW`, `rn_Rc_RJW`, `nn_liquidus`, `rn_alpha_GN`, `rn_Rc_GN`, `rn_alpha_CW`, `ln_sal_chk`, plus the six shared O9 constants (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:115-687`) |
| O11 | 8 lateral-melt entry; 9 lateral-melt exit | complete category state and flux accumulators at both sides (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:201-210`) |
| O12 | 10 pond entry/category done; 11 pond exit | common global state; `dh_i_sum_2d`, `dh_s_sum_2d`, `sprecip`, `tprecip`, `qsr_ice`, `qns_ice`; `nice_pnd`, `rn_apnd_min`, `rn_apnd_max`, `rn_pnd_flush`, `ln_pnd_lids`, `ln_pnd_rain`, `ln_pnd`, `ln_pnd_LEV`, `ln_pnd_CST`, `ln_pnd_TOPO`, `rn_apnd`, `rn_hpnd`, `ln_pnd_alb`, `rn_pnd_hl_min`, `rn_pnd_hl_max` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:162-264`) |
| O13 | 11 ITD entry; 12 ITD exit | common global state plus explicit `h_i_b`, `a_i_b`, `hi_max`, `rn_amax_2d` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90:175-232`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90:464`) |
| O14 | 12 `ice_thd_do` entry; 13 exit | common global state plus explicit `t_bo`, `ht_i_new`, `fraz_frac`, `sss_m`, `hi_max`, `rn_amax_2d` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:196-206`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:304`) |
| O15 | 13 correction entry; 14 exit | common global state plus explicit `sss_m`, `hi_max`, `hi_mean`, `rn_amax_2d` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icecor.f90:94-118`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90:632-668`) |
| O16 | 18 pre-aggregation; 15 `ice_update_flx` entry; 16 ocean-format exit | pre-aggregation category state; then `at_i_b`, `a_i_b`, `qcn_ice`, `qml_ice`, `qtr_ice_top/bot`, `wfx_snw_dyn`, `tmask_k1`, `cloud_fra` and all other update inputs; finally all ocean boundary outputs (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icestp.f90:254-257`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceupdate.f90:119-223`) |

The per-category Picard counter is reset at every category entry, so no value
can leak between categories. The scalar header contains all 51 shared,
salinity and pond constants by recorded name.

## Storage, disk and wall-clock gates

The schema-derived worst case is **131,392,072,220 bytes (122.37 GiB)**.
Per-rung projected sizes are O8 0.10 GiB, O9 33.62 GiB, O10 29.42 GiB, O11
19.62 GiB, O12–O15 17.27 GiB each, and O16 16.05 GiB. Shared boundaries are
referenced rather than copied.

The driver gates free and final evidence disk use at **150,000,000 KiB
(143.05 GiB)** and the total preflight/build/two-run acquisition wall clock at
**14,400 seconds (4 hours)**. It records actual record bytes, total evidence
KiB and elapsed seconds before admission; admission rejects a missing,
inconsistent or over-budget measurement.

## Preflight evidence

The generated driver passed all ten phases:

- fail-closed schema/readers, signed-zero and corruption plants for all nine
  projections;
- exact 100-step production deck and group-membership plants;
- patch placement against real preprocessed NEMO sources;
- production-flag FP-passivity census for all seven patched units;
- production-flag plus `-fcheck=bounds` compilation of the real writer and all
  seven real patched units, followed by an executed frazil writer fixture with
  distinct data and retained `-0.0`;
- same-program comparison against a separately copied observer-free build and
  independently patched expected sources, with a same-program plant;
- an end-to-end 100-step stub-NEMO `--run` and `--admit-existing`, repeated-run
  refusal, failed-run evidence/build retention, syntax plant, and both
  `--move-aside-builds` paths.

The explicit operator run remains required. Refused evidence and builds are
retained, and only `--move-aside-builds` enables a rebuild after failure. No
physics result or fidelity claim is made here.

## Gates

- Acquisition unit tests and complete preflight: 22 passed.
- Required default receipt-citation gate and this receipt's 15 live source
  citations: PASS; the shifted-line control exits nonzero with a FAIL verdict.
- Seven CI ratchet files: 14,959 passed, 2,552 skipped, 9 failed. All nine
  failures name pre-existing, untouched files; none is in this acquisition,
  citation-map, test or receipt change.

independent review: operator (Claude)
