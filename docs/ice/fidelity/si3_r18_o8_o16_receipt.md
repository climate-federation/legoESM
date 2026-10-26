# SI3 round 18 receipt — O8–O16 master-record acquisition tool

## Outcome

The operator-only acquisition tool is ready and its read-only preflight passes.
No real NEMO integration was run. It produces one shared, one-rank, 100-step
record under the unmodified production ice namelist. The O7 log verifier now
accepts an explicit expected step count, preserving its two-step default while
admitting this preregistered 100-step window. The stage order is taken from
NEMO's own thermodynamics call chain
(`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:174-229`).

The revised schema records only `qlead` at frazil entry, before NEMO initializes
`ht_i_new` and `fraz_frac`; those outputs first appear at frazil exit. Category
records omit never-packed `a_ib_1d` and late-packed `h_ib_1d`, include the
`o_i_1d` value consumed when `oa_i` is written back, and share a full
pre-category state snapshot so cells skipped by `a_i <= epsi10` remain defined.

Resume-4 review repairs move kind 3 after all nine `dh_*` zero assignments and
make the reader refuse a nonzero category-entry diagnostic. A source scanner now
derives each O8--O16 routine/callee read-set from NEMO ppsrc and refuses any
array or scalar without binary-boundary, header, metadata, resolved-namelist, or
pinned binary/source coverage. That gate found and added `qcn_ice_1d` as an O9
input. O15 now records `sst_m`; `vm_ibr` is dropped because this no-XIOS path
does not call `ice_var_brine`; and the resolved `ln_drgice_imp=T` deck makes
kind 20 record the filled full-halo `rCdU_ice` output.

## Complete boundary map

Every row names the exact record kinds and the NEMO-owned state recorded on
both sides. Common category state includes `nptidx`, surface/ocean forcing,
ice/snow temperatures and enthalpies, thickness/salinity, all `dh_*`
diagnostics (zero at kind 3), gathered `qcn_ice`, ice age `o_i`, and all thermodynamic heat/water/
salt flux accumulators. Common global state includes category prognostics,
every post-`ice_var_agg(2)` aggregate/equivalent field that NEMO fills
(including `tm_si`, excluding unfilled `vm_ibr`), pond
state, fluxes, `h_i_b`, `a_i_b`, `t_bo`, `ht_i_new`, `fraz_frac`, `sss_m`,
`sst_m`, `rn_amax_2d`, `tmask`, `smask0`, `hi_max`, `hi_mean`, precipitation and category
surface fluxes.

| Rung | Recorded boundaries | Additional inputs/constants mapped to recorded names |
|---|---|---|
| O8 | 1 frazil entry; 2 frazil exit | `qlead`, `at_i`, `smask0`, `u_ice`, `v_ice`, `umask`, `vmask`, `utau_ice`, `vtau_ice`, `hi_max`, `ht_i_new`, `fraz_frac`; scalar `rn_hinew` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:393-477`) |
| O9 | 21 pre-category state; 3 category entry; 4 ZDF exit; 5 `dh` exit; 6 first-temperature exit; 17 final Picard state at steps 1, 2, 100 | complete state for cells skipped by the category loop; zero-entry `dh_*`; `qcn_ice_1d`; `o_i_1d`; `t_su`, `qns_ice`, `t_i`, `t_s`; filled `zkappa_i/s(0:nlay)`, `zradtr_i/s(0:nlay)`, and `zradab_i/s(1:nlay)` only—no layer-0 or single-cell tridiagonal scratch; scalar `rcnd_s`, `rn_kappa_i/s`, `rn_snwblow`, `rn_sinew`, `nn_icesal` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:112-139`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:387-577`) |
| O10 | 21 pre-category state; 6 salinity entry; 7 salinity exit; 8 second-temperature exit | complete category state and fluxes; `rn_icesal`, `rn_sal_gd`, `rn_time_gd`, `rn_sal_fl`, `rn_time_fl`, `ln_drainage`, `ln_flushing`, `nn_drainage`, `nn_flushing`, `rn_sal_himin`, `rn_flushrate`, `rn_vbrc`, `nn_sal_scheme`, `rn_alpha_RJW`, `rn_Rc_RJW`, `nn_liquidus`, `rn_alpha_GN`, `rn_Rc_GN`, `rn_alpha_CW`, `ln_sal_chk`, plus the six shared O9 constants (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:115-687`) |
| O11 | 21 pre-category state; 8 lateral-melt entry; 9 lateral-melt exit | complete category state and flux accumulators plus configured `rn_beta` and `rn_dmin` at both sides (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:201-210`) |
| O12 | 10 pond entry/category done; 11 pond exit | common global state; `dh_i_sum_2d`, `dh_s_sum_2d`, `sprecip`, `tprecip`, `qsr_ice`, `qns_ice`; `nice_pnd`, `rn_apnd_min`, `rn_apnd_max`, `rn_pnd_flush`, `ln_pnd_lids`, `ln_pnd_rain`, `ln_pnd`, `ln_pnd_LEV`, `ln_pnd_CST`, `ln_pnd_TOPO`, `rn_apnd`, `rn_hpnd`, `ln_pnd_alb`, `rn_pnd_hl_min`, `rn_pnd_hl_max` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:162-264`) |
| O13 | 11 ITD entry; 12 ITD exit | common global state plus explicit `h_i_b`, `a_i_b`, `hi_max`, `rn_amax_2d` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90:175-232`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90:464`) |
| O14 | 12 `ice_thd_do` entry; 13 exit | common global state plus explicit `t_bo`, `ht_i_new`, `fraz_frac`, `sss_m`, `hi_max`, `rn_amax_2d` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:196-206`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:304`) |
| O15 | 13 correction entry; 14 exit | common global state plus explicit `sss_m`, `sst_m`, `hi_max`, `hi_mean`, `rn_amax_2d` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icecor.f90:94-118`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90:632-668`) |
| O16 | 18 pre-aggregation; 15 `ice_update_flx` entry; 16 ocean-format exit; 19 `ice_update_tau` entry; 20 stress exit | post-`ice_var_agg(2)` state; then `at_i_b`, `a_i_b`, `qemp_ice`, `smask0`, `qcn_ice`, `qml_ice`, `qtr_ice_top/bot`, `wfx_snw_dyn`, `tmask_k1`, `cloud_fra`, all seven `rn_alb_*` scalars and the other flux inputs; stress entry includes `uu`, `vv`, `umask`, `vmask`, ice/ocean velocity, `drag_io`, `at_i`, and full halo `utau`/`vtau`; exits retain full `fr_i`, `utau`, `vtau`, `rCdU_ice` and interior `taum` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icestp.f90:254-257`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceupdate.f90:119-223`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceupdate.f90:332-423`) |

The per-category Picard counter is reset at every category entry, so no value
can leak between categories. The scalar header contains all 58 shared,
salinity, pond, lateral-melt and albedo constants by recorded name.

## Storage, disk and wall-clock gates

The schema-derived worst case is **140,714,793,940 bytes (131.05 GiB)**.
Per-rung projected sizes are O8 0.22 GiB, O9 40.91 GiB, O10 36.62 GiB, O11
26.61 GiB, O12–O15 17.39 GiB each, and O16 16.43 GiB. Shared boundaries are
referenced rather than copied.

The driver gates free and final evidence disk use at **150,000,000 KiB
(143.05 GiB)** and the total preflight/build/two-run acquisition wall clock at
**14,400 seconds (4 hours)**. It records actual record bytes, total evidence
KiB and elapsed seconds before admission; admission rejects a missing,
inconsistent or over-budget measurement.

## Preflight evidence

The generated driver passed all ten phases:

- fail-closed schema/readers, signed-zero, nonzero category-entry `dh_*`, and
  corruption plants for all nine projections;
- mechanically generated routine/callee array and scalar read-sets for every
  O8--O16 rung, with missing `qlead`, `sst_m`, and `rn_hinew` plants;
- exact 100-step production deck and group-membership plants;
- patch placement against original NEMO `.F90` sources preprocessed exactly once;
- production-flag FP-passivity census for all eight patched units;
- production-flag plus `-fcheck=bounds` compilation of the real writer and all
  eight real patched units, followed by a fixture that executes every emitted
  kind 1–21 with distinct data, retained `-0.0`, nonzero lateral-melt/albedo
  values, and a quiet-NaN layer-0 absorption sentinel that is not written;
- same-program comparison against a separately copied observer-free build and
  independently patched expected sources, with a same-program plant;
- before/after digests for every pre-existing file in each in-scope `cfgs/`
  tree, a digest-mismatch plant, and unlink-before-edit handling for copied
  symlinked build metadata;
- an end-to-end 100-step stub-NEMO `--run` and `--admit-existing` through real
  preflight, patching, same-program and `mpirun` call sites, repeated-run refusal,
  failed-run evidence/build retention, a truncation of `run.sh` itself, and both
  `--move-aside-builds` paths. The fixture replaces only `nemo.exe` and the
  sandbox-blocked `mpirun` executable; the driver call site is unchanged.

The explicit operator run remains required. Refused evidence and builds are
retained, and only `--move-aside-builds` enables a rebuild after failure. No
physics result or fidelity claim is made here.

## Gates

- Acquisition unit checks: 25 passed; the complete clean-tree preflight passed
  separately through all ten phases with 45 plants and the full driver retention fixture.
- Required default receipt-citation gate: 35 passed. This receipt's 16 live
  source citations pass; every built-in citation plant fires.
- Seven CI ratchet files: 11,954 passed, 4 skipped, 14 failed. All 14
  failures name pre-existing, untouched files; none is in this acquisition,
  citation-map, test or receipt change.

independent review: requested from operator (Claude-4.6 and Argon)
