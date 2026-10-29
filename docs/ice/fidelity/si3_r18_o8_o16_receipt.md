# SI3 round 18 receipt — O8–O16 master-record acquisition tool

## Outcome

The operator-only acquisition completed and the retained evidence is admitted.
It produced one shared, one-rank, 100-step record under the unmodified
production ice namelist. The O7 log verifier now
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
pinned binary/source coverage. Each rung is now checked against only its
registered entry projection; the gate refuses removal of `snwice_mass` from
O16 kind 15. O15 records `sst_m`; `vm_ibr` is dropped because this no-XIOS path
does not call `ice_var_brine`; and the nine diagnostics first filled by
`ice_var_agg(2)` are absent from kinds 10--14 but retained at kind 15.

Resume-6 review repairs retain indexed, conditional and read-modify-write left
hand sides as inputs, and gate every rung against only its `ENTRY_KINDS` rather
than subtracting fields found in unrelated record kinds. The scanner resolves
the `ice_var_snwfra` and `ice_var_snwblow` generic interfaces
(`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:105-111`) to their concrete
procedures (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:1578-1643`), follows
function references, and records `nn_snwfra`. O16 now starts from both calls at
the true pre-aggregation boundary and records the `t_si`, `hi_max`, and `sss_m`
inputs they consume (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:185-190`,
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:296-299`,
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:362-387`,
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:505-514`).

Resume-7 corrects the executed drag branch. SAS enters `nemo_init` and then
`stp` without calling `zdf_phy_init`
(`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/nemogcm.f90:138-161`,
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/nemogcm.f90:355-391`), so
`zdf_drg_init` is unreachable and the module initializer leaves
`ln_drgice_imp=.FALSE.`
(`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/zdfdrg.f90:61`). The gate proves that
reachability from the complete real SAS ppsrc and requires real
`output.namelist.dyn` to omit both `&namdrg` and `ln_drgice_imp`. Kind 20 no
longer records `rCdU_ice`, which is assigned only on the unreachable true
branch (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceupdate.f90:400-418`).

The filled-or-not rule also removes `qcn_ice_top_1d`, `qcn_ice_bot_1d`, and
`qtr_ice_bot_1d` from kind 3: the category-entry 2D-to-1D gather does not fill
them (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:387-484`); they are later
stage outputs scattered by case 2
(`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:484-567`). A generic
previous-category-stale plant proves that adding an ungathered kind-3 field is
refused. `qt_atm_oi` and `qt_oce_ai` are likewise omitted from entry records
and classified as values produced inside `ice_update_flx` before consumption
(`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceupdate.f90:119-175`).

A shared allocation gate parses NEMO's preprocessed `ALLOCATE` statements and
explicit automatic-array declarations, expands `A2D(n)`/`T2D(n)`, resolves the
deck's `Ni0glo=180`, `Nj0glo=148`, `jpkglo=31`, and `nn_hls=2`, and validates
all 285 writer `put_*` actual arguments. It still derives `rCdU_ice` as 182 by
150 from `Nis0-1:Nie0+1,Njs0-1:Nje0+1` because the bounds fixture executes the
real routine, but the binary writer does not record that unfilled array. The
same catalog generates the bounds fixture allocations.

## Complete boundary map

Every row names the exact record kinds and the NEMO-owned state recorded on
both sides. Common category state includes `nptidx`, surface/ocean forcing,
ice/snow temperatures and enthalpies, thickness/salinity, all `dh_*`
diagnostics (zero at kind 3), gathered `qcn_ice`, ice age `o_i`, and all thermodynamic heat/water/
salt flux accumulators. Common global state includes category prognostics,
every aggregate/equivalent field already filled at that boundary (excluding
the nine `ice_var_agg(2)` diagnostics and unfilled `vm_ibr`), pond
state, fluxes, `h_i_b`, `a_i_b`, `t_bo`, `ht_i_new`, `fraz_frac`, `sss_m`,
`sst_m`, `rn_amax_2d`, `tmask`, `smask0`, `hi_max`, `hi_mean`, precipitation and category
surface fluxes.

| Rung | Recorded boundaries | Additional inputs/constants mapped to recorded names |
|---|---|---|
| O8 | 1 frazil entry; 2 frazil exit | `qlead`, `at_i`, `smask0`, `u_ice`, `v_ice`, `umask`, `vmask`, `utau_ice`, `vtau_ice`, `hi_max`, `ht_i_new`, `fraz_frac`; scalar `rn_hinew` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:393-477`) |
| O9 | 21 pre-category state; 3 category entry; 4 ZDF exit; 5 `dh` exit; 6 first-temperature exit; 17 final Picard state at steps 1, 2, 100 | complete state for cells skipped by the category loop; zero-entry `dh_*`; gathered `qcn_ice_1d` but no not-yet-filled top/bottom conductive or transmitted-bottom flux at kind 3; those stage outputs at kind 4; `o_i_1d`; `t_su`, `qns_ice`, `t_i`, `t_s`; filled `zkappa_i/s(0:nlay)`, `zradtr_i/s(0:nlay)`, and `zradab_i/s(1:nlay)` only—no layer-0 or single-cell tridiagonal scratch; scalar `rcnd_s`, `rn_kappa_i/s`, `rn_snwblow`, `rn_sinew`, `nn_icesal` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:112-139`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:387-577`) |
| O11 | 21 pre-category state; 8 lateral-melt entry; 9 lateral-melt exit | complete category state and flux accumulators plus configured `rn_beta` and `rn_dmin` at both sides (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:201-210`) |
| O12 | 10 pond entry/category done; 11 pond exit | common global state; `dh_i_sum_2d`, `dh_s_sum_2d`, `sprecip`, `tprecip`, `qsr_ice`, `qns_ice`; `nice_pnd`, `rn_apnd_min`, `rn_apnd_max`, `rn_pnd_flush`, `ln_pnd_lids`, `ln_pnd_rain`, `ln_pnd`, `ln_pnd_LEV`, `ln_pnd_CST`, `ln_pnd_TOPO`, `rn_apnd`, `rn_hpnd`, `ln_pnd_alb`, `rn_pnd_hl_min`, `rn_pnd_hl_max` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:162-264`) |
| O13 | 11 ITD entry; 12 ITD exit | common global state plus explicit `h_i_b`, `a_i_b`, `hi_max`, `rn_amax_2d` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90:175-232`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90:464`) |
| O14 | 12 `ice_thd_do` entry; 13 exit | common global state plus explicit `t_bo`, `ht_i_new`, `fraz_frac`, `sss_m`, `hi_max`, `rn_amax_2d` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:196-206`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:304`) |
| O15 | 13 correction entry; 14 exit | common global state plus explicit `sss_m`, `sst_m`, `hi_max`, `hi_mean`, `rn_amax_2d` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icecor.f90:94-118`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90:632-668`) |
| O16 | 18 pre-aggregation; 15 `ice_update_flx` entry; 16 ocean-format exit; 19 `ice_update_tau` entry; 20 stress exit | pre-aggregation `t_si`, `sss_m`, `hi_max` and aggregate/equivalent inputs; kind 15 post-aggregation state including its nine diagnostics, `at_i_b`, `a_i_b`, `qemp_ice`, `smask0`, `qcn_ice`, `qml_ice`, `qtr_ice_top/bot`, `alb_ice`, `snwice_mass`, `wfx_snw_dyn`, `tmask_k1`, `cloud_fra`, all seven `rn_alb_*` scalars and the other flux inputs; `qt_atm_oi`/`qt_oce_ai` are stage-local outputs rather than entry fields; stress entry includes `uu`, `vv`, `umask`, `vmask`, ice/ocean velocity, `drag_io`, `at_i`, carried `tmod_io`, `utau_oce`, `vtau_oce`, and full halo `utau`/`vtau`; exits retain full `fr_i`, `utau`, `vtau` and interior `taum`, but not unfilled `rCdU_ice` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icestp.f90:254-257`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceupdate.f90:119-223`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceupdate.f90:332-423`) |

The per-category Picard counter is reset at every category entry, so no value
can leak between categories. The scalar header contains all 59 shared,
salinity, pond, lateral-melt, snow-fraction and albedo constants by recorded name.

## Storage, disk and wall-clock gates

The schema-derived worst case is **140,965,556,460 bytes (131.28 GiB)**.
Per-rung projected sizes are O8 0.23 GiB, O9 41.19 GiB, O10 36.87 GiB, O11
26.80 GiB, O12–O15 17.07 GiB each, and O16 16.97 GiB. Shared boundaries are
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
  O8--O16 rung against its entry projection, with missing `qlead`, `sst_m`,
  `snwice_mass`, `nn_snwfra`, read-modify-write `sfx_lam`, and `rn_hinew`
  plants, plus current-category freshness and previous-category-stale plants;
- exact 100-step production deck and group-membership plants;
- patch placement against original NEMO `.F90` sources preprocessed exactly once;
- production-flag FP-passivity census for all nine patched units; shared
  source-allocation checks for all writer calls, with missing-allocation,
  writer-mode, dummy/actual, and generated-fixture mismatch plants;
- production-flag plus `-fcheck=bounds` compilation of the real writer and all
  nine real patched units, followed by a fixture that executes every emitted
  kind 1–21 with distinct data, retained `-0.0`, nonzero lateral-melt/albedo
  values, the deck's real halo width, source-generated allocations, and a
  quiet-NaN layer-0 absorption sentinel that is not written;
- same-program comparison against a separately copied observer-free build and
  independently patched expected sources, with a same-program plant;
- before/after digests for every pre-existing file in each in-scope `cfgs/`
  tree, a digest-mismatch plant, and unlink-before-edit handling for copied
  symlinked build metadata;
- an end-to-end 100-step stub-NEMO `--run` and `--admit-existing` through real
  preflight, patching, same-program and `mpirun` call sites, repeated-run refusal,
  failed-run evidence/build retention, a truncation of `run.sh` itself, and both
  `--move-aside-builds` paths. The fixture replaces only `nemo.exe` and the
  sandbox-blocked `mpirun` executable; the driver call site is unchanged. The
  stub copies checksum-pinned `ocean.output`, `output.namelist.dyn`, and
  `output.namelist.ice` from the admitted real O7 SAS run rather than inventing
  a namelist group. Default-value, call-tree, and fabricated-output drag plants
  are all refused.

The explicit operator run completed successfully. Refused evidence and builds
remain retained by the tool, and only `--move-aside-builds` enables a rebuild
after failure. No physics result or fidelity claim is made here.

## Acquisition admission

`STATUS: ADMITTED`

The admitted master contains 4,815 rank-0 binary records totaling
**81,440,242,576 bytes (75.85 GiB)**. The run manifest records **2,342.5088832378387
seconds (39 minutes 2.509 seconds)** wall clock. The measured evidence footprint
is 85,771,108 KiB. The producer is
`cd34f1be648a05abbb23e4cb6bb2a9ed66e60fc5`; Amendment (2026-10-29 provenance audit): this was a pre-rebase commit and is no longer reachable. The run-manifest SHA-256 is
`bfb0b71d1500b7bc91c68a9e39c8af7a654b139258c0635ad5b1f25284b31fd0`,
the resource-usage SHA-256 is
`b30a5febe9248e5008999cb419953dad40c78658542a3c7221d301825270967c`, and
the complete record-inventory SHA-256 is
`fbd3cb4d3450f428d8444dcb145bec3c1a832554e3e7e05d7f5b6a159daadd85`.

Each SHA-256 below covers the retained relative-path projection manifest,
which contains one digest for every admitted record in that projection.

| Rung | Kinds | Records | Projection-manifest SHA-256 |
|---|---|---:|---|
| O8 | 1, 2 | 200 | `303a29d76034c3856e4cf6fa60ed551b7146e5decd3618038b058501c818ed82` |
| O9 | 21, 3, 4, 5, 6, 17 | 2,115 | `3d97b7bb71c91530a89f60498c8a84316829200e89d8c45470f5a8d1814ee0a3` |
| O10 | 21, 6, 7, 8 | 1,600 | `ded24ddbc825a9a331d763665e51559022c97288cb7c7f516b3a587693deb554` |
| O11 | 21, 8, 9 | 1,100 | `0c5197c52331975ff77d813f0154e0df6ea0aa1e0593e9ca062990d664465666` |
| O12 | 10, 11 | 200 | `fe395cab681d59a8c28a722cd7fc8222ec3ac0ee8c4e569d4043643fd4a47e50` |
| O13 | 11, 12 | 200 | `a94d126ddaee3194a053ef437d762707dff93045287392d4e0586aaee995a4aa` |
| O14 | 12, 13 | 200 | `8130e5ac516cf8c1549badbab2caf90cf5cce03f3167c317a10532b836572cd6` |
| O15 | 13, 14 | 200 | `e1cd64c354294a071aca8496b014b59eb0b8f7be93d0304219e5c94b192ea292` |
| O16 | 18, 15, 16, 19, 20 | 500 | `6a3f9ed6f9330af1c65b603726c2ba407a28683f2de1b2a9dcaf9fd7dd3dc5e0` |

## Gates

- Acquisition unit checks: 28 passed; the complete clean-tree preflight passed
  separately through all ten phases with 56 plants and the full driver retention fixture.
- Required default receipt-citation gate: 35 passed. The cumulative default
  report passes 274 citations and this receipt passes all 28 live citations;
  both have zero unmapped/failing citations and every built-in plant fires.
- Eight CI ratchet files: 6,413 passed, 2 skipped, 8 failed. All 8 failures
  name pre-existing, untouched files; none is in this acquisition,
  citation-map, test or receipt change.

independent review: Sonnet RUN; Argon RUN (operator review of resume 7)

## Resume 9 semantic rebase resolution

The rebase retains both sides of the citation-gate change: O10's composed
delete-and-reinsert reanchoring (including its moved-anchor completeness check)
and this round's receipt registration and full citation map. This round's
inserted map entries moved O10's self-cited gate block by 38 lines, so the O10
receipt citation and its matching map key alone were moved from 10234--10271 to
10272--10309. Their pinned endpoint text and 38-line extent are unchanged.
