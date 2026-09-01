# NEMO testcase lane 3 dynamics — phase-1 preregistration

Status: **PREREGISTERED BEFORE BUILD OR RUN.**  Scope is NEMO 5.0.2 oracle
only, rungs 3.1--3.3 from `si3_lane3_scoping_dossier.md`.  No legoESM
comparison or claim of alignment is authorized in this phase.

Date: 2026-09-01
Tracker: climate-federation/legoESM #1699 (issue contents unavailable from this
session because `api.github.com` was unreachable)
Oracle commit: `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`

## Copy and execution contract

Each build is a new copy under the NEMO `tests/` directory.  The shipped cases
remain byte-unchanged.  No file is excluded from any source copy and no source
file is deleted.

| rung | copied source | copied destination | run root |
|---:|---|---|---|
| 3.1 | `tests/ICE_ADV1D` | `tests/ICE_ADV1D_OMIP_L3` | `/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv1d` |
| 3.2 | `tests/ICE_ADV2D` | `tests/ICE_ADV2D_OMIP_L3` | `/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d` |
| 3.3 | `tests/ICE_ADV2D` | `tests/ICE_ADV2D_RHG_OMIP_L3` | `/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d_rhg` |

All builds use `arch-conda.fcm`, whose Fortran flags include
`-fdefault-real-8` (`tests/LOCK_EXCHANGE_OMIP_L1/BLD/arch_nemo.fcm:18-20`).
The runtime instrument also refuses `STORAGE_SIZE(1._wp) /= 64`.  Runs are
single-process CPU executions of `nemo.exe`: no `mpirun`, no GPU.

The copied cpp files resolve exactly to `key_si3 key_linssh key_vco_1d`.
These are the shipped selectors (`tests/ICE_ADV1D/cpp_ICE_ADV1D.fcm:1` and
`tests/ICE_ADV2D/cpp_ICE_ADV2D.fcm:1`) minus `key_xios`, as required by the
lane-1 no-XIOS build pattern.  Thus the built components remain OCE + SAS +
ICE.  `ln_icethd=.false.` is retained from the shipped ice files
(`ICE_ADV1D/EXPREF/namelist_ice_cfg:27`,
`ICE_ADV2D/EXPREF/namelist_ice_cfg:25`).

The initial conditions are generated only by each shipped case's own script
after its copied case has written `mesh_mask.nc`.  ICE_ADV1D's Python-2 print
syntax may be mechanically translated for Python 3; its formulas and output
schema (`make_initice.py:44-117`) must remain unchanged.  This is not an
analytic substitute.

## Declared deviations from the shipped cases

Every row below is either explicitly requested or run-output infrastructure.
No other model-state selector may differ.

| rung(s) | shipped value → receipt value | source and reason |
|---|---|---|
| all | `key_xios` present → absent | shipped cpp line 1; lane-1 CPU/no-XIOS pattern |
| all | `nlay_i=2` (3.1) or ref `10`; ref `nlay_s=5` → `3,3` | ORCA1 `namelist_ice_cfg:25-26`; dossier rung 3.1, inherited by 3.2/3.3 |
| 3.2, 3.3 | ref `jpl=5` → `1` | ORCA1 `namelist_ice_cfg:24`; shipped `ICE_ADV2D/EXPREF/README:48-55` says prescribed-velocity ADV2D works only with one category, and 3.3 retains the same testcase category layout |
| all | `ln_adv_Pra=F, ln_adv_UMx=T` → `T,F` | ORCA1 `namelist_ice_cfg:75-76`; dossier rungs 3.1--3.3 |
| all | `ln_icediachk=F` → `T` | shared ice ref :316; dossier phase structure requests SI3's conservation ledger |
| all | `nn_stock=0` → final step (`40` or `485`) | shared ocean ref :54; user requires final ice restart |
| all | shipped `cn_exp` → copied-rung name | output/restart namespace only |
| 3.3 | `ln_dynADV2D=T, ln_dynRHGADV=F` → `F,T` | shipped ICE_ADV2D ice cfg :35-39; dossier rung 3.3 |
| 3.3 | inherited `rn_ishlat=2` made explicit | shared ice ref :57; ORCA1 no-slip choice |
| 3.3 | inherited/ORCA1 `ln_str_H79=T, rn_pstar=2e4, rn_crhg=20` made explicit | ORCA1 ice cfg :52-54 |
| 3.3 | ref `ln_str_smooth=T` → `F` | ORCA1 ice cfg :55 |
| 3.3 | EVP/aEVP rows made explicit: `ln_rhg_EVP=T`, `ln_rhg_EAP=F`, `ln_aEVP=T`, `rn_creepl=2e-9`, `rn_ecc=2`, `nn_nevp=100`, `rn_relast=.333`, `nn_rhg_chkcvg=0` | ORCA1 cfg :70 over shared ice ref :109-116; `rn_relast` is recorded but inert in aEVP (`icedyn_rhg_evp.F90:236-247`) |
| 3.3 | ORCA1 `ln_landfast_L16=T` → shipped/ref `F`, explicitly retained | user §8 pick; landfast is UNVERIFIED-deferred to lane 4 |

The shipped/ref `nn_icesal=4` and active pond defaults remain untouched in
these dynamics cases.  They are not silently replaced by ORCA1 thermodynamic
choices.  The restart gate therefore follows the resolved testcase and must
VERIFY-load layer-salinity and pond moments when NEMO writes them.

## Step-entry frame registry

The committed `MY_SRC/icestp.F90` writes one binary stream frame per ice step,
immediately before `store_fields` creates the `*_b` copy
(`src/ICE/icestp.F90:151`).  Header: 16-byte magic `NEMO_L3_ICE_1`, then nine
native 32-bit integers: version, `kt`, `jpi`, `jpj`, `jpl`, `nlay_i`,
`nlay_s`, real storage bits, registry count.  Payload is native fp64 Fortran
order.

The fail-closed registry in `nemo_si3_oracle_gate.py` contains all 19 payload
arrays and cites both the `icestp.F90:154-171` entry location and their defining
or restart writer: `v_i, v_s, a_i, t_su, oa_i, a_ip,
v_ip, v_il, sv_i, u_ice, v_ice, stress1_i, stress2_i, stress12_i,
snwice_mass, snwice_mass_b, e_s, e_i, szv_i`.  State fields are
`STEP_ENTRY_CURRENT`; stresses and snow/ice mass are
`CARRIED_PREVIOUS_STEP`, except `snwice_mass_b`, which is the
`CARRIED_PREVIOUS_BEFORE_LEVEL` because `iceupdate.F90:190-193` advances the
pair in sequence.  An unknown field or trailing byte is fatal.

## Preregistered phenomenology gates

The comparisons use the first step-entry frame and the completed-run final ice
restart.  The raw fp64 concentration and volume drifts are reported but are
explicitly **UNCLASSIFIED**: no unsourced reduction tolerance is invented.
Conservation is instead checked against SI3's own online diagnostic and its
native thresholds (`icectl.F90:67-78,166-190`; shared ice ref :318-319): any
printed SI3 `: violation` diagnostic (conservation, negative state,
over-concentration, or advection-scheme violation; `icectl.F90:166-190`) is
REFUTE.  SI3 writes these messages to NEMO's shared `ocean.output`; pinned
source has no separate `ice.output` writer, so no synthetic ice log is claimed.
A failed predicate is REFUTE/DEBT; it is never relabelled after seeing output.

| rung | number produced | CONFIRM | REFUTE |
|---:|---|---|---|
| 3.1 | final y-spread and initial/final x-centroids for `a_i`, `v_i`, and `h_i=v_i/a_i`; each final field range | every y-spread is exactly zero; all three centroid distances from basin centre strictly decrease; all three final ranges remain nonzero, checking the documented unidirectional convergence, y homogeneity, consistency, and non-collapse of the initial shapes (`tests/README.rst:191-199`) | any opposite predicate |
| 3.2 | initial/final `max(a_i)`; final-minus-initial `max(h_i)`; count of cells above initial `max(h_i)` | `max(a_i)` is exactly preserved; thickness overshoot is strictly positive and its upper-side-lobe count is positive, checking the shipped Prather description (`ICE_ADV2D/EXPREF/README:48-65`; `tests/README.rst:181-186`) | maximum differs or no positive overshoot/side-lobe cell |
| 3.3 | maximum final x-velocity; maximum volume-state change | positive x-velocity under shipped `utau_ice=1.3 N/m2` (`usrdef_sbc.F90:93`) and nonzero state change; SI3's native process-conservation diagnostic remains clean | any opposite predicate; total concentration is not classified because `Hpiling` may rescale it (`icedyn.F90:209-232`) |

These checks certify only the documented testcase phenomenology.  They do not
measure legoESM alignment, post-dynamics internal stages, or landfast.

## Coverage and planted controls

For each run, the gate inventories every variable in `mesh_mask`, the final
SI3 restart, `output.namelist.dyn`, and `output.namelist.ice`.  Every mesh and
restart item must be VERIFIED-loaded or WAIVED with a nonempty source-based
reason; UNMEASURED is forbidden in those namespaces.  A separate Appendix-A contract expands the fixed restart state,
3+3 enthalpy/salinity layers, all mandatory Prather moments, salinity-mode
moments, pond moments, EVP stresses, `snwice_mass{,_b}`, and explicit
WAIVED-NOT-RESTARTED rows for Appendix A's reconstructed `t_s(l)`.  The gate
regenerates this exhaustive contract and requires exact equality, so deleting
a row is fatal.  Active floating contract rows must exist and be
VERIFIED-loaded as fp64; integer counters are VERIFIED-loaded in their native
integer type.  Every claimed namelist row is checked against an exact resolved
value, including `cn_exp` and `jpl`.

Two controls are preregistered and must exit red on every final run:

1. add `PLANTED_UNACCOUNTED_FILE_ARRAY` to the discovered mesh inventory;
2. add `1 m` to one in-memory `e1t` value before the uniform-metric check.

## Loud UNMEASURED rows

* legoESM alignment: **UNMEASURED** (out of dispatch scope)
* post-dynamics SI3 stage arrays: **UNMEASURED** (entry + final restart only)
* landfast L16: **UNMEASURED-deferred** to lane 4 by user decision
* tracker #1699 live issue contents: **UNVERIFIED** (network unavailable)
