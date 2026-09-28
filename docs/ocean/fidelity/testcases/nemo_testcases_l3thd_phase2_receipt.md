# NEMO testcase fidelity receipt — lane 3b, SI3 thermodynamics phase 2

Tracker: `climate-federation/legoESM#1699`

## Verdict

**Phase-2b continuation:** the former first-divergence stopping scope below is
superseded by the committed full-year measurement in
`nemo_testcases_l3thd_phase2b_receipt.md`.  That continuation reads all 70,080
frames, retains the kt3 first operator divergence, and reports later rows as
measured debt.  The statements below describe the earlier Phase-2 gate and are
not a claim that later frames remain unmeasured.

**DEBT.**  The reviewed `kt1.POST_ZDF.t_su` discrepancy is owned and fixed:
the former card reconstructed a ZDF-entry heat flux from a post-ZDF value at
the wrong Picard time level.  Replacing only that scalar with NEMO's directly
dumped entry value reduces the `t_su` error from `6.51945413210342e-08 K` to
`1.4210854715202004e-13 K` (a `458765.8`-fold improvement and normalized
error `5.562656545875019e-16`).  Every registered boundary at `kt=1` and
`kt=2` is now at the `1e-15` bar.

The Phase-2 post-fix sweep stopped at the next measured debt,
`kt3.POST_ZDF.e_i`: absolute error `7.152557373046875e-07 J m-3`, oracle
scale `279433808.11424434 J m-3`, normalized error
`2.559660701514903e-15`.  At that dispatch, no boundary after that row and no
later step was an accepted claim.  Phase 2b has since measured them under its
separately preregistered operator and continuous protocols.  The ordinary
Phase-2 CLI exits 1 after writing its complete JSON result; a scientific debt
is not a green gate.

### Coupled-rung coverage debt

| coupled component | status | reason |
|---|---|---|
| NEMO bulk-flux computation (`sbcblk`) | **DEBT / NOT CERTIFIED BY THIS RUNG** | The isolated legoESM column consumes NEMO's dumped entry `qns_ice` and `dqns_ice`.  It therefore tests SI3 thermodynamics downstream of the bulk-flux boundary, but does not independently recompute or certify NEMO's bulk fluxes.  The upstream path is `sbcblk.F90:1273,1480-1491` to `icestp.F90:201,206`. |

All comparisons ran on CPU after `set_policy(PrecisionPolicy.fp64())`.
Storage, compute, accumulation, and control policy dtypes printed `float64`;
every compared legoESM array printed `float64`.

## Phase-1 citation corrections landed first

Commit `8d3a14c348` makes the five review-only corrections before Phase 2:

- `ln_ECMWF=.false.` is inherited from `SHARED/namelist_ref:236`; ORCA1
  `EXPREF/namelist_cfg:129-142` never sets it explicitly.
- ice-reference citations are `ln_cat_usr:43`, `rn_snwblow:141`,
  `nn_flxdist:143`, and `ln_frazil:191`.

The Phase-2 preregistration is commit `bb227d2b71`; the operand-owner
preregistration is commit `15de2569af`.  Copy-only oracle instrumentation is
commit `3a348ab5c1`, and the committed implementation measured below is
`dbab66555537736d3430561f26a31461187071e1`.

## Pre-implementation search

Before implementation, the search covered every Python file below
`packages/ice/legoesm/ice/`, dossier §4 and Appendix B, the Phase-1 frame
registry/gate, the shared Thomas solver, and the ocean `ConstantsConfig`
pattern.  Findings, recorded before implementation in the preregistration:

- Active thermodynamics was the zero-layer path in `sea_ice.py`; `scm.py`
  already supplied the standalone column driver.  The SI3 choice therefore
  belongs inside those modules, not in a second ice model.
- `state.py` had bulk thickness, temperature, snow, and salinity but no
  three-layer ice/snow enthalpies or ice age.  `snow.py` and `brine.py` had
  reusable bulk operations but not the ordered SI3 sequence or option-2
  salinity profile.
- `_future/bitz_lipscomb.py` was explicitly parked and unwired.  It had an
  ice-only shared-Thomas solve and Untersteiner conductivity, but no snow
  layers, P07 conductivity, `dqns` Picard iteration, `ice_thd_dh`, option-2
  salinity, or `ice_thd_do`.
- `packages/core/legoesm/timestepping/tridiagonal.py` is the reusable JAX
  Thomas solve.  `packages/ocean/legoesm/ocean/constants_config.py` supplies
  the named-config precedent.

The parked implementation is now promoted to
`packages/ice/legoesm/ice/bitz_lipscomb.py`; its former `_future` path is only
a compatibility import.  The old public ice-only API remains tested.  The
selected production dispatch is `SeaIceConfig.thermo_scheme="si3_bl99"` in
the existing `step_sea_ice`; the private `_si3_step_with_trace` hook exposes
the registered sub-call states to the fidelity instrument.

## Constructible identity and source map

`SI3ThermoConfig` has exactly one executable identity.  Validation rejects
any changed layer count, conductivity, salinity choice, drainage/flushing
choice, ponds, lateral melt, category count, dynamics/transport activation,
or non-NEMO constants set before a tendency can run.

| resolved choice | source |
|---|---|
| one category, HFN | `iceitd.F90:129-180`; accepted `ocean.output:615-627` |
| BL99, 3 ice + 3 snow layers | `icethd.F90:148-152`; `icethd_zdf_bl99.F90:34-590` |
| P07 conductivity | `icethd_zdf_bl99.F90:261-275` |
| option-2 salinity, drainage/flushing, `rn_sinew=.75` | `icethd_sal.F90:204-249`; `icethd_dh.F90:328-362` |
| vertical thickness change/open-water growth on; lateral melt off | `icethd.F90:154-183`; `icethd_dh.F90:91-533`; `icethd_do.F90:130-307` |
| ponds off; `ln_icedA=.false.` | `icethd.F90:161,176-177`; accepted `ocean.output:635-727` |
| aEVP, Prather, ridging/rafting selected but C1D-inert | accepted `ocean.output:820-846,861-862`; `ln_c1d` skips `ice_dyn` at `icestp.F90:167-171` |

The apparent `rn_sinew` inconsistency remains a **FINDING**, not a repair:
ORCA1 resolves `nn_icesal=2` with `rn_sinew=.75` even though its adjacent
comment says `.30` is required for option 2.  The card keeps `.75`.

## Immutable column card and inputs

Card: `C1D_OMIP_L3/EXP_SASICE`, `dt=3600 s`, documented `nsteps=8760`.
Accepted oracle root:
`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_phase2_inputs`.
This is a complete 8760-step CPU/no-MPI copy-run (`STOP 0`) built from the
copy-only source root
`/data/abyssal/dbalwada/nemo-testcases-l3/nemo502_si3thd_zdf_owner_src`.
Both that clone and the shipped read-only source resolve to NEMO commit
`dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`.  The copy was built with
`makenemo -n C1D_OMIP_L3 -d 'OCE SAS ICE' -m conda -j 8`; the only initial
build miss was the environment's Perl module path, after which the identical
command with the conda `PERL5LIB` completed.  The run invoked `./nemo.exe`
directly with `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, and no MPI
launcher.  `run.stderr` ends with `STOP 0`; `ocean.output` records restart and
diagnostic output at step 8760/date 20181231.
The card consumes the same official ERA5 member and the exact exchange/forcing
arguments dumped immediately after `ice_thd_1d2d` and immediately before
`ice_thd_zdf` (shipped `icethd.F90:140-148`; instrumented copy `:145-154`).  The older
exchange stream remains hash-checked and cursor-checked; its `qns_ice` value is
post-ZDF and is not silently re-labelled as an entry value.

| artifact | digest |
|---|---|
| official `C1D_v5.0.0.tar.gz` | MD5 `9456e6a0a84d40630ad1804fd4061caf`; SHA-256 `54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382` |
| `ERA5_NorthGreenland_surface_84N_-36E_1h_y2018.nc` | SHA-256 `e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe` |
| `nemo.exe` | SHA-256 `1bd1b979f6dda35d77e2a1254589296ee370681a18d99429135b17ef66890662` |
| resolved `namelist_cfg` | SHA-256 `6151c0fdd2431d07c7897d5852846a620edd58c55255f11b7fb3569803b5342c` |
| resolved `namelist_ice_cfg` | SHA-256 `da7b4fc5865edf6a6a912d6316a51f6b845aaf7e8b87e9da121c6f0a46278033` |
| `oracle_si3_thd_frames.bin` | SHA-256 `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b` |
| `oracle_si3_exchange_frames.bin` | SHA-256 `7f22ca914bc25890c7ccda810553eefae97625e437974c2208399e9afc5b2c50` |
| `oracle_si3_zdf_inputs.bin` (1,261,440 bytes) | SHA-256 `5522eadce595408b00065fa30d8b41fccb3815bee76d6fbf5ba3adbb2656cb27` |
| `oracle_si3_zdf_operands.bin` (2,300 bytes) | SHA-256 `587454974cb07590454d2bb63d745dc73488c220e84785175c9c110d26954632` |
| final `C1D_SASICE_00008760_restart_ice.nc` | SHA-256 `b61cb8443e14b3f0868ef125f621c0b1748aff7bc827dfab0d4044fc3f773b4e` |
| final `C1D_SASICE_00008760_restart.nc` | SHA-256 `84ed40c5e5d46f9830f4c203b3e3a79f6dfffaf142dc4c44347648e2cd9f5265` |
| `ocean.output` | SHA-256 `3e47d39f061ca1f9fa111a46b01bb3d1dbcfa10be73422fced0abc9e4af25430` |
| copy-only `MY_SRC/icethd.F90` | SHA-256 `cd8cb6efef7bb53d8974168850cffa1680c369f2be68dc87eaef259fc73f933a` |
| copy-only `MY_SRC/icethd_zdf_bl99.F90` | SHA-256 `2c479ad6849056d7a6c25b72a37b1c4d82cb154107677b371e6c86a97b0d5dd2` |
| `nemo_si3thd_phase2_gate.py` at `dbab665555` | SHA-256 `e8920c6723d89c677de754db23af565a01b4da5328e9d290a6cd7a20887eb624` |
| `bitz_lipscomb.py` at `dbab665555` | SHA-256 `69c9c529c449f0374330aa7576b060f02e8a4ef4d753819c7f2b1390bd6983d1` |
| committed-SHA gate JSON | SHA-256 `81ab7e66274456324bd4c8b68994fd8af57e1630522960144f6d4f2a7285df85` |

Archive provenance and URL are the Phase-1 pinned NEMO `sette_inputs` r5.0.0
listing.  No analytic or synthetic forcing is used.

The case does **not** read an `init_*` file: the accepted control print has
`nn_iceini_file=0`.  The actual resolved analytic initial values printed in
`ocean.output:747-759` and verified exactly by the gate are
`nlay_i=nlay_s=3`, `hti=2`, `hts=.2`, `ati=.9`, `tsu=270`, and `smi=6.3`.
The bridge uses `h_i=v_i/a_i`, `h_s=v_s/a_i`, `S_i=sv_i/v_i`, and converts
NEMO layer energy content to volumetric enthalpy using the category volume.

## Appendix-B constants recheck

NEMO values were re-read from `phycst.F90:39,57-66`,
`eosbn2.F90:1898-1899`, and the accepted run's actual `ocean.output:138-149,
615-727`.  They are selected through `IceConstantsConfig`; no global constant
is mutated.

| quantity | legoESM default | selected NEMO value |
|---|---:|---:|
| ice heat capacity [J kg-1 K-1] | 2106 | 2096.7 |
| fusion latent heat [J kg-1] | 333700 | 333360.1 |
| sublimation latent heat [J kg-1] | 2834000 | 2834400 |
| pure-ice conductivity [W m-1 K-1] | 2.04 | 2.034396 |
| snow conductivity [W m-1 K-1] | .31 | .5 |
| ice density [kg m-3] | 917 | 917 |
| snow density [kg m-3] | 330 | 330 |
| ocean reference density [kg m-3] | 1025 | 1026 |
| ocean heat capacity [J kg-1 K-1] | 3994 | 3991.86795711963 |
| liquidus slope [K PSU-1] | .054 | .054 |
| lead albedo | .06 | .066 |

## Rule-1d frame registry

| id | frame | time level | source |
|---:|---|---|---|
| 0 | ENTRY | now: global after salinity check, before frazil/thermodynamics | shipped `icethd.F90:109-115`; write at instrumented `:113-120` |
| 1 | POST_ZDF | now: selected-category 1-D | shipped call `icethd.F90:148`; write at instrumented `:154-155` |
| 2 | POST_DH | now: selected-category 1-D | shipped call `icethd.F90:150`; write at instrumented `:157-158` |
| 3 | POST_TEMP1 | now: selected-category 1-D | shipped call `icethd.F90:152`; write at instrumented `:160-161` |
| 4 | POST_SAL | now: selected-category 1-D | shipped call `icethd.F90:154`; write at instrumented `:163-164` |
| 5 | POST_TEMP2 | now: selected-category 1-D | shipped call `icethd.F90:156`; write at instrumented `:166-167` |
| 6 | POST_DO | now: global post-open-water growth, pre-correction | shipped call `icethd.F90:181`; write at instrumented `:192-193` |
| 7 | EXIT | now: global post-correction/LBC | shipped sequence `icethd.F90:183-216`; write at instrumented `:228` |

SI3 has no leapfrog temperature level inside this chain.  ENTRY is global;
stages 1-5 are selected-category one-dimensional arrays; POST_DO and EXIT are
global again.  The gate fails closed on magic, version, step, stage, shape,
layer counts, bit width, payload size, non-finite values, hashes, and dtype.

The entry-input stream is one frame per ice step with
`qns_ice,qsr_ice,dqns_ice,qtr_ice_top,t_bottom,sss,evaporation,
snow_precipitation,qprec_ice,qcn_ice_bottom,qsb_ice_bottom,fhld,qlead`, at the
selected-category 1-D time level after conversion and before ZDF
(shipped `icethd.F90:140-148`; instrumented copy `:145-154`).  The first ZDF
call has this operand registry:

| id | frame | time level | source |
|---:|---|---|---|
| 0 | INIT | selected-category 1-D, before Picard iteration | `icethd_zdf_bl99.F90:159-230` |
| 1 | ITER_P07_KAPPA | current Picard iterate | `icethd_zdf_bl99.F90:261-335` |
| 2 | ITER_CAP_FLUX | current Picard iterate | `icethd_zdf_bl99.F90:338-379` |
| 3 | ITER_MATRIX | current Picard iterate | `icethd_zdf_bl99.F90:393-514` |
| 4 | ITER_FORWARD | current Picard iterate | `icethd_zdf_bl99.F90:516-529` |
| 5 | ITER_SOLUTION | newly solved Picard iterate | `icethd_zdf_bl99.F90:531-558` |
| 6 | ITER_CONVERGENCE | newly solved iterate and convergence state | `icethd_zdf_bl99.F90:563-589` |

The reader requires INIT followed by all six frames for every iteration, in
order, and a false convergence flag until the final true flag.  A corrupted
magic-byte plant proves this registry fails closed.

## `kt=1` gate and first-divergence ownership

Metric: `max(abs(legoesm-oracle))/max(1,max(abs(oracle)))`; bar `1e-15`.
Exact geometry/count checks use equality.  All compared field rows are emitted
by `nemo_si3thd_phase2_gate.py`.  The post-fix summary through the stopping
row is:

| step/boundary | rows | over-bar fields | largest normalized error |
|---|---:|---|---:|
| geometry/IC | 7 | none | 0 |
| kt1 ENTRY | 12 | none | `1.7782396272275097e-16` (`e_i`) |
| kt1 POST_ZDF | 7 | none | `6.38200632183559e-16` (`e_s`) |
| kt1 EXIT | 12 | none | `7.11150934033674e-16` (`e_i`) |
| kt2 ENTRY | 12 | none | `1.777877335084185e-16` (`e_i`) |
| kt2 POST_ZDF | 7 | none | `6.22882295021345e-16` (`e_s`) |
| kt2 EXIT | 12 | none | `6.496138276185682e-16` (`e_s`) |
| kt3 ENTRY | 12 | none | `1.2992276552371363e-16` (`e_s`) |
| kt3 POST_ZDF | 7 | `e_i` | `2.559660701514903e-15` (`e_i`) |

### Rule-0 BL99 result and owner arm

The active NEMO arm has a maximum of 200 Picard iterations and a `1e-4 K`
criterion (`icethd_zdf_bl99.F90:76,87,233-243,583-589`).  Radiation is
transmitted/absorbed at `:150-157,205-230` and enters the RHS at
`:404,424,448`.  P07 conductivity uses current iterate temperatures, fixed
salinity, interface means, and bottom temperature at `:261-275`.  The surface
flux update and boundary row are `:367-379,438-448`; forward elimination and
back substitution/update are `:516-558`.  The executable legoESM counterpart
is `bitz_lipscomb.py:335-496` in commit `dbab665555`.

Both NEMO and legoESM take two iterations.  The registered temperature
solutions are all at bar but are **not bit-exact**: their largest errors are
`1.7053025658242404e-13 K` in iteration 1 and
`1.4210854715202004e-13 K` in iteration 2.  Iteration-1 `qns_ice` is bit-exact;
iteration-2 `qns_ice` is DEBT at normalized `2.0979870951387943e-14`.
Accordingly, this receipt does not say that the iteration is identical.

Scale was computed before ownership.  NEMO's exact `qns_ice` entry is
`-442.51394032071073 W m-2`.  The reviewed reconstruction differs by
`1.9853591766150203e-06 W m-2`, normalized
`4.48654606265316e-09`.  The private one-variable arm changes only that
entry scalar and produces the `458765.8` improvement stated in the verdict.
This **CONFIRMS** the entry-time bridge as owner of the reviewed kt1 debt.
NEMO updates `qns_ice` before the final surface solve
(`icethd_zdf_bl99.F90:372-379,553-558`), so a reconstruction using final
POST_ZDF `t_su` is not the input time level.  This is a harness time-level fix,
not a physical selector; there is no SI3 model switch to change and no
Frankenstein combination was introduced.

The private ZDF boundary hook also perturbs one input at a time by `1e-6` of
its dimensional scale at the new kt3 stopping state.  Those values are
sensitivity diagnostics only, not causal ownership of the new debt.  The
new `kt3.POST_ZDF.e_i` debt is left unowned for a separately preregistered
dispatch.

## Controls and tests

At committed implementation SHA `dbab665555`, the normal CPU/fp64 CLI writes
`phase2_gate_dbab665555.json` (SHA-256
`81ab7e66274456324bd4c8b68994fd8af57e1630522960144f6d4f2a7285df85`)
and exits 1 on the measured kt3 debt.  All three independent CLI plants also
exited 1:

- `--plant-geometry`: `GateError: geometry/IC gate`;
- `--plant-stage`: `GateError: ENTRY gate`;
- `--plant-selector`: `ValueError` naming the rejected ORCA1 identity.

The post-owner Phase-2 test file reports:

```text
============================= 11 passed in 15.52s ==============================
```

CPU/fp64 tests from the implementation round also had 92 passing tests across
`tests/test_param_specs.py`, the existing
parked-BL99 compatibility tests, the existing ice-column tests, and the new
Phase-2 tests.  The new tests directly exercise eager execution,
`jax.jit`, finite reverse-mode gradient, existing-column dispatch, exact card
hashes/geometry, first-divergence order, the operand-registry corruption plant,
and all three CLI red controls.  A further 41 configuration/complexity tests
passed in that round.  `git diff --check` is clean.

The original receipt's statement that all seven ratchet failures were
pre-existing was wrong: the hardcoded-constants ratchet correctly failed on
the lane-owned `constants_config.py` literal `273.15`.  The review correction
uses `constants.T_freeze`, `constants.rho_ocean_nemo`, and
`constants.L_fus_nemo`.  The exact per-file ratchet nodes for every changed
Python file, followed by the Phase-2 test file, report:

```text
============================= 13 passed in 18.20s ==============================
```

For the post-owner Python changes, the same exact-path ratchet selection
reports:

```text
====================== 5 passed, 3382 deselected in 0.62s ======================
```

The new flooding regression exercises an active-flooding column and proves
bit-for-bit equality between the former `_dh_step` expression and the shared
`snow_ice_flooding` function before the duplicate production expression is
removed.

## Review and tree integrity

An independent Claude review confirmed the dispatch, gate, planted controls,
first-divergence value, and non-vacuity, then placed the work on HOLD for the
three corrections recorded above.  Commit `0fa70bb55a` resolves those findings.
The exact-input/operand investigation is commits `15de2569af`, `3a348ab5c1`,
and `dbab665555`.

No file in `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2` or any shipped NEMO
configuration was modified or deleted.  No stale item was removed.  There is
nothing newly identified for `FLAGGED FOR FUTURE DELETION`; the `_future`
module remains as a compatibility shim so existing imports are not broken.

## End-of-task choice register

- ASKED — use only the ORCA1-resolved single-category HFN, BL99 3+3/P07,
  salinity option 2 (`rn_sinew=0.75`), no ponds, and no lateral melt identity.
- ASKED — reuse the existing ice model, parked BL99 work, shared Thomas solver,
  and shared snow-ice flooding implementation; do not construct a second model.
- ASKED — use the pinned ERA5/exchange inputs, one-hour step, CPU, and fp64.
- ASKED — use the `1e-15` DINO bar, required time-level registry, first-
  divergence stopping rule, one-variable private arms, and planted controls.
- ASKED — keep all shipped NEMO files read-only; use copy-only instrumentation.
- ASKED — correct the three review findings before continuing ZDF ownership.
- ASKED — determine actual BL99 iteration behavior, instrument operands only
  as needed, scale before ownership, and use a private one-variable arm.
- ASKED — fix within the SI3 identity when the source identifies a model-side
  owner; here the confirmed owner is instead the card's time-level bridge.
- ASKED — stop the post-fix sweep at the next over-bar frame and make no later
  trajectory claim.
- ASKED — move the accepted oracle root from `c1d_omip_l3_sasice_scope_gate2`
  to `c1d_omip_l3_sasice_phase2_inputs`; `namelist_cfg` and
  `namelist_ice_cfg` are byte-identical across the two roots (respectively
  SHA-256 `6151c0fdd2431d07c7897d5852846a620edd58c55255f11b7fb3569803b5342c`
  and `da7b4fc5865edf6a6a912d6316a51f6b845aaf7e8b87e9da121c6f0a46278033`).
- ASKED — continue in Phase 2b with NEMO-order ULP replays, all 8,760 steps,
  all 70,080 frames, and the measured legoESM fp32/fp64 phenomenology floor;
  see `nemo_testcases_l3thd_phase2b_receipt.md`.
- ASKED — do not push.
- UNASKED — none.
