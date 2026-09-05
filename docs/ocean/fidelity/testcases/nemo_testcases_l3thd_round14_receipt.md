# SI3 lane 3b round 14 receipt — one-layer ocean closure

Date: 2026-09-04  
Tracker: `climate-federation/legoESM#1699`  
Implementation commits: `5c38400c3fb0`, `3b39d8b0df57`  
Preregistration: `nemo_testcases_l3thd_round14_preregister.md`

## Outcome

**MEASURED PREFIX AT BAR; STOPPED AT THE FIRST CONTINUOUS DEBT.**  The active
TEOS-10 freezing-point row is now 2,190/2,190 bit-identical.  Exact NEMO entry
states put all six requested ocean boundaries at the pointwise `1e-15` bar for
all 8,760 steps.  Every registered step-1 boundary is bit-identical.  The
preregistered stronger all-year bit-identity prediction is refuted only by the
post-`stp_2D` barotropic velocities: `uu_b` is 8,301/8,760 bit-identical and
`vv_b` is 8,324/8,760, but their largest normalized differences are only
`2.117582368135751e-22` and `8.470329472543003e-22`.

The continuous driver therefore advanced one full legoESM step.  Its first
over-bar registered value is `CONTINUOUS.kt2_PRE_SSM.u`, absolute and
normalized error `1.0844585561836217e-4`.  The registered exact-entry prefix
ends at `kt=1 PRE_DYN_ZDF_SOLVE`; the next register is `kt=2 PRE_SSM`.
Consequently the measured owner interval is the shared implicit-ZDF solve / RK3
momentum reconciliation between those registers.  Pinning the individual
operand inside that interval requires a new oracle dump and a user decision,
so the ordered walk stops here.  No coupled-year or downstream-phenomenology
claim is made.

## Rule 0 and the single implementation

The pre-implementation search found the existing shared production owners:
`surface_stress_faces` and the NEMO bottom-drag helper in
`ocean_pe_latlon_cgrid.py`, `barotropic_substeps_latlon_cgrid`, the `rk3_ws`
program in `ocean_model_latlon_cgrid.py`, and the shared tridiagonal vertical
solver.  It found no separate slab solver and no executing top-drag consumer.
This round extended those owners.  It did not create a second coupler or ocean
step.  `build_c1d_omip_l3_slab_ocean_card` composes that shared path and binds
fp64 plus `transcendentals="libm"` explicitly
(`nemo_testcase_recipe.py:307-382`).  C1D's resolved `ln_dynadv_OFF` makes
horizontal momentum advection inert; the card binds the existing dynamics and
vertical selectors together rather than constructing an unexercised mixture.

For the active TEOS-10 surface freezing point, NEMO computes `z1_S0`, then
`zs`, the Horner polynomial, and finally the separate salinity multiplication
at `src/OCE/TRA/eosbn2.F90:1674-1682`.  The optional depth correction at
`:1685-1688` is not called by `icestp`; the EOS-80 statement at `:1691-1701` is
an inactive alternative.  legoESM now preserves those assignment boundaries
with the shared `nemo_source_round` at `ocean/eos.py:2983-3006`.  This moved
`POST_FZP.t_bo` from 2,189/2,190 to 2,190/2,190 bit-identical.  The pre-fix
implementation is the one-variable no-polynomial-round arm and reproduces the
sole old non-bit row.

The executing ocean statements and their shared legoESM owners are:

| boundary | NEMO 5.0.2 source | legoESM owner |
|---|---|---|
| `POST_SBC_STAGGER` | `src/OCE/SBC/sbcmod.F90:539-547` | `nemo_stagger_surface_stress`, `ocean_pe_latlon_cgrid.py:3903-3939` |
| `POST_ZDF_DRG_COEFF` | `src/OCE/ZDF/zdfdrg.F90:116-129` | `nemo_top_drag_rate_faces`, `ocean_pe_latlon_cgrid.py:3638-3662` |
| `PRE_DYN_SPG_TS` and `POST_STP2D` | `src/OCE/stp2d.F90:112-202,243-281` | shared RK3/stp composition in `ocean_model_latlon_cgrid.py` and `ocean_pe_latlon_cgrid.py` |
| `SSH_SUBSTEP` | `src/OCE/DYN/dynspg_ts.F90:623-630,734-760,861-895` | source-literal SSH and flux-form helpers, `barotropic_latlon_cgrid.py:1106-1140` |
| `PRE_DYN_ZDF_SOLVE` | `src/OCE/DYN/dynzdf.F90:293-345,466-519` | shared implicit surface-stress/top-drag assembly and shared vertical solver |

`dynzdf.F90:298-304,472-478` adds the signed top-drag coefficient to the
matrix diagonal, `:326-345,500-519` adds surface stress and solves it.  The
new shared consumer does not merge top and bottom drag merely because this
one-wet-layer case places them at the same vertical index.

## Slab construction and oracle provenance

User Decision 6 selects `rn_bathy=10 m` as a **free construction parameter**,
not an ORCA1 value.  With the initial ice load, the wet layer is approximately
8.33 m.  In levitating-ice mode NEMO first forms snow-plus-ice mass
(`src/ICE/iceistate.F90:400-401`), computes the area-weighted global mean SSH
adjustment at `:408-411`, and subtracts it at `:424-427`.  That global mean
equals the local adjustment only because this domain has one column.  This is
not a general identity and is not stated as one.

The final oracle is the retained scalar-math config copy
`C1D_OMIP_L3_COUPLED10M_R14F_SM`.  It was run twice for 8,760 CPU steps without
`mpirun`:

* `/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_coupled10m_r14f_oracle_a`
* `/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_coupled10m_r14f_oracle_b`

Both runs ended `STOP 0`; `nm -D` reports zero `_ZGV*` symbols.  Executable
SHA-256 is
`d26561c27b7d2f8fab1270097027ed21599485386fdf10228c0ae6175886797c`.
The scalar-math arch file SHA-256 is
`132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561`;
its relevant added flag is `-fno-tree-vectorize`.  The sorted config-local
`MY_SRC` manifest SHA-256 is
`411f46a6c44e2d76b78ddf9de7739b81d4e63606b31fbd1d078b0d50a8047bd1`.

### Header-validity correction and reproducibility

Every new writer derives its count with `SIZE` from its actual write list, and
the reader validates magic, count, precision, record alignment, and first/last
record headers.  The 2.75 GB split-explicit stream stays under `/data`; no
runtime stream is committed.

R14D's QSR writer used a full-grid dummy declaration for the compact allocated
`itab(T2D(0),jpk-1)` object in `traqsr.F90:293,303`.  R14F corrects that
config-local WRITE-only declaration.  The R14D and R14F physical QSR fields,
registry, every other stream, `ocean.output` numerics, and both restarts are
byte-identical.  Only the invalid diagnostic `itab` read differed (8,753 of
8,760 records; first R14D value 0 versus R14F value 37).  R14F A/B QSR is
stable at
`e83dfe2e8317961eeca188cb978a9c293f1cd4e40c77652da79c0b2f512a9e84`.
R14D remains retained and is not an accepted oracle.

Key R14F A/B-stable SHA-256 values:

| artifact | SHA-256 |
|---|---|
| exchange frames | `b97b85ef8381a06e6ac548084337684875a8a634486415075ab35d239945188a` |
| stagger frames | `3e8bc8a57b0a0e43b00478dd5d9ef5443c723f28d4f0802c2e19df3ff5df93af` |
| drag frames | `ee19dbb3cf6fa85ec5788fb350a547c6cb167c3da87445a5887c39eb1a67d7bb` |
| split-explicit frames | `ccde476a5d7bb51c7a8ae5edd093348fc1bf85d9681b545b5a64b989906777e1` |
| split-explicit statement frames | `331b5370364c5e3fb3a81b11ed8e235e24ab137bb2a2e97df86b85f38fbeb586` |
| `stp_2D` frames | `dd99596a588f30c564ba7e53cfdab86760a5dbd8d64cfdcdf310c3ab39ca64cf` |
| pre-ZDF frames | `dd0fb1dc28b67b31d04615291b8d754409096ea8a3c4c6c06064e224052862e1` |
| QSR frames | `e83dfe2e8317961eeca188cb978a9c293f1cd4e40c77652da79c0b2f512a9e84` |
| FWB frames | `15f822437c82b8d3ffde71e6706e22bc4ea12dbff8e9bdfa36bd803b704e39d4` |
| thermodynamics frames | `8e6ea3881d2f08b646f3d0e407749b781d5fa8c50f36821f0a480983d355fb54` |
| `ocean.output` | `53bc133cb27f3391cf8813b5b39ec5534ab258b3e02a7d0f878e480d8fb27018` |
| ocean restart | `45aecc2778fdf786d082363fcc869d63a99f810a801939c7c47903ce72b7b16c` |
| ice restart | `c756f3e2117ad84ba73fa17a8b00470aba1b6824f504f86f8159a23777288b79` |

Pinned inputs are unchanged: ERA5
`e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe`,
initial state
`0ded4378da5d454d1658a889190428cd19404e2e9fd8f27a6781d1c635e838fa`,
and NEMO-WEIGHTS column weights
`715c51c4528feb7cb1f1d409584e5257d1b680f350682e2444b478217ec3b4dc`.
The resolved ocean and ice namelists hash to
`12abc86c56a1d5222ede1a8213ccbae1b783b0c7f512e4fa795e00502c726132`
and `d4da9b215b1d762b7a776324ee7ce2ea6a2c0ee18452b434bc9cec65cc8fbce6`.

## Ordered measurements

Gate artifact:
`round14_ocean_gate.json`, SHA-256
`95823608f0bce89fbbd711c113f36c55bc6d08ffc5673a648039f3f9fbce7628`.

| scored field group | bit-identical | maximum normalized error | verdict |
|---|---:|---:|---|
| staggered `utauU`, `vtauV` | 17,520 / 17,520 | 0 | AT-BAR |
| top drag plus u/v face coefficients | 43,800 / 43,800 | 0 | AT-BAR |
| `PRE_DYN_SPG_TS` SSH/u/v RHS | 3 / 3 | 0 | AT-BAR |
| SSH-substep statement and post-LBC u/v | 3,784 / 3,784 | 0 | AT-BAR |
| post-`stp_2D` SSH and advected u/v | 26,280 / 26,280 | 0 | AT-BAR |
| post-`stp_2D` `uu_b` | 8,301 / 8,760 | `2.117582368135751e-22` | AT-BAR, non-bit |
| post-`stp_2D` `vv_b` | 8,324 / 8,760 | `8.470329472543003e-22` | AT-BAR, non-bit |
| pre-ZDF diagonal and RHS u/v | 35,040 / 35,040 | 0 | AT-BAR |

The table uses the gate's pointwise normalization
`abs(model-oracle)/max(abs(oracle),1)`.  “AT-BAR” is not relabeled “matched” or
“faithful.”

### Continuous stop

`round14_ocean_trajectory.json` SHA-256 is
`81dabc4e10871ed461bdf8244dc9c9921614b7bcdeb884582d244d3684caaf50`.

| `kt=2 PRE_SSM` field after one legoESM step | absolute error | normalized error | status |
|---|---:|---:|---|
| u | `1.0844585561836217e-4` | `1.0844585561836217e-4` | DEBT; first |
| v | `1.5809605695616855e-6` | `1.5809605695616855e-6` | DEBT |
| temperature | `2.267231727692831e-5` | `1.308615514721144e-5` | DEBT |
| salinity | `7.624427225110253e-4` | `2.2424454879980685e-5` | DEBT |
| SSH | `4.440892098500626e-16` | `2.6642450748120423e-16` | AT-BAR |
| e3t | 0 | 0 | AT-BAR, bit-identical |

This is a first-register interval attribution, not yet a first-statement
attribution.  The requested `kt=2..8760` growth sweep is withheld by the
ordered stop rule.

## Scaling and planted controls

The one-variable top-drag arm scales only `rCdU_top`.  Factor 1 is exact.
Factors 0, 0.5, and 2 make the `rCdU_top` row red with maximum absolute changes
`9.77125699490778e-5`, `4.88562849745389e-5`, and
`9.77125699490778e-5`.  At `kt=1`, however, the oracle top coefficient is zero;
top drag therefore cannot own the continuous first split.

Each row-level plant returned nonzero and named its red target: FZP,
stress-stagger, top drag, pre-SPG RHS, SSH substep, post-`stp_2D`, pre-ZDF, and
continuous trajectory.  Their retained JSON SHA-256 values are listed by
`sha256sum round14_plant_*.json` in the oracle root; no control compares a file
with itself or perturbs a structural zero.

## Tests and repository hygiene

Focused Round-14 tests: **13 passed**.  Neighboring ocean/rung-3.6 tests:
**61 passed**.  Each of the seven touched production files passes
`tests/test_no_hardcoded_constants.py` individually.  The full ratchet reports
3,391 passed, 2 skipped, and five pre-existing failures in FV3/DINO files not
touched by this round; no clean-suite claim is made.  `git diff --check` is
clean.  The gate and trajectory stay hash-bound under `/data`; no multi-MB
runtime artifact is in Git.

## Pierre overlap

Pierre's unmerged `origin/fix/omip-gm-treguier-one-variable` changes
`lead_freeze_source`.  Its `icesbc` `zqld` identity overlaps the already shared
`_nemo_si3_ice_flx_other` path from the ice-side rung, not the new ocean
operators.  This round added no second lead-heat implementation.  The overlap
must be reconciled when that branch merges.

## ASKED / UNASKED

| choice | status | disposition |
|---|---|---|
| source-literal TEOS-10 FZP | ASKED | 2,190/2,190 bit-identical |
| 10 m slab bathymetry | ASKED | recorded as a free construction parameter |
| shared ocean-path extension | ASKED | implemented without a slab solver |
| six-boundary exact-entry year gate | ASKED | all rows at bar; bit counts disclosed |
| continue after bit-exact step 1 | ASKED | advanced one step; stopped at first over-bar register |
| exact owner operand dump inside the post-ZDF/RK3 interval | UNASKED | requires a new oracle instrument and decision |
| modify/delete shipped or retained NEMO files | UNASKED | not done |
| push, GPU, `mpirun`, or coupled-year claim beyond the stop | UNASKED | not done |

## CONFIRMED / PLAUSIBLE / DEBT

**CONFIRMED:** the FZP bit closure; the R14F header correction; A/B oracle
reproducibility; the all-year exact-entry bar result; all step-1 registered
boundaries bit-identical; and every planted control red.

**DEBT:** the first continuous split at `kt2_PRE_SSM.u` and its companion
velocity/tracer rows.  The owner interval is after the registered pre-ZDF
operands and before the next `PRE_SSM` entry.

**PLAUSIBLE, not claimed:** the precise statement owner lies in the shared
one-layer implicit-ZDF solve or the following RK3 mean reconciliation.  An
operand-level oracle dump is needed to discriminate them.

## FLAGGED FOR FUTURE DELETION (nothing deleted)

The incomplete `C1D_OMIP_L3_COUPLED10M_R14E_SM` build copy and superseded R14D
diagnostic run are stale.  R14D is useful evidence for the QSR header ticket.
Both remain retained; no shipped NEMO configuration or input was modified or
deleted.
