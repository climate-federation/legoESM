# Receipt — SMT-RUNGS round 4: SMT-6 (BBL + geothermal) survey and deck proposal

**Status: STOPPED_FOR_DECISION (survey only, no model code, no acquisition).**
Two findings change what SMT-6 can prove: (1) on this deck the NEMO bottom
boundary layer gate is closed on every sloped face at every one of the 100
saved days, so `ln_trabbl` with ORCA2's values moves no tracer; (2) legoESM has
no geothermal heating inside the lat-lon C-grid model at all. Base `51b6144c3`;
evidence `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/smtrungs_rounds/round4/`
(`survey.json` sha256 `2af40465…`, `survey_plant.json` `15be7f7d…`).

**ORCA2 pointer.** ORCA2 rung 2 = rung 1 + exactly these two switches (a `diff`
of the two rung `namelist_cfg` files shows only `ln_trabbc` and `ln_trabbl`
changed). Everything in sections 1-2 below is what the ORCA2 rung-2 walk has to
reproduce, with ORCA2's own values and a TEOS-10 equation of state.

## Round 4 — 1. NEMO's program (compiled SMT-5 build `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/`)

Placement in the RK3 stage ladder, all in the tracer right-hand side of
stage 3 only:

| step | statement | where |
|---|---|---|
| BBL coefficients (masks, upslope velocity) computed once per step, after the advective transports, before the tracer RHS | `CALL bbl( kstp, nit000, Kbb, Kmm )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:436` (the Shuman branch repeats it at `:458`) |
| lateral mixing, then geothermal, then BBL, then damping, then vertical mixing | `tra_ldf` `:526`, `tra_bbc` `:527`, `tra_bbl` `:528`, `tra_dmp` `:529`, `tra_zdf` `:538` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:527` and `:528` |

**BBL (trabbl).** Operands, in order:

| item | statement | citation |
|---|---|---|
| bottom T/S used by the gate: BEFORE level (`Kbb`) at the bottom T-level `mbkt` | `zts(ji,jj,jp_tem) = ts(ji,jj,ik,jp_tem,Kbb)` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:409` |
| depth passed to the expansion coefficients: bottom T-level reference depth times the stretch `(1+r3t(Kmm))` | `zdep = gdept_1d(ik)*(1+r3t(Kmm))` | `:412` |
| alpha/beta at each bottom cell | `CALL eos_rab( zts, zdep, zab, Kmm, kbnd=1 )` | `:417` |
| density-difference proxy at the U face, mask `ssumask` | `zgdrho = za*(T(i+1)-T(i)) - zb*(S(i+1)-S(i))` | `:427-428` |
| gate: `SIGN(0.5, -zgdrho*mgrhu)`; open only if grad(rho).grad(H) < 0; a zero argument gives `+0.5` = closed | `zsign = SIGN( 0.5_wp, -zgdrho * REAL( mgrhu ) )` | `:430` |
| masked coefficient | `ahu_bbl = (0.5 - zsign) * ahu_bbl_0` | `:431` |
| static slope sign from bottom-LEVEL depth `gdept_1d(mbkt)` (partial cells in the same level give 0) | `IF( gdept_1d(mbkt(ji+1,jj)) - gdept_1d(mbkt(ji,jj)) /= 0._wp )` | `:584-585` |
| BBL thickness = min of the two faces' `e3u_3d` at the two bottom levels | `e3u_bbl_0 = MIN( e3u_3d(.., mbkt(ji+1,jj)), e3u_3d(.., mbkt(ji,jj)) )` | `:594` |
| static coefficient | `ahu_bbl_0 = rn_ahtbbl * e2_e1u * e3u_bbl_0 * ssumask` | `:600` |

Diffusive trend (`nn_bbl_ldf = 1`, ORCA2 rung 2): called from `tra_bbl` with the
**before** tracers and the **now** thickness:

- `CALL tra_bbl_dif( pts(:,:,:,:,Kbb), pts(:,:,:,:,Krhs), jpts, Kmm )` at
  `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:184`.
- bottom before value `zptb = pt(ji,jj,ik,jn)` at `:253`.
- the trend, added to the bottom cell's RHS only, with the U pair and V pair
  grouped separately, times `r1_e1e2t`, divided by
  `e3t_3d(bottom)*(1+r3t(Kmm)*tmask)`: `:258-263`.

Advective BBL (`nn_bbl_adv /= 0`, `:193`) is OFF in ORCA2 rung 2 (`nn_bbl_adv = 0`),
so its transports (`:464` upper-velocity form, `:480` and `:498` density-driven form
with `zgbbl = grav*rn_gambbl`) are not a rung-2 deliverable; `rn_gambbl = 10` is
read and printed but multiplies nothing.

**Geothermal (trabbc).**

| item | statement | citation |
|---|---|---|
| heating of the bottom wet cell only, added to the temperature RHS, per unit thickness `e3t_3d(mbkt)*(1+r3t(Kmm)*tmask)`; no tracer-before term | `pts(..,mbkt,jp_tem,Krhs) = pts + qgh_trd0 / (e3t_3d(..)*(1+r3t*tmask))` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbc.f90:158-159` |
| `nn_geoflx = 1`: constant flux, used **as W/m2** (no unit factor) | `qgh_trd0 = r1_rho0_rcp * rn_geoflx_cst` | `:226` |
| `nn_geoflx = 2`: file read once at `nit000` (a one-record climatology, no time interpolation) | `CALL fld_read( nit000, 1, sf_qgh )` | `:242` |
| and converted from mW/m2 | `qgh_trd0 = r1_rho0_rcp * sf_qgh(1)%fnow(:,:,1) * 1.e-3` | `:243` |

Constants: the run printed `rho0 = 1026`, `rcp = 3991.86795711963`, `rho0*rcp = 4095656.524`
(SMT-5 `ocean.output`, phycst block). The shipped
`namelist_ref` says `rn_geoflx_cst = 86.4e-3` with a `[mW/m2]` label
(`VORTEX_SMT5_VEC_R8_OMIP_L1/EXP00/namelist_ref:848`), but trabbc statement 226
applies no 1e-3, so the constant is **86.4 mW/m2 expressed in W/m2**; the
label is wrong and the number is right for the file's own mean (below).

**S-EOS expansion coefficients used by the gate on this deck** (`ln_seos`,
`rn_a0 = 0.28`, `rn_b0 = 0`, `nu = lambda = mu = 0`): alpha = `rn_a0*r1_rho0`,
beta = 0 (`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:1344` and `:1347`).
With beta = 0 and alpha > 0 the gate reduces to a sign test on the bottom-level
temperature difference alone.

**What ORCA2 rung 2 resolves them to** (`orca2_hierarchy/rung2/deck/namelist_cfg`
and the record's `ocean.output`):

| setting | ORCA2 rung 2 | citation |
|---|---|---|
| `ln_trabbc` | `.true.` | `orca2_rung2/namelist_cfg:273` |
| `nn_geoflx` | `2` (read) | `orca2_rung2/namelist_cfg:274` |
| `sn_qgh` | `geothermal_heating.nc`, variable `heatflow`, `clim=.true.`, `'yearly'`, no time interp | `orca2_rung2/namelist_cfg:280` |
| `rn_geoflx_cst` | not set; ref value 86.4e-3 printed, unused | `orca2_rung2/ocean.output:685` |
| `ln_trabbl` | `.true.` | `orca2_rung2/namelist_cfg:285` |
| `nn_bbl_ldf` / `nn_bbl_adv` | `1` / `0` | `orca2_rung2/namelist_cfg:286` and `:287` |
| `rn_ahtbbl` / `rn_gambbl` | `1000. m2/s` / `10. s` | `orca2_rung2/namelist_cfg:288` and `:289` |

Geothermal input statistics (`heatflow`, mW/m2, 148x180 grid, sha256 of the
file `e369762f…`, over the 16433 wet columns of the rung-2 `mesh_mask`;
instrument: `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_smtrungs_round4_smt6_survey.py`):

| min | max | mean | median | area-weighted mean | zero cells |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 400.0 | 84.49 | 59.93 | **86.28** | 5 |

The area-weighted mean is 0.14% below the constant 86.4: ORCA2's file mean and
NEMO's default constant are the same number.

## Round 4 — 2. legoESM's existing code (searched: `grep -rn "geothermal\|bbl_" packages/ocean src scripts`)

| item | NEMO statement | legoESM | verdict |
|---|---|---|---|
| diffusive BBL operator and gate | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:258-263`, `:427-431`, `:584-600` | `bbl_adv.py:151` (`nemo_bbl_diffusive_geometry`), `bbl_adv.py:220` (`nemo_bbl_diffusive_coefficients`), `bbl_adv.py:274` (`apply_bbl_diffusive_tendency`): source-ordered transcription incl. the doubled alpha/beta and Fortran `SIGN` zero | MATCH by reading; ORCA2 has a cellwise gate script for it (`nemo_testcase_l4_orca2_diffusive_bbl_gate.py`) |
| placement and time level | stage 3 RHS, Kbb tracers, Kmm thickness | `ocean_model_latlon_cgrid.py:2282` hook at stage index 2, uses the Kbb base tracers, folds in as `ocean_model_latlon_cgrid.py:2336` | MATCH by reading (UNVERIFIED by a stage-state replay: the stage-3 thickness is taken to be `e3t_0*(1+r3t(Kmm))`) |
| **alpha/beta for the gate under S-EOS** | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:1344`, `:1347` | `bbl_adv.py:243` calls `nemo_roquet_alpha_beta`, which accepts only the TEOS-10/EOS-80 coefficient sets and raises otherwise (`eos.py:1501-1507`); the card's `eos` string is `nemo_seos` and is passed straight through (`ocean_model_latlon_cgrid.py:2325-2326`) | **GAP**: SMT-6 cannot be built on the existing card until an S-EOS alpha/beta branch exists (a new, cited, card-selected statement) |
| geothermal inside the model | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbc.f90:158-159` (stage-3 RHS before `tra_zdf`, `Kmm` thickness) | none: `ocean_model_latlon_cgrid.py` contains no geothermal term; the only users are the standalone `geothermal_apply.py:86` (explicit-Euler `T <- T + dt*dTdt` after the step) called from the OMIP driver | **GAP**: operator-split post-step, not an RK3 RHS term; no card field |
| geothermal denominator and constants | `r1_rho0_rcp = 1/(1026*3991.86795711963)`; denominator `e3t_0*(1+r3t)` | `geothermal.py:172` divides by `rho_0*c_sw*h_safe` with defaults `constants.rho_ocean = 1025` and `c_sw = 3994` (`geothermal_apply.py:67-68`) | **GAP**: rho*cp differs by 0.044% (4093850 vs 4095656.5) unless the NEMO preset is passed; `h_safe` is the live thickness, equal only if it carries `(1+r3t)` |
| file input and units | trabbc line 243 reads mW/m2 and multiplies by 1e-3 | `geothermal.py` takes W/m2 from the caller; no `heatflow` reader, no `clim/yearly` one-record read | **GAP** (only if the file route is chosen) |
| default off | ref `ln_trabbc = .false.`, `ln_trabbl = .false.` | `GeothermalConfig.enabled = False`; `bbl_adv_option`/`bbl_diffusive_option` default 0 | MATCH (a card must select both explicitly) |

Callers of `bbl_diffusive_option = 1` in a certified card: the ORCA2 zps card
only (`nemo_testcase_recipe.py:1708`); OVERFLOW certifies the *advective* arm
(`nemo_testcase_recipe.py:1071`). No lat-lon seamount card selects either.

## Round 4 — 3. What this deck can and cannot exercise (measured)

- **Slopes.** The seamount's bottom levels are `mbkt` in {8, 9, 10}; with the
  level depths of `gdept_1d` this gives exactly **24 U faces and 24 V faces**
  (of 3660 wet faces each) with a non-zero `mgrh`. Everywhere else the gate is
  exactly closed by construction (`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:584-585`): partial cells inside one level
  produce no slope. 172 distinct bottom `e3t_0` values sit on 3721 wet columns
  (50.67 to 500 m).
- **The gate is closed on all of them, at every saved day.** Initial state
  (record 1 of the dumped target): bottom T runs 4.02 to 7.39 degC with the
  colder (denser) side always the deeper one, so all 48 sloped faces are on the
  stable side of the sign test. Over the 100 daily restarts of the SMT-5 NEMO
  record the open-face count is **0 U, 0 V in every one**; the largest
  same-sign margin stays near -1.63 K (`survey.json`, `max_open_u/v = 0`).
  Planted control: making one shelf-side bottom cell 1 K colder than its deep
  neighbour in the first restart gives 1 open face and the script exits 1;
  unplanted it exits 0; five unit tests cover the sign and zero convention.
  Scope: restarts are every 30 steps (every 24 h model time), not every step;
  a transient inversion between saves would not be seen (PLAUSIBLE that none
  exists; not measured).
- **Consequence.** With ORCA2 rung 2's values (`nn_bbl_ldf = 1`,
  `nn_bbl_adv = 0`) the BBL term is exactly zero in NEMO on this deck, so a legoESM match
  proves the *gate* (sign, zero convention, slope mask) but none of the flux
  statement at trabbl lines 258-263. The advective BBL would be closed too (`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:464` and
  `:498` carry the same sign tests).
- **Geothermal is live, small, and masked by the SMT-5 damping.** Per-step
  heating of a bottom cell at `rn_Dt = 2880 s`: 1.2e-7 K (500 m cell) to 1.2e-6
  K (50.67 m cell), 1e8 times the double-precision spacing near 4 degC, so the
  step-1 increment test (the same instrument as round 3's damping increment) can
  see it. The damped equilibrium is only `Q/(rho0 cp e3t) / resto`: 3.6e-6 K
  (500 m) to 3.6e-5 K (50.67 m), the same size as the 1.08e-5 K day-100 rms of
  SMT-5, so a day-100 score cannot attribute it.

## Round 4 — 4. SMT-6 deck proposal

SMT-6 = SMT-5 deck + two explicit namelist blocks (the SMT-5 `namelist_cfg`
has neither block; both switches default `.false.` in the namelist_ref, so
today they are inherited, not chosen — RULE: name them):

```
&nambbc
   ln_trabbc     = .true.
   nn_geoflx     = 1          ! DECISION (a)
   rn_geoflx_cst = 86.4e-3    ! W/m2 as used by trabbc.f90:226 (ref label "mW/m2" is wrong)
/
&nambbl
   ln_trabbl  = .true.
   nn_bbl_ldf = 1             ! ORCA2 rung 2
   nn_bbl_adv = 0             ! ORCA2 rung 2
   rn_ahtbbl  = 1000.         ! ORCA2 rung 2, m2/s
   rn_gambbl  = 10.           ! ORCA2 rung 2, s (multiplies nothing at nn_bbl_adv = 0)
/
```

Everything else is the SMT-5 deck (damping, drag, mixing, timestep 2880 s,
EOS, forcing).

**Decision (a), geothermal flux.**

| option | deck | new input | comment |
|---|---|---|---|
| **A (pick)** | `nn_geoflx = 1`, `rn_geoflx_cst = 86.4e-3` | none | ORCA2's area-weighted file mean is 86.28 mW/m2; certifies the heating statement (trabbc lines 158-159) and the constants, no file |
| B | `nn_geoflx = 2`, uniform `heatflow = 86.28` mW/m2 file | one generated file | also exercises trabbc lines 242-243 (read + `1e-3`); adds a file whose value is our choice |
| C | `nn_geoflx = 2`, ORCA2's pattern mapped to the seamount | one mapped file | the mapping is arbitrary (no analogue of ORCA2 ridges on this grid) |

**Decision (b), the closed BBL gate.**

| option | deck | what SMT-6 then proves |
|---|---|---|
| **I (pick)** | ORCA2 values only, as above | gate sign / zero / slope mask; not the flux statement (that stays with the ORCA2 cellwise gate) |
| II | I plus a second arm whose initial bottom T is made colder (denser) on the seamount flank than the adjacent deep cell, so the gate opens | also the diffusive flux statement (trabbl lines 258-263); changes the initial state (an unasked physical choice: magnitude, extent) — a separate rung, not a mix |

Pick (b) = I now, II afterwards as a separate arm, because II alters the model
state and D93 asks for one variable per rung.

**Legoesm work SMT-6 needs before an acquisition is useful** (not done this
round; none exists): (1) an S-EOS branch of the BBL expansion coefficients
(eosbn2 S-EOS rab statements, lines 1344 and 1347), card-selected; (2) a card field and an RK3
stage-3-RHS geothermal arm with NEMO's constants (`r1_rho0_rcp`, `e3t_0*(1+r3t)`)
instead of the post-step helper; (3) card selection of `bbl_diffusive_option=1`
and `bbl_aht_m2_s=1000` on the seamount card; (4) an acquisition `run.sh`
(SMT-5 pattern) with `ln_trabbc`/`ln_trabbl` blocks.

## Choices made this round (end-of-task list)

- Survey scope = the four items asked, read-only: ASKED.
- Evidence = 100 daily restarts at 30-step spacing; geothermal stats over wet
  columns of the ORCA2 rung-2 mesh mask, area-weighted with `e1t*e2t`: UNASKED
  (measurement convention; revert = restate with another weighting).
- Added one probe script + one test (the instrument is committed, not heredoc):
  UNASKED, justified by the repo's "commit the probe" rule.
- No deck, card, default or model code changed: ASKED.

## Review and gates

- Citation gate (from heading "## Round 4 —"): see the commit message and
  `round4/citation_gate.json`.
- New tests: `tests/ocean/fidelity/test_nemo_testcase_l1_vortex_smtrungs_round4_smt6_survey.py`,
  5 passed (run with the repo `packages/*` and `src` on `PYTHONPATH`; the venv's
  default `legoesm` resolves to a different checkout and fails collection).
  The two round-3 tests `test_smt5_inputs_are_nemos_own_and_mesh_equals_smt4`
  and `test_planted_resto_cell_is_refused_by_the_card_the_gate_builds` fail in
  this environment with a card-build `ValueError` (transcribed e3t vs
  h_partial); no code was changed this round, so this is not caused by it
  (UNVERIFIED which environment difference triggers it).
- Review: pending (single review (codex) of the diff; result in the commit).

UNVERIFIED: legoESM-side placement/time-level rows are read from source, not
replayed; the stage-3 thickness equality; that no transient gate opening exists
between daily restarts; the 0.044% rho*cp figure uses `constants.rho_ocean` and
`c_sw` as printed in `src/legoesm/constants.py`/`ocean/constants_config.py`.
