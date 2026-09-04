# SI3 lane 3b round 13 preregistration — 10 m coupled slab

Date: 2026-09-06  
Tracker: `climate-federation/legoESM#1699`  
State: **PREREGISTERED — UNMEASURED**

This is the execution addendum to the independently reviewed rung-3.6 design
in `nemo_testcases_l3thd_rung36_preregister.md`.  All selectors, interfaces,
signs, time levels, coverage rows, and plants in that document remain fixed,
except for the user-selected construction parameter below.

## Decision 6: valid one-layer geometry

The copied oracle shall set `rn_bathy=10 m`.  This is a free construction
parameter selected by the user on 2026-09-06; it is neither copied from nor
attributed to ORCA1.  The latitude, longitude, `jpk=2`, exact ERA5 initial
T/S/U/V/SSH, initial ice, and ORCA1 `ln_ice_embd=.false.` stay unchanged.

SI3 first forms local snow-plus-ice mass at `iceistate.F90:400-401`.  In the
selected levitating branch, `:410-411` computes an **area-weighted global
mean** displacement, and `:424-427` subtracts that mean from the wet ocean
SSH levels.  The mean equals this case's local displacement only because the
domain contains one wet column.  This identity must not be generalized to a
multi-column domain.  With the already measured initial mass
`1710.0000000000002 kg m-2` and printed `rho0=1026 kg m-3`, the preregistered
prediction is `ssh=-1.6666666666666667 m` and
`e3t = 10 m + ssh = 8.333333333333334 m` (subject to the literal binary64
association executed by NEMO).  A non-positive `e3t` refutes the construction.

## Predictions and ordered stop

Before execution, this round predicts:

1. the scalar-math executable contains zero dynamic `_ZGV*` symbols;
2. the initial T/S/U/V bits and ice-mass/load identities remain those measured
   in Round 12, while only the configured bathymetry and dependent layer
   thickness change;
3. the 10 m wet layer remains positive for 8,760 hourly steps;
4. the NEMO-generated column CHLA equals the value read through `fld_read` and
   the sanctioned WEIGHTS file, bit for bit at the registered dump;
5. all WRITE-only headers pass the formula-based schema gate;
6. the first numerical comparison outside the pointwise `1e-15` bar, if any,
   is reported in NEMO execution order with its owner.  No later boundary will
   be tuned or claimed if that owner requires a scientific decision.

The existing exchange card will be extended only after a valid full oracle
exists.  The implementation must reuse `coupler/ocean_forcing.py`, explicitly
select fp64 plus `transcendentals="libm"`, and must not duplicate either the
ocean exchange path or Pierre's unmerged `lead_freeze_source` work.  If the
executing `icesbc` `zqld` path overlaps that work, this lane transcribes the
NEMO source into the existing path and records the merge overlap.

## Measurement contract

The oracle is a new config/run name and does not overwrite the retained 1 m
construction.  It is built with `arch-conda-scalarmath.fcm`, run directly on
CPU without `mpirun`, and continues through step 8,760 only if the ordered
construction checks remain green.  Every actually read immutable or generated
input is hashed.  Large streams and runtime outputs remain under `/data`;
only summaries and hashes enter git.

The full gate walks: initial geometry; PRE/POST `sbc_ssm` with `nn_fsbc=4`;
`eos_fzp`; `ice_update_flx`; each RK3 `tra_sbc_RK3`/SSH consumer;
`ice_update_tau` with `ln_drgice_imp`; and `sbc_fwb` with
`nn_fwb_voltype=1`.  Every row reports normalized error and `0 / n` differing
bits.  Each plant must change the targeted row and exit nonzero.

## ASKED / UNASKED

| choice | state | disposition before run |
|---|---|---|
| set the one-layer slab to `rn_bathy=10 m` | ASKED | fixed above as a free construction parameter |
| correct the obsolete 1 m design row | ASKED | corrected in the parent design |
| explain the levitating adjustment as global mean, local only for one column | ASKED | source-bounded above |
| rebuild/run 8,760 steps with scalar math and formula-derived headers | ASKED | pending |
| implement the reviewed exchange card after a valid oracle | ASKED | pending |
| treat 10 m as an ORCA1 quantity | UNASKED | forbidden; it is not |
| extrapolate the one-column equality to multiple columns | UNASKED | forbidden |
| change initial SSH/ice, generate another input, duplicate a shared routine, modify shipped NEMO, push, GPU, or `mpirun` | UNASKED | none authorized |

## Pre-run construction correction (recorded before retry B)

The first 10 m attempt exposed a geometry-composition error before the claimed
rung: although `rn_bathy=10`, copied C1D's L75 reference grid places its second
W level at about 1.024 m.  Its bottom partial-step assignment
`usrdef_zgr.F90:167-169` therefore evaluates
`MIN(rn_bathy,pdepw_1d(2))-pdepw_1d(1)` to about 1.024 m.  After the ice-load
adjustment that attempt still had negative `e3t` and was stopped/retained; it
does not implement Decision 6.

Retry B is preregistered to make the config-local `jpk=2` geometry actually be
one 10 m wet layer: before the shared `depth_to_e3` call, its W depths are
`[0, rn_bathy]` and its T depths `[rn_bathy/2, 3*rn_bathy/2]`.  NEMO's shared
`depth_to_e3_1d` definition (`src/OCE/DOM/depth_e3.F90:44-75`) then derives
both scale factors as `rn_bathy`, and the existing bottom partial-step line
retains that value.  This is a copied-configuration geometry correction that
executes the already asked 10 m slab; it changes no ocean or SI3 physics and
introduces no new free parameter.  Retry B must satisfy the original
`e3t≈8.333 m` prediction before any downstream row is scored.

## Pre-rescore branch correction (recorded before retry E)

The first complete 58-field `PRE_UPDATE_FLX` replay exposed a registry defect,
before any rung-3.6 result was accepted: the card transcribed the
`ln_cndflx=.true.` arm, but the resolved ORCA1 overlay explicitly has
`ln_cndflx=.false.` (`ORCA1-omip/EXPREF/namelist_ice_cfg:81`, confirmed in
`output.namelist.ice`).  The executed arms are therefore
`iceupdate.F90:109-113` for `qt_atm_oi` and `:132-134` for `zqsr`; the mass row
at `:178` also requires `emp_oce`, not the heat-of-freshwater operand
`qemp_oce`.  A new retained run will add `qsr_ice` and `emp_oce` to the
WRITE-only input frame and replay only that resolved branch.  Prediction: the
order-one first-step `qt_atm_oi`, `qt_oce_ai`, `qns`, and `emp` residuals will
disappear; any remaining first over-bar row will then be reported in NEMO
execution order without expanding the selected identity.
