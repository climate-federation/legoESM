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
