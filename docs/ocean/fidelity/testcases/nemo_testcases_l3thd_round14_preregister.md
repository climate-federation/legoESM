# SI3 lane 3b round 14 preregistration — one-layer ocean closure

Date: 2026-09-04  
Tracker: `climate-federation/legoESM#1699`  
State: **PREREGISTERED; UNMEASURED**

This round extends the already measured rung-3.6 prefix.  It does not alter or
re-score the accepted ice-side result except for the single freezing-point row.
The retained round-13 oracle and all shipped NEMO files remain immutable.

## Pre-implementation search and source identity

The search found one existing production path, not a slab implementation:
`nemo_eos_fzp` in `packages/ocean/legoesm/ocean/eos.py`,
`surface_stress_faces` and `nemo_bottom_drag_rate_faces` in
`dynamics/ocean_pe_latlon_cgrid.py`, the shared
`barotropic_substeps_latlon_cgrid`, the `rk3_ws` program in
`dynamics/ocean_model_latlon_cgrid.py`, and the shared implicit tridiagonal
solver in `physics/vertical_mixing/implicit_solver.py`.  The coupler already
stores signed `rCdU_top` in `OceanSurfaceForcing`, but no executing production
ocean consumer of that field was found.  Any required top-drag support will be
added to those shared owners; this round will add no slab-specific solver or
alternate ocean step.

The active oracle resolves TEOS-10 (`output.namelist.dyn:43-46`).  Its scalar
surface call is `icestp.F90:136-138`, which invokes the two-dimensional
`eos_fzp` arm.  NEMO forms `z1_S0` and `zs` at
`eosbn2.F90:1674-1678`, evaluates the Horner polynomial in one source
statement at `:1679-1681`, and multiplies by salinity in the next statement at
`:1682`; the optional pressure statement at `:1685-1688` does not execute.
The EOS-80 alternative is present at `:1691-1701` but is inactive.  legoESM
will preserve these source-statement boundaries with the shared
`nemo_source_round`; it will not add a selector or alter the polynomial.

The remaining oracle order is fixed by NEMO: T-to-face stress staggering at
`sbcmod.F90:533-547`; `zdf_drg('TOP')` copies `rCdU_ice` to `rCdU_top` at
`zdfdrg.F90:116-129`; `stp_2D` assembles the complete momentum and freshwater
RHS at `stp2d.F90:112,126-202,243-281`; split-explicit SSH advances in
`dynspg_ts.F90`; and the stage-3 implicit solve consumes top drag and surface
stress at `dynzdf.F90:293-345,466-519`.  Coincident top and bottom indices in
one wet layer do not make the two coefficients interchangeable.

## Predictions and ordered stop rule

P14-FZP: source-statement rounding moves `POST_FZP.t_bo` from 2,189/2,190 to
**2,190/2,190 bit-identical**, with normalized error zero.  A private arm that
omits only the polynomial-statement round must reproduce the prior sole
non-bit row; a nonzero salinity-operand plant must make that row red.

P14-OCE: a new config-local scalar-math oracle copy will emit the six missing
frames, with derived header counts and runtime indices.  The production
legoESM driver will consume exact NEMO entry frames and walk, in order:

1. `POST_SBC_STAGGER`;
2. `POST_ZDF_DRG_COEFF`;
3. `PRE_DYN_SPG_TS`;
4. every `SSH_SUBSTEP`;
5. `POST_STP2D`;
6. `PRE_DYN_ZDF_SOLVE`.

For each boundary the prediction is pointwise normalized error at or below
`1e-15` and **all scored rows bit-identical**.  This is a falsifiable target,
not an agreement claim.  The gate stops at the first over-bar row, reports its
step/stage/substep/field and owner, and does not score later boundaries as
passed.  Each boundary receives a nonzero row-level planted corruption that
must exit nonzero.  Top-drag scaling uses factors 0.5, 1, and 2 on only
`rCdU_top`; the private arm zeros only that operand while retaining bottom
drag.

Only if all step-1 boundaries are bit-identical will the continuous driver and
exact-entry sweep advance through `kt=2..8760`.  The trajectory report will
name the first over-bar boundary in NEMO order and error growth; it will stop
if the owner requires a user decision.  `nn_fsbc=4` carries the registered
four-hour ice cadence through `sbc_ssm`, `eos_fzp`, `ice_update_flx`,
`tra_sbc_RK3`/SSH, `ice_update_tau`, and `sbc_fwb` without relabeling time
levels.

## Rule 1d and scope

Every new frame stores the executing `kt`, `Kbb`, `Kmm`, `Krhs`, `Kaa`, RK3
stage, and external `jn/icycle` when those indices exist.  NEMO entry frames,
not inferred “now” labels, are authoritative.  Oracle writes are WRITE-only
copies under a newly named config and data root.  The card explicitly selects
fp64 CPU and `transcendentals="libm"`.

Pierre's unmerged `lead_freeze_source` overlaps the existing shared
`_nemo_si3_ice_flx_other` transcription, not these ocean operators.  No second
lead-heat implementation is planned; the overlap remains a merge note.

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| source-literal TEOS-10 surface freezing point | ASKED | preregistered above |
| shared one-layer WS-RK3 ocean step and six boundaries | ASKED | preregistered above |
| full-year walk after a bit-exact step 1 | ASKED | conditional ordered continuation |
| shared top-drag consumer | ASKED | extend the canonical path only |
| new NEMO config/data copies and WRITE-only dumps | ASKED | required because round 13 has no six-boundary streams |
| modify shipped NEMO, delete retained runs, add a slab fork, GPU, `mpirun`, or push | UNASKED | forbidden / not planned |
| claim coupled fidelity beyond measured boundaries | UNASKED | withheld |

## CONFIRMED / PLAUSIBLE

**CONFIRMED:** the active NEMO branches and source order cited above; the
round-13 prefix has exactly one non-bit FZP row; the six requested ocean streams
are absent from its retained root; the shared legoESM path lacks an executing
top-drag consumer.  **PLAUSIBLE, UNMEASURED:** source rounding alone closes the
FZP bit and the shared one-layer composition reaches bit identity at every new
boundary.
