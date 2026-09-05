# NEMO testcase Lane 4 — ORCA2 Phase-2g O1/V2 preregistration

Date: 2026-09-06

Parent: `d44ca7abaaf4461ffb57014bbff31d89216c9c26`

Decision owner: user Phase-2g resume (ASKED).  The 91 reproducible streams from
the Phase-2f canonical twin are promoted to VARIANT oracle V2 now; the O1
stream is compared on source-defined fields and regenerated after its last two
unowned fields are canonicalized.

## Frozen provenance and claims

The two completed user-shell runs are
`variant_icebergs_off_phase2f_canonical_a_10step_np2` and
`variant_icebergs_off_phase2f_canonical_b_10step_np2`.  Each must independently
pass the frozen schema, ordinary-output identity, and planted controls.  The V2
record inventory is the 90 Phase-1 streams plus
`oracle_ocean_surface_input_kt00000001.bin`: all 91 must be raw-byte identical
between twins.  Run A supplies the pinned V2 bytes.  The earlier icebergs-off
instrumented records remain present and are labelled superseded; none is
deleted.

The O1 record remains provisional until the replacement twin is executed.  Its
two headers and all source-defined fields are nevertheless valid operands for
the O1 numerical gates below.

## O1 undefined-slot inventory and WRITE-only correction

The O1 schema stores only reduced `A2D(0)` arrays, 90 by 148 on rank 0; it has
no stored halo band.  Source inspection fixes the complete undefined set:

| frame | field | defined status | source reason and correction |
|---|---|---|---|
| 0 | all nine `sf(...)%fnow(:,:,1)` fields | defined in every stored cell | `fldread.F90:181-227,370-399,705-759` reads, spatially interpolates, applies its boundary link, rotates the wind pair, and selects/interpolates the current record |
| 1 | `pcd_du` | wholly unowned for this deck | the only assignment is guarded by inactive `ln_abl` at preprocessed `sbcblk.f90:896-906`; the resolved NCAR arm goes through `:908-936` and never consumes it |
| 1 | `qlwn` | wholly unowned for this deck | optional MFS-only output/consumer under inactive `ln_MFS`; it has no NCAR assignment |
| 1 | the other eighteen fields | defined in every stored cell by the selected pre-bulk/bulk/post-bulk path | retained exactly; no land or halo exclusion is introduced |

The writer-only fix creates one local `A2D(0)` temporary, fills it with
`0._wp`, and writes that value in the two schema positions instead of reading
`pcd_du` or `qlwn`.  It may not alter any model operand, expression, resolved
option, header, count, frame ordering, or any other writer.  The replacement
twin must prove 92 / 92 raw-byte identity.  Until then the defined-field twin
gate compares all headers, nine frame-0 fields, and eighteen owned frame-1
fields exactly, excludes exactly `2*90*148 = 26,640` f64 values, reports the
loop-derived counts, and carries a one-ULP plant in every compared O1 field.

## O1 numerical boundaries

Both gates run on CPU with JAX fp64 plus the explicit scalar-libm policy.  They
score the rank-0 reduced subdomain (90 by 148), not an independently gathered
global reconstruction.  Every field is cellwise exact: acceptance is `0 / n`
unequal source-defined cells and the plant changes one compared value through
the same gate and must exit nonzero.

### O1-M — resolved `fld_read` mapping (`ORCA2_OWNER`)

The nine registered CORE fields are U wind, V wind, air temperature, specific
humidity, shortwave, longwave, total precipitation, snow, and sea-level
pressure.  The candidate must use the deck's raw files, weight files, declared
time behavior, units/signs, and wind rotation.  NEMO establishes the order:
spatial interpolation and `lbc_lnk` in `fldread.F90:370-399`, wind rotation in
`:705-759`, and temporal selection/interpolation in `:181-227`.  At `kt=1`,
`ocean.output:1480-1512` resolves record 1 for all nine; the 6-hour fields span
0--0.25 day, radiation 0--1 day, and precipitation/snow 0--31 days.  A missing
shared `fld_read` implementation is `UNMEASURED_MISSING_IMPLEMENTATION`, not a
numeric pass.

### O1-B — certified NCAR bulk leaf (`LANE3B_OWNER`)

The candidate reuses the one shared Lane-3b-certified NCAR implementation,
`bulk_flux_omip.air_sea_fluxes(algo="ncar")`, under production JIT.  Operand
substitution supplies the nine oracle O1-M fields plus the oracle-recorded
ocean surface state; SI3 exchange remains `ORACLE_SUPPLIED` and its operators
remain `UNMEASURED_PENDING_ICE_MERGE`.  This isolates the NCAR leaf from
`fld_read`.  The compared outputs are the leaf-owned stress, sensible/latent
heat, evaporation, saturation humidity, wind magnitude, and any directly
corresponding stored field.  Assembly performed after the leaf is not claimed.

If O1-M or O1-B is over bar, report the first boundary in NEMO execution order
and its owner.  Lane 4 may fix O1-M because it is ORCA2-specific.  Any shared
bulk debt is registered for its owner and is not repaired here.

## Mechanical gates and stop

1. Validate both existing twins, derive the 91-stream inventory from the files,
   require 91 / 91 exact, and bind run A to a hash manifest as VARIANT oracle V2.
2. Run ordinary-output identity and every inherited/O1 plant through its real
   validator; all plants must exit nonzero.
3. Score O1-M and O1-B as preregistered.  Stop at the first decision boundary,
   naming the owner and retaining any later operand-substitution diagnostic as
   non-ordering evidence.
4. Apply only the O1 WRITE fix, rebuild with `conda-scalarmath`, require zero
   `_ZGV*`, and prepare two hash-guarded two-rank (`jpni=2`, `jpnj=1`) launch
   directories.  Do not execute MPI from the agent shell.

## ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| pin the reproducible 91 streams as VARIANT oracle V2 | ASKED | run A bytes, manifest-bound; older variant arms preserved and superseded |
| enumerate/fix the remaining O1 undefined slots | ASKED | exactly whole-field `pcd_du` and `qlwn`; zero-first WRITE-only temporary |
| start O1 mapping and certified-bulk comparisons | ASKED | exact cellwise gates, owners separated by operand substitution |
| prepare replacement twins | ASKED | one rebuilt binary, two independent directories and hash-guarded launchers |
| execute MPI | UNASKED and prohibited | user shell executes after this handoff |
| repair a shared numerical operator | UNASKED and forbidden | register for GYRE/Lane-3b owner only |
| enter SI3 dynamics/thermodynamics | UNASKED pending merge | oracle-supplied exchange only |
| delete or overwrite prior runs | UNASKED and forbidden | all prior records retained and provenance-labelled |
