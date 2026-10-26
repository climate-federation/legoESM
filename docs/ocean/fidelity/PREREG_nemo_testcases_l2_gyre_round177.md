# Preregistration — round 177, developed tracer-LDF operand record

Committed before building or running the Round-177 acquisition.  Round 176
measured lateral tracer diffusion as carrying
`0.9999899512293968` of the developed step-1081 accumulated-temperature-
content error's signed projection.  The admitted Round-123 record contains the
temperature `Krhs` immediately before and after `tra_ldf`, but not the call's
direct tracer, live QCO geometry, slope, diffusivity, or flux operands.  The
Round-40/64 LDF records are from-rest, while Round 148 records momentum LDF.
No admitted later record contains the developed tracer-LDF operands.  This
round therefore acquires those operands before naming an internal statement.

No production physics, configuration, carried state, restart schema, card
default, or immutable Round-163 before arm changes.

## Compiled program and record

The admitted Round-132 GYRE card resolves the standard iso-neutral Laplacian
arm: `ln_traldf_lap=T`, `ln_traldf_iso=T`, `ln_traldf_triad=F`, and
`ln_traldf_msc=F`.  Its compiled dispatcher selects `traldf_iso_lap` at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/traldf.f90:69-110`.
The selected routine calls `traldf_iso_a33` and then evaluates masked tracer
gradients, horizontal fluxes, vertical fluxes, and their divergence in
compiled order at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/traldf_iso.f90:154-305`.

The acquisition extends that exact source card under the new target
`GYRE_OMIP_L2_P3_SM_R177TRALDF`.  It records step 1081, temperature only:

1. `T(Kbb)` and temperature `Krhs` before and after the call;
2. `r3t(Kmm)`, `r3u(Kmm)`, `r3v(Kmm)`, reference thicknesses and all metric
   reciprocals read by the selected statements;
3. `tmask`, `umask`, `vmask`, and `wmask`;
4. `ahtu`, `ahtv`, `uslp`, `vslp`, `wslpi`, `wslpj`, `ah_wslp2`, and `akz`;
5. the compiled masked gradients, matrix coefficients, horizontal and vertical
   fluxes, and the exact per-cell `Krhs` increment.

The writer is additive and active only for `kt=1081` and `jn=jp_tem`.  The run
continues through step 1440.  It uses the Round-132 prepared card and differs
only in `nn_itend=1440`, `nn_stock=360`, and the additive observer.  It never
changes an operand or state value.

## Frozen admissions and falsifiers

1. The syntax-proved source must contain exactly one begin, gradient,
   horizontal-flux, vertical-flux, and finish observer site in the active
   Laplacian routine.  Removing any site makes the layout plant print a named
   refusal and exit nonzero.
2. The compiled card must retain `key_qco`, `key_vco_1d3d`, `key_RK3`, the
   literal active call, and every cited statement.  A missing key, call, or
   statement refuses the acquisition.
3. The candidate step-1080 and step-1440 restart files must be byte-identical
   to the admitted uninstrumented Round-132 daily restarts.  Any moved byte
   means the observer perturbed NEMO and refuses the record.
4. The record has one fixed header and a fixed byte count derived from its
   declared arrays.  Wrong magic, dimensions, time levels, item count, dtype,
   or byte count refuses it.  A one-byte magic plant must print
   `STATUS PLANT-FIRED` and exit nonzero.
5. The record's `Krhs_after - Krhs_before` must be bit-identical over all
   18,000 active temperature cells to both (a) its stored per-cell increment
   and (b) the admitted Round-123 step-1081 after-LDF minus after-QSR
   increment.  Any unequal active cell refuses admission.  The cross-build
   Round-123 comparison is informational only if the two differently
   instrumented builds differ outside active cells; the Round-132 restart
   comparisons remain the passivity criterion.
6. Every stored intermediate is finite on its registered domain.  Each
   registered array must have at least one nonzero active value unless the
   compiled source proves that row is structurally zero under
   `ln_traldf_msc=F`; an unexpected all-zero row refuses interpretation.

The acquisition is evidence only.  Round 178 will extend the existing
year-owner harness, not create a second walk, and compare the recorded rows
under the production JIT step and the complete eager step.  It will name the
first non-bit statement only after the record passes every admission above.
The full Decision 43/45/55/59 trajectory gate runs only if that walk produces
a one-variable source-exact candidate.
