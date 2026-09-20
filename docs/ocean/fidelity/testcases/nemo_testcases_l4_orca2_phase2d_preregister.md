# NEMO testcase Lane 4 — ORCA2 Phase-2d preregistration

Date: 2026-09-06

Parent: `6e935823577b3a60056e96bfca429f43e84ecac3`

Oracle label: `VARIANT` (`ln_icebergs=.false.`)

Scope: accept the executed variant oracle, bind the card to the same resolved
option, then walk the ocean kt=1 chain until the first over-bar or missing
source boundary.  No SI3 dynamics or thermodynamics is entered.

## V0–V3: oracle acceptance boundaries

| boundary | registered measurement | pass condition |
|---|---|---|
| V0 run closure | launcher logs, `time.step`, `ocean.output`, output census | 10, 10, and 240 steps; zero launcher/MPI status; `RUN DONE`; no NEMO error |
| V1 record schema | frozen Phase-1 90-stream walk plus `NEMO_L4_SBCIN_1` | exactly 91 streams; every magic/header/derived count/size/kt sequence valid; all payloads finite; six inactive iceberg slots exact zero |
| V2 WRITE-only identity | instrumented vs uninstrumented variant ten-step pair | every emitted restart/history/ordinary output exact under the Phase-1 timestamp and dump-notice rules; actual loop counts reported |
| V3 one-variable effect | variant vs accepted shipped-deck records | every frame emitted before the first kt=1 `icb_stp` effect byte-identical; later differences restricted to fields downstream of removed iceberg `emp/qns`; variant `utau_icb/vtau_icb` and iceberg diagnostic slots zero |

The Phase-1 identity gate hard-coded the icebergs-on restart and trajectory
families.  The variant correctly has no such files.  Before V2 is measured,
the same shared gate will derive restart/trajectory names from both controlled
directories, require the sets to be identical, require ocean+SI3 shards on
both ranks, require iceberg files iff the resolved namelist enables them, and
report the actual loop counts.  A planted asymmetric optional file and the
existing planted restart-byte mutation must both fail through
`validate_identity`.  This is an inventory correction, not a changed identity
rule.

The V3 pre-effect set is source-driven.  At kt=1 it includes the complete step
entry record, every kt=1 SI3 bulk/thermodynamic/ZDF/exchange frame, and both
kt=1 Prather files.  `stprk3.F90:91-103` writes step entry before `sbc`;
`icestp.F90:205-238` writes the SI3 streams and final SI3 exchange before
returning; `sbcmod.F90:484` is the first possible iceberg call.  Later frames
are not declared byte-identical because earlier removed melt/heat can already
have changed ocean state.

V3 will decode named payloads rather than treat a whole later record as one
number.  A changed field outside the source-allowed dependency cone is a hard
failure.  The shipped run has no post-`sbc` Phase-2b stream, so its missing
counterpart is stated; the variant stream still binds exact-zero inactive
slots and the pre-effect streams bind the controlled one-variable comparison.

All record, ordinary-output, restart, launcher, build, and retained gate
artifacts receive SHA-256 pins in the manifest.

## C0–C1: card parity and entry identity

The ORCA2 card will explicitly record `icebergs_enabled=False` and an empty
iceberg-input inventory.  `iceberg_state` is removed from its unmeasured
selected-feature list because the operator is inactive, not waived.  The card
validator and a planted enabled/nonempty iceberg arm bind both facts.

C1 re-runs the production-JIT CPU gate with fp64 plus scalar-libm against the
variant `oracle_step_entry_kt00000001.bin`.  Geometry and `T,S,u,v,ssh` must
remain byte-exact (`0 / n`) on the rank-0 owned `90 x 148 x 30` domain.  Any
movement stops the ladder.

## O1 onward: ordered ocean ladder

The full legoESM card remains `180 x 148 x 30`; detailed comparisons use only
the rank-0 interior represented by each `94 x 152` NEMO record after stripping
two halos.  Every comparison is cellwise and reports `unequal / n`, maximum
absolute error, maximum ULP distance, time level, and owner.

| order | boundary | oracle source/record | owner if over bar |
|---:|---|---|---|
| O1 | resolved CORE/NCAR `fld_read` inputs and bulk surface fields | `sbcmod.F90:415-424`; `sbcblk.F90`; `fldread.F90:181-227`; post-`sbc` and SI3 bulk records | `ORCA2_OWNER` for file semantics/fold regridding; shared bulk arithmetic otherwise `LANE3B_OWNER` |
| O2 | exact oracle-supplied post-SI3 surface exchange operands | `iceupdate.F90`; `icestp.F90:217-238`; `oracle_ocean_surface_input_kt00000001.bin` | producer `UNMEASURED_PENDING_ICE_MERGE`; substitution loader `ORCA2_OWNER` |
| O3 | `eos_rab`, `bn2`, ZDF/TKE/EVD/IWM entry | `stprk3.F90:175-203`; EOS/ZDF records | shared debt `GYRE_OWNER`; ORCA2 IWM file semantics `ORCA2_OWNER` |
| O4 | external-mode slow forcing and 65 QCO substeps, including north fold | `stprk3.F90:207-211`; slow-forcing/barotropic records | fold `ORCA2_OWNER`; shared external mode `GYRE_OWNER` |
| O5 | stage-1 transport construction and momentum state | `stprk3_stg.F90:248-347`; transport/stage records | shared transports `GYRE_OWNER` |
| O6 | stage-1 FCT tracer advection and surface tracer forcing | `stprk3_stg.F90:566-730`; tracer operand/stage records | shared FCT `GYRE_OWNER`; ORCA2 runoff/RGB forcing `ORCA2_OWNER` |
| O7 | stage-3 RGB `tra_qsr`, BBL, geothermal, lateral and vertical closure entries | `stprk3_stg.F90:734-798`; QSR/tracer/ZDF records | BBL/RGB/geothermal/IWM `ORCA2_OWNER`; shared closure debt `GYRE_OWNER` |

The apparent ordering in a subsystem list does not override the executed call
order above.  The lane stops at the first boundary that is over bar or lacks a
required oracle operand.  An authorized ORCA2-specific missing implementation
is added only to the shared path with NEMO source citation; a shared numerical
debt is registered and handed to GYRE without repair.

All measurements use production JIT, CPU, explicit fp64 and scalar-libm.  Each
new gate has a one-variable planted control that exits nonzero.  Existing
canonical blocks are searched and extended before any helper is added.

## ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| accept and pin three executed icebergs-off runs | ASKED | V0–V3 gates above |
| explicit card `ln_icebergs=F`, no iceberg inputs | ASKED | C0 metadata plus validator/plant |
| oracle-supplied SI3 exchange; ice operators pending merge | ASKED | O2 input substitution only |
| full ocean stage-1 ladder, stop first over-bar/decision boundary | ASKED | O1–O7 order frozen |
| fix only ORCA2-specific work; hand shared debt to GYRE | ASKED standing | owner column binds disposition |
| derive identity inventory for the inactive component | UNASKED enabling gate repair | fail-closed controlled-pair census; scientific equality rule unchanged |
| score rank-0 interior only | ASKED inherited | `90 x 148`, with field staggering and wet-mask census |
| any SI3 implementation or iceberg model | UNASKED and forbidden | not entered |

No unasked scientific option, threshold, cadence, state variable, or input
source is selected.

## O1 acquisition addendum (before measurement)

The accepted 91-stream variant record set closes V0--V3 and C0--C1, but it
cannot separate the two owners inside O1.  The final post-`sbc` record is
written after `sbcblk`, SI3, runoff, freshwater-budget carry, and halo/stress
assembly.  The inherited SI3 bulk stream contains only three selected column
samples.  Neither records the complete `fld_read` result or the open-ocean
`blk_oce_2` result on rank 0.  A disagreement against the final record would
therefore be owner-ambiguous between `ORCA2_OWNER` input semantics and
`LANE3B_OWNER` bulk arithmetic.

Before implementing or scoring O1, one replacement instrumented variant run
is registered.  A configuration-local, `lwp`-guarded WRITE-only extension of
`MY_SRC/sbcblk.F90` will write exactly two kt=1 frames:

1. immediately after `fld_read` (`sbcblk.F90:559`): the nine rotated/mapped
   `sf%fnow` fields selected by `namsbc_blk`;
2. immediately after `blk_oce_2` (`sbcblk.F90:623-634`): processed air state,
   precipitation, ocean surface state, NCAR intermediates, and the open-ocean
   stress/heat/freshwater outputs, before SI3 or runoff can change them.

Both frames use the live `A2D(0)` allocation and carry their field count in
the header; the validator will derive payload size from `(nx,ny,nfields)` and
walk both headers.  The accepted icebergs-off scalar-math binary is rebuilt
with no other source change.  The replacement run is ten steps at `np=2`,
`jpni=2,jpnj=1`, and must remain byte-identical to the accepted variant
uninstrumented ten-step control for all ordinary outputs.  A header-count
plant and one-ULP payload plant must exit nonzero.  No O1 implementation or
comparison is permitted until this record passes.
