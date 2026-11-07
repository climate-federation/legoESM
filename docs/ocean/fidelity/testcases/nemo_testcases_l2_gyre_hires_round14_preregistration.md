# GYRE high-resolution round 14 online-budget preregistration

Date registered: 2026-10-18

This registration precedes every round-14 online-budget acquisition.  It
freezes the measurement requested after round 13 found that daily state means
cannot identify production tendencies.

## Claim and window

The comparison will measure the native surface-V momentum increment on the
`r4_smag` card for each model day 30 through 180 inclusive.  Each daily field
is the arithmetic mean of the 24 one-hour, per-step increments accumulated
inside the compiled step.  The later owner score uses the already registered
western-boundary wet-span bin; the archive retains every native surface-V
point so that location is not selected after seeing the terms.

The trajectories are not expected to match.  The test is which production
term carries their algebraic days-30--180 separation, not whether the two
models follow the same weather.

## Registered rows

The archive order and definitions are fixed as follows:

1. `advection`: stage-3 `dyn_zad`, including its vertical-advection part,
   multiplied by the full stage-3 `rDt`.
2. `pressure_kinetic_energy_gradient`: stage-3 `dyn_hpg + dyn_keg`, multiplied
   by `rDt`.
3. `total_vorticity_coriolis`: stage-3 `dyn_vor`, including the card's total
   vorticity/Coriolis flux, multiplied by `rDt`.
4. `smagorinsky_lateral_diffusion`: stage-3 `dyn_ldf`, multiplied by `rDt`.
5. `vertical_mixing`: the implicit-ZDF solution with the surface-stress RHS
   withheld minus the pre-ZDF stage-3 velocity.
6. `surface_stress`: the production implicit-ZDF solution minus the companion
   no-stress solution through the identical already-formed matrix.
7. `external_mode_correction`: the velocity after the final barotropic-mean
   replacement minus the production post-ZDF velocity.
8. `total_increment`: the final stage-3 surface V minus the independent
   whole-step entry surface V.

No unregistered remainder may be relabelled as a physical row.  The first
seven rows must sum to the independently recorded eighth row.

## Closure and controls

The preregistered absolute closure bar is
`1.0e-12 m s-1` at every wet surface-V point on every admitted day.  This is
the lane's existing two-day composition-control bar, now applied only to an
algebraic sum of source-boundary increments; it is not a model-agreement bar.
Closure above the bar is `REFUSE`, never an owner attribution.

Both providers must also supply a bitwise passive-identity artifact with zero
moved prognostic words for recorder-off versus recorder-on integrations from
the same entry state.  Plants must prove that (a) changing one active term
breaks closure and (b) changing one active prognostic word breaks passive
identity.  Non-finite fields, a changed term registry, a changed window, a
missing external-mode row, or a combined vertical/stress row all refuse.

## Precision and provenance

The NEMO reference is the vector-math `GYRE_OMIP_L2` `r4_smag` program used by
the five-year vector twin, run on one MPI rank because the custom records are
not gathered.  The legoESM acquisition uses the card's native-fp64 GPU policy
and an explicit physical-GPU argument.  Both contracts must contain the
producer commit, exact resolved namelist, writer/source hashes, day/step
cadence, data hash, passive-identity hash, and this preregistration's commit
and hash.  Data collected before this file's commit are inadmissible.

Tracer terms are intentionally out of scope.  Round 13 registered native
surface V as the leading field, and an exact implicit tracer-content split
would be a second instrument rather than a cheap extension of this momentum
measurement.
