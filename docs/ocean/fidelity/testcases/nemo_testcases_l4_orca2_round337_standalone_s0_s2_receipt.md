# ORCA2 round 337 — standalone S0 and S2 model preparation

Date: 2026-10-26.  Base `aae42be88d67`; preregistration commit
`f6b789706`; model commits `50e135540` and `4b7083a09`.  Final disposition:
**LANDED model preparation; no acquisition tool, NEMO configuration, or NEMO
run was made.**  S0/S2 groups present in the immutable round-5 record are at
the exact bar; all other groups remain explicitly unmeasured pending the
already planned S4 acquisition.

## Source and executing branch

All source transcription was taken from the compiled round-5 record branch,
not a pristine or dead source arm.  `fld_read` forms the no-`kit` model time
at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/fldread.f90:205`, updates
only at the forcing cadence at `:217`, and evaluates the two record weights
and blend at `:244-246`.  Its before/after record search and swap are
`:309-366`; record centres, including integer halves, odd carry,
and `rec_shft`, are `:932-936`.

The every-step mean is the `nn_fsbc > 1` arm: from-rest initialization,
reset, accumulation, and division of `ssu_m,ssv_m,sst_m,sss_m,ssh_m,e3t_m,
frq_m` are
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcssm.f90:109-174`.
For `nn_fsbc=2`, kt 1 initializes and divides the duplicated first sample, kt
2 resets and accumulates once without division, and kt 3 completes the next
two-sample mean.  Restart-provided mean state is an explicit operand and does
not enter the from-rest branch.

After the separately owned ice-interface boundary, the compiled call order is
runoff, restoring, freshwater-budget correction, halo fill, and native-face
stress construction in
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcmod.f90:428-568`.
The deck's active runoff arm and disabled runoff temperature/salinity files
use SST-clamped-at-zero temperature content and zero salinity content at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcrnf.f90:185-227`.
The `nn_sssr=2`, `nn_sssr_ice=1` volume-flux restoring arm, runoff-mouth
damping, bound, and associated heat flux are
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcssr.f90:147-205`.
The deck's `nn_fwb=2` annual state and forcing-cadence `emp/qns` correction
are
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcfwb.f90:277-332`.
The resolved card pins `nn_fwb=2` at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/EXP00/namelist_cfg:92`, and
`nn_sssr=2`, `rn_deds=-166.67 mm/day`, a live bound, and
`rn_sssr_bnd=4 mm/day` at `:153-156`; `rn_rfact=1` is `:164`.  Its inherited
runoff depth, temperature, and salinity input switches are all false at
`cfgs/SHARED/namelist_ref:474-476`.

## What landed

S0-F extends the shared field reader with the compiled integer-time program:
Fortran `NINT`, model midpoint time, record-centre construction with
truncating integer division and Fortran `MOD`, strict before/after selection,
and the source-ordered temporal blend.  File cycling and calendar assembly
remain loader policy; all physical/calendar operands are required and an
unbracketed time refuses.

The exact boundary inherited from phase 2i is narrow: it certified the nine
spatial CORE-to-ORCA2 maps and rotation, and its admitted chlorophyll input
was already time-interpolated.  It did **not** certify generic temporal record
selection, record-centre construction, `rec_shft`, or blending.  This round
reuses that spatial implementation without relabeling it as temporal
evidence.

S0-M adds one JAX-pure state transition for the seven means.  It accepts the
complete prior state and all instantaneous state operands, supports both
potential- and conservative-temperature branches, inverse-barometer removal,
from-rest initialization, restart preservation, reset, accumulation, and
forcing-boundary division.  A four-step `jax.lax.scan` test pins the stateful
cadence.

S2 extends, rather than duplicates, the existing shared runoff and
`sss_restoring.py` code.  Its entry is explicitly post-S3: post-ice
`utau,vtau,taum,qsr,qns,emp,sfx` are required operands.  It performs the
forcing-cadence runoff mass/tracer update, SSS restoring, freshwater-budget
merge, native U/V stress construction, and `qsr` penetrative hand-off.  The
prior `rnf/rnf_tsc/erp/qrp` states are also required so the non-update step
preserves NEMO's module diagnostics.  The adapter returns the existing
`OceanSurfaceForcing` and `FreshwaterForcing` interfaces and selects the
existing surface-only runoff deposition route; it introduces no second
depth-distribution formula.

For K0 `nn_ice=1`, there is no additional S2 ice freshwater or salt channel.
All heat changes made by the ice-interface routine arrive only through the
explicit post-ice operands.  No `sbcice_if` code, ice-cover plan, vertical
solve, card, or configuration was touched.

Every new physical input has no default.  Every NEMO assignment and internal
association has an M1 source-rounding barrier, including multiplication
associations and `1/rho0` before runoff tracer content.  The whole-call eager
arm is inside `jax.disable_jit()`; it is not a compiled function renamed
eager.

## Object-code tie and transcendental audit

The round-5 `sbcssr.o` executes scalar `MAXSD` for the salinity floor,
`ANDPD` plus `MINSD` for the magnitude bound, and calls
`__lib_fortran_MOD_sign_scalar` at relocations `0x4e57`, `0x4f61`, and
`0x5043`.  Disassembly of the linked `lib_fortran.o` helper compares the sign
operand with positive zero and branches only for below, so `SIGN` treats
negative zero as positive.  The shared Fortran-sign helper and a negative-zero
test pin that result.

The executing `sbcrnf.o` `MAXSD(sst,+0)` retains the source positive-zero
operand on a signed-zero tie.  The S2 scalar test pins runoff temperature
content to positive zero.  Equal positive operands make the `MINSD` bound and
salinity-floor `MAXSD` result bit-identical whichever operand survives;
stress masks are only zero or one.  These exhaust every executing
MIN/MAX/SIGN site in S0/S2.

There is no exponential, logarithmic, trigonometric, or power operation in
the S0/S2 numeric path.  The required scalar-glibc arm is therefore present
but vacuous here: no transcendental call is made.  The record object identifies
GCC conda-forge 15.2.0-19.

## Exact offline score

The immutable round-5 record was verified by digest before scoring:

- entry `O1.npz`:
  `751b2d9181778e81f01bc5872d47100afc0fc3ad02c4ffcccc9613a0af2ad045`;
- `surface_every_step.npz`:
  `34fa2926f7a0fd0e91eec326973b070fb5f9b1193cb1133de60b10b020fca344`;
- mesh:
  `0f373c6609d287bc818b1f6ef96bafdcc0b620c730219018de9728369d01ec5f`.

The registered before/current counts on CPU binary64 JIT were:

| Independently scorable group | Parent over-bar / words | Current over-bar / words |
|---|---:|---:|
| runoff temperature content | 1,340 / 8,794 | **0 / 8,794** |
| runoff salinity content | 0 / 8,794 | **0 / 8,794** |
| native U stress | 8,489 / 8,489 | **0 / 8,489** |
| native V stress | 8,554 / 8,554 | **0 / 8,554** |
| total | 18,383 / 34,631 | **0 / 34,631** |

One-ULP plants in each of the four output groups exit nonzero.  S0-F record
selection/blending, S0-M means, SSS/FWB merge, and the complete
`qns/emp/sfx/taum/qsr` merge are `UNMEASURED_NEEDS_ACQUISITION`: the round-5
record lacks their entry operands.  No output was injected as an input and no
aggregate was used to promote those groups.

The inherited round-335 S1/O1 results remain 0 / 158,292 bulk words and
0 / 119,880 spatial-map words.  This round changes neither S1 arithmetic nor
the certified spatial operators.

## Non-NEMO compatibility and tests

The new exact arms are opt-in.  A real parent/current census exercised both
generic SSS-restoring configurations, freshwater normalization/dilution and
surface-state updates, the coupler SSS application, and the OMIP scan surface
path.  All **25 arrays / 328 binary64 words are identical**; a one-ULP plant
in `omip_scan/q_net` exits nonzero.  This adds focused coverage to round 335's
unchanged 214-array / 10,231,076-word full-caller control.  No change to a
non-NEMO caller is requested or observed.

- focused S0/S2 fidelity suite: 6/6 PASS; independent scalar transcription is
  bit-exact under CPU JIT and whole-call eager execution, four-step S0 scan,
  signed-zero ties, finite-difference gradient, production-call-site
  monkeypatch plant, and no physical defaults;
- shared field-reader, freshwater/runoff, SSS-restoring, and coupler
  regression suites: 159/159 PASS;
- offline measured-group gate: 0 / 34,631 after, with all four one-ULP plants
  nonzero;
- focused non-NEMO parent/current control: 0 / 328, with its plant nonzero;
- hardcoded-constant and inline-coefficient ratchets: 6,176 PASS, 2 SKIP, 2
  inherited FAIL.  The identical parent failures are the JRA55 test's
  `273.15` literal and two land-restart findings; neither names a changed file.

The authoritative default ORCA2 inventory passed 152/152 in 1,341.39 s.  The
default citation test and this receipt's 14-citation gate both pass with zero
unmapped entries, zero failures, every map entry shift-sensitive, and every
self-test firing.  A shifted round-337 source citation exits nonzero.

## Acquisition preregistration and choices

The committed preregistration freezes independent A/B twins, ranks 0--1,
steps 1--10, every S0-F call, every S0-M step, S2 entry/sub-boundaries/exit,
native domains, manifests, and all ASKED/UNASKED choices.  It deliberately
contains no writer, patch, build, configuration, or `run.sh`.  The future S4
owner builds the one composed acquisition after S3 is available.

- ASKED and complete: literal S0-F numeric helpers, stateful S0-M, literal
  post-S3 S2, shared runoff/restoring extensions, explicit inputs, barriers,
  object tie audit, tests, offline score, compatibility control, and future
  acquisition preregistration.
- UNASKED and unchanged: `sbcice_if`, the ice-cover plan, vertical solve,
  NEMO acquisition tooling/execution, all configurations/cards, comparison
  bars, and non-NEMO defaults.
- No new decision is needed.  The missing evidence is already assigned to
  the composed S4 acquisition, not to an unrequested round-337 tool.

independent review: operator (Claude)
