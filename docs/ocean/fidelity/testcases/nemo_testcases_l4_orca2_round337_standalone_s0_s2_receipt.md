# ORCA2 round 337 — standalone S0 and S2 model preparation

Date: 2026-10-26.  Base `aae42be88d67`; preregistration commit
`f6b789706`; model commits `50e135540`, `4b7083a09`, and review repair
`6c9e6c00a`.  Final disposition:
**LANDED model preparation; no acquisition tool, NEMO configuration, or NEMO
run was made.**  S0/S2 groups present in the immutable round-5 record are at
the exact bar; all other groups remain explicitly unmeasured pending the
already planned S4 acquisition.

Resume 2 rebases the round onto the landed S3 tip `94ec246261`, retains both
histories, and closes the second Claude HOLD.  It changes no NEMO run,
configuration, acquisition tool, or comparison bar.

## Independent-review HOLD closure

The Claude and Argon HOLDs were confirmed and corrected.  SSS restoring now
evaluates the compiled `erp * rcp * sst_m` association once and subtracts that
materialized term independently from `qns` and zero-initialized `qrp`, matching
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcssr.f90:147-205`.
The seven-field mean now evaluates inverse-barometer accumulation as
``(ssh_m + ssh) - ssh_ib`` in the order shown by
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcssm.f90:109-174`.

The saved restoring coefficient is now an explicit input/output state and is
multiplied by the runoff-mouth factor at every forcing update, rather than
being rebuilt.  Between forcing updates, runoff/restoring diagnostics and
`emp/qns` are returned without executing an add-zero expression.  The scalar
replica was rewritten directly from the Fortran assignments and independently
handles update and non-update arms.

S0 and S2 are now selectable around the existing NCAR S1 entry point.  The
composed path rejects half-selection and shadow mean inputs, returns the S0
and S2 carried states plus the existing forcing interfaces, and has a real
entry-point plant for each call.  The complete composed call is bit-identical
under whole-call eager execution and CPU JIT.  The strengthened mean test was
also run against parent `0be9d2be51`: its association case failed with parent
`+0` versus the required `-1/3`, then passed on the corrected tree.

The composed production path now obtains the centre, east, and north stress
operands from the same-step bulk stress through an explicit NEMO T-vector LBC
index/sign plan.  Caller-supplied neighbour stresses no longer exist in the
composed input.  This follows the halo-fill-before-native-face ordering in
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcmod.f90:428-568`.
A production-call plant replacing those derived neighbours with deliberately
wrong values changes 5 of 6 U-face words and exits nonzero.

Because S3 landed meanwhile, the path composes it rather than silently naming
the raw bulk result post-ice: the order is S0 mean, S1 bulk, S3 `sbc_ice_if`,
then S2.  S3 receives the same-step S0 `sss_m/sst_m`, not stale copies from its
caller.  Selecting S0/S2 without `nn_ice=1` now refuses explicitly.

The field-reader record loop is literal at equality too.  NEMO advances only
for the strict `model time > after time` test at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/fldread.f90:309-366`, so an
exact centre remains the after record.  The equality test fails when the
selector is mutated back to `searchsorted(..., side="right")`.

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
`sss_restoring.py` code.  Its direct helper entry remains explicitly post-S3,
but the composed production entry constructs those operands itself.  It runs
the landed `nn_ice=1` S3 implementation after S1 and before S2, applies the
explicit T-vector halo/LBC plan to the freshly computed stress, and passes the
resulting centre/east/north values to native-face construction.  It performs
the forcing-cadence runoff mass/tracer update, SSS restoring,
freshwater-budget merge, native U/V stress construction, and `qsr`
penetrative hand-off.  The prior `rnf/rnf_tsc/erp/qrp` states are also
required so the non-update step preserves NEMO's module diagnostics; the
saved restoring coefficient is carried too.  The adapter returns the existing
`OceanSurfaceForcing` and `FreshwaterForcing` interfaces and selects the
existing surface-only runoff deposition route; it introduces no second
depth-distribution formula.

For K0 `nn_ice=1`, there is no additional S2 ice freshwater or salt channel.
All heat changes made by the composed S3 ice-interface routine arrive through
its returned post-ice operands.  Resume 2 changes no `sbcice_if` arithmetic,
ice-cover plan, vertical solve, card, or configuration.

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
operand on a signed-zero tie.  `sbcssr.o` is site-specific: the peeled-cell
`MINSD` at offset `0x4e4a` and the second unrolled site at `0x5034` retain the
bound, while the first unrolled site at `0x4f55` retains `ABS(zerp)`.  GCC
peels one cell for an odd local-i extent and then unrolls by two, so the test
pins `[bound, abs, bound]` for width three and `[abs, bound, abs, bound]` for
width four.  A checked proof accompanies the site map: both operands are
nonnegative, and finite numeric equality therefore makes every reachable
MINSD tie bit-identical; the former unreachable negative-zero-bound test is
gone.  The reachable `MAXSD` signed-zero test remains bit-changing, and the
S2 scalar test separately pins runoff temperature content to positive zero
and Fortran `SIGN` treating negative zero as positive.  These exhaust every
executing MIN/MAX/SIGN site in S0/S2.

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

The registered current counts on CPU binary64 JIT are:

| Independently scorable group | Current over-bar / words |
|---|---:|
| runoff temperature content | **0 / 8,794** |
| runoff salinity content | **0 / 8,794** |
| native U stress | **0 / 8,489** |
| native V stress | **0 / 8,554** |
| total | **0 / 34,631** |

The earlier “Parent” column was removed: it was a hand-built counterfactual,
not execution of parent code, and therefore was not a valid parent control.

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

The independent re-review also reports **0 unequal in 117 arrays / 3,316,846
binary64 words**, including **0 unequal in 46 JRA55-bulk arrays / 745,200
words**, across three seeds.  Resume 2 reran the focused parent/current
control above after the rebase and obtained the same 0 / 328 result.

- focused S0/S2 fidelity suite: 7/7 PASS; independent scalar transcription is
  bit-exact under CPU JIT and whole-call eager execution, four-step S0 scan,
  the reachable signed-zero tie plus the per-site MINSD proof, coefficient
  carry, finite-difference gradient, composed production-call-site plants,
  strict field-record equality, and no physical defaults;
- affected round-337/S3/NCAR-bulk suites: 49/49 PASS; the production-path
  stress-neighbour plant, S3-order plant, S0 plant, and S2 plant all turn the
  relevant composed test red;
- NCAR bulk plus the directly affected freshwater and SSS-restoring suites
  pass;
- offline measured-group gate: 0 / 34,631 after, with all four one-ULP plants
  nonzero;
- focused non-NEMO parent/current control: 0 / 328, with its plant nonzero;
- the unchanged S1/O1 numerical gate remains exact for all 158,292 bulk and
  119,880 mapping words, and all per-field plants fire;
- hardcoded-constant and inline-coefficient ratchets: **6,186 pass, 2 skip**.
  Decision 166's two inherited reds were cleared without production
  arithmetic changes: the JRA55 test now names the canonical freezing-point
  constant, while the land restart I/O chunk size and validation probe carry
  explicit non-physics provenance.  The land restart suite is 22/22 PASS.

The authoritative default ORCA2 inventory passed 152/152 in 1,324.33 s.  The
default citation test and this receipt's 14-citation gate both pass with zero
unmapped entries, zero failures, every map entry shift-sensitive, and every
self-test firing.  A shifted round-337 source citation exits nonzero.  The
rebase initially exposed one duplicated S3/S0 `fldread` citation-map key; the
duplicate was removed, its dedicated ratchet turned green, and the full
152-test default battery was then rerun at the corrected tip.

## Acquisition preregistration and choices

The committed preregistration freezes independent A/B twins, ranks 0--1,
steps 1--10, every S0-F call, every S0-M step, S2 entry/sub-boundaries/exit,
native domains, manifests, and all ASKED/UNASKED choices.  It deliberately
contains no writer, patch, build, configuration, or `run.sh`.  The future S4
owner builds the one composed acquisition after S3 is available.

- ASKED and complete: literal S0-F numeric helpers, stateful S0-M, composed
  S0/S1/S3/S2 production order, same-step LBC stress neighbours, shared
  runoff/restoring extensions, explicit inputs, barriers, per-site object tie
  audit, tests, offline score, compatibility control, and future acquisition
  preregistration.
- UNASKED and unchanged: S3 arithmetic, the ice-cover plan, vertical solve,
  NEMO acquisition tooling/execution, all configurations/cards, comparison
  bars, and non-NEMO defaults.
- No new decision is needed.  The missing evidence is already assigned to
  the composed S4 acquisition, not to an unrequested round-337 tool.

codex review: operator.  Argon LAND and Claude HOLD were re-reviewed; all
confirmed findings are dispositioned above.
