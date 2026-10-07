# ORCA2 round 165 — kt=8 vertical-coordinate boundary hold

Date: 2026-10-07. Base `f589ec773`; preregistration `91594ccc5`;
measurement instrument `f77edfe19`. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round165/`.
Verdict: **HELD**. The complete private V-transport arm still refuses between
kt=8 stages 2 and 3, but the first invalid vertical geometry is already
present in the exposed stage-1 state. The next producer boundary is the
barotropic after-SSH state, before stage-1 interpolation.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity and zero sea surface. No given-NEMO-entry
rung-7 number is mixed into the table. Sea ice, all six sea-ice selectors and
the shipped card's `unmeasured_features` tuple are unchanged.

## Source order and passive measurement

The compiled rung-0 stage program saves the barotropic after-SSH as `ssha`,
then constructs the hybrid stage-1 SSH and its live free-surface ratios at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/stprk3_stg.f90:132-179`. Stage 2
reuses the same saved after-SSH at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/stprk3_stg.f90:185-226`. The later
adaptive vertical split divides by the live Kmm thickness and partitions the
vertical velocity at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/sshwzv.f90:736-767`.

The measurement reused round 81's scalar-only bad-`e3w` callback and extended
the existing round-103 ladder with a host-only observer after its already
materialised stage-1/2 states. It did not replace, clip or feed an observed
value back into the model. Both the unobserved and observed complete private
arms completed kt=7 stage 3, exposed kt=8 stages 1-2, did not complete kt=8
stage 3, and refused on the exact invariant text `raw-mesh e3w_int must
contain only finite values > 0`. Thus R165-P1 is **CONFIRMED**.

## First invalid boundary

The callback first fired in
`legoesm.ocean.physics.convection.enhanced_diffusion` at flat index 6,641,
or `[j=1, i=49, k=0]`, in a `[148, 180, 29]` field. It counted 476,557
invalid interfaces: 16,433 columns times 29 interfaces. Its reported
`-Infinity` is the observer's non-finite ordering sentinel, not a physical
thickness.

| exposed kt=8 boundary | SSH (m) | `r3t` | stretch | raw `e3w_0` (m) | product (m) | invalid interfaces |
|---|---:|---:|---:|---:|---:|---:|
| step entry | -1.149276160889075 | -0.0015765105087641631 | 0.9984234894912358 | 10.00035061113249 | 9.984584953302713 | 0 |
| stage 1 | NaN | NaN | NaN | 10.00035061113249 | NaN | 476,557 |
| stage 2 | NaN | NaN | NaN | 10.00035061113249 | NaN | 476,557 |

This refutes R165-P2: the first exposed invalid geometry is stage 1, not the
predicted stage-3 half-step. It also refutes R165-P3: the raw mesh operand is
finite and positive, but the stretch is NaN rather than a finite crossing of
zero. The callback sentinel cannot equal a NaN source product, so the gate
retains the preregistered falsifier as
`UNMATCHED_REQUIRES_NEXT_WALK`; it does not manufacture a match.

The entry SSH is finite. The compiled stage-1 hybrid expression combines that
finite entry with `ssha` using finite constants. Therefore the saved
barotropic after-SSH is already non-finite before the stage-1 interpolation.
This is the next source boundary; localization alone does not identify or
authorize a barotropic statement. R165-P4 is **CONFIRMED**: there is no
production package or configuration change.

## Controls, validation and review

The `source-product` plant changes the disposition to a value outside the
closed classifier set and exits nonzero with `PLANT-FIRED`; the unmodified
report passes as `PASS_ROUND165_VERTICAL_BOUNDARY`. The measurement artifacts
and controls are hashed in `SHA256SUMS`.

The new gate's focused unit coverage passes 7/7, including direct CLI import,
terminal passivity, the finite-entry requirement, and the retained unmatched
falsifier. The cumulative and round-receipt citation gates pass with zero
unmapped spans; shifting the stage-1 citation by two lines makes the planted
citation gate fail.

The separate `codex exec --sandbox read-only` attempt returned **independent
review unavailable in-sandbox** before reading the diff: `failed to initialize
in-process app-server client: Read-only file system`.

ASKED choices: continue round 164's passive vertical-coordinate walk. UNASKED
choices: empty. No configuration, forcing, carried-state policy, stabiliser,
sea-ice selector or production model statement changed.

## OPEN

1. Under the same complete private arm, expose kt=8's barotropic after-SSH and
   its external substeps using the existing passive barotropic trace pattern;
   prove the observer reproduces this round's terminal boundary.
2. Walk the first finite-to-non-finite SSH/transport substep in compiled source
   order, one recorded operand at a time. Do not add a stabiliser or land only
   part of the held four-statement source unit.
3. Keep the atomic V-transport unit and halo pair **HELD** until rung 0
   completes and the full Decision-96 direction, exact-row and SSH predicates
   pass.
