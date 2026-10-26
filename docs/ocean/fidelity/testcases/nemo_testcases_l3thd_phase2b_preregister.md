# Lane 3b SI3 thermodynamics Phase 2b preregistration

Tracker: `climate-federation/legoESM#1699`

Status: **PREREGISTERED before the arithmetic replay and full-year sweeps.**

## Fixed identity and inputs

This continuation changes no physical selector.  The only executable identity
remains the ORCA1-resolved single-category HFN column: BL99 3+3 layers, P07
conductivity, option-2 salinity with `rn_sinew=0.75`, aEVP/Prather and
ridging/rafting inert in C1D, `ln_pnd=.false.`, and `ln_icedA=.false.`.  It runs
on CPU with explicit legoESM fp64 policy.  The accepted oracle root is
`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_phase2_inputs`.
Its thermodynamics stream has 8 frames for each of 8,760 steps (70,080 total),
SHA-256 `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b`;
the exact ZDF-entry-input stream has SHA-256
`5522eadce595408b00065fa30d8b41fccb3815bee76d6fbf5ba3adbb2656cb27`.

The column consumes NEMO's exact dumped `qns_ice` and `dqns_ice`; therefore
this rung does **not** certify the upstream bulk-flux computation at
`sbcblk.F90:1273,1480-1491` feeding `icestp.F90:201,206`.

## Arithmetic discriminators

The two registered near-bar debts are treated independently:

1. kt1 iteration-2 `qns_ice`, normalized error
   `2.0979870951387943e-14`;
2. `kt3.POST_ZDF.e_i`, normalized error `2.559660701514903e-15`.

NEMO stores the unnormalised Thomas diagonal/RHS, updates them as
`b[k]-a[k]*c[k-1]/b[k-1]` and `d[k]-a[k]*d[k-1]/b[k-1]`, then performs back
substitution (`icethd_zdf_bl99.F90:516-558`).  The shared legoESM Thomas solver
normalises each row first (`tridiagonal.py:103-159`).  NEMO updates
`qns_ice = qns_ice + dqns_ice * (t_su-ztsub)` before the solve
(`icethd_zdf_bl99.F90:367-379`), and recomputes ice enthalpy after convergence
(`icethd_zdf_bl99.F90:799-804`; `icevar.F90:938-946`).

A copy-only probe will dump the kt3 NEMO matrix, forward-elimination state,
solution temperatures, and post-call temperature/salinity/enthalpy operands.
For each row, ULP distance means the monotonic integer distance between finite,
same-sign IEEE-754 binary64 values.

- **FLOAT RE-ASSOCIATION:** using identical dumped operands, a scalar fp64
  replay in NEMO's written operation order reproduces the NEMO target within
  2 ULP, while the normalised shared-Thomas order reproduces the recorded
  residual.  Disclose the arithmetic floor; do not alter physical selectors.
- **OPERAND/TIME-LEVEL DIFFERENCE:** the NEMO-order replay remains more than
  2 ULP away or a registered input differs before the arithmetic operation.
  Replace only the first differing operand through a private arm.  If the arm
  moves the downstream residual by at least 100x, fix that difference inside
  the fixed SI3 identity and rerun; otherwise report it unowned.

For iteration-2 `qns_ice`, replay both the flux update itself and its upstream
iteration-1 surface-temperature operand.  For kt3 `e_i`, replay both the
unnormalised Thomas solve and the exact parenthesisation in
`icevar.F90:943-945`.  A physical conclusion is forbidden from norm magnitude
alone.

## Full-year protocols and metrics

Two protocols are reported so operator error is not conflated with accumulated
trajectory error:

1. **Oracle-entry operator sweep:** for each step independently, initialise
   legoESM from that step's NEMO ENTRY frame, apply the exact dumped boundary
   inputs, and compare all eight registered frames in write order.  This is the
   protocol that defines the first over-bar `(step, sub-call, variable, size)`.
2. **Continuous driven column:** initialise once from NEMO kt1 ENTRY, advance
   legoESM's own EXIT state through all 8,760 hourly inputs, and compare each
   registered boundary to NEMO.  This protocol defines accumulated error and
   phenomenology.  It remains an isolated-ice result because bulk fluxes and
   ocean boundary values are supplied by NEMO.

At every continuous EXIT, record normalized L-infinity error for `t_su`, all
three `e_i` and `e_s` layers, category ice/snow thickness `h_i=v_i/a_i` and
`h_s=v_s/a_i`, `a_i`, and `sv_i`.  The denominator is
`max(1,max(abs(NEMO)))`; the fixed classification bar is `1e-15`.  The receipt
will tabulate steps 1, 10, 100, 1000, and 8760 and report maxima without
discarding later debt.  Readers fail on a missing/trailing frame, wrong step,
shape, hash, dtype, or non-finite value.  A truncated-stream plant must exit
nonzero.

## Phenomenology and measurable precision floor

Phenomenology uses each model's continuous hourly EXIT category ice thickness.
Minimum and maximum include all finite hours with positive concentration and
are reported with their UTC timestamp.  End-of-day thickness is kt 24, 48,
..., 8760.  **Melt onset** is the first day after that model's annual maximum
starting seven consecutive negative day-to-day thickness changes.  **Growth
onset** is the first day after its subsequent annual minimum starting seven
consecutive positive changes.  If either run has no qualifying sequence, the
event is UNMEASURED rather than chosen by inspection.

Because no NEMO float32/FCT column spread exists, rerun the same continuous
legoesm column with every floating state/input explicitly float32.  For each
phenomenology row, `distance` is the absolute legoESM-fp64 minus NEMO
difference; `floor` is the absolute legoESM-fp32 minus legoESM-fp64 difference
in the same units.  Classify **AT-FLOOR** when `distance <= floor`, otherwise
**ABOVE-FLOOR**.  A zero distance and zero floor is AT-FLOOR.  This is a
measured legoESM precision floor, not an oracle spread and not a coupled-model
certification.

## Choice register

- ASKED — resolve both named near-bar rows with an order-of-operations replay
  and a fixed 2-ULP discriminator.
- ASKED — run all 8,760 hours against all 70,080 oracle frames and report the
  requested time-slice error table.
- ASKED — report year phenomenology beside the measurable legoESM fp32/fp64
  precision floor with AT-FLOOR/ABOVE-FLOOR classifications.
- ASKED — retain only the ORCA1 identity, CPU execution, copy-only NEMO
  instrumentation, explicit pathspec commits, local-git bundle, and no push.
- UNASKED — coupled bulk-flux certification, alternate ice identities, and any
  NEMO precision/scheme spread.
