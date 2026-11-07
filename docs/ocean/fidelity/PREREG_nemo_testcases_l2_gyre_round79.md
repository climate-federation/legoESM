# Preregistration: NEMO-testcases L2 GYRE round 79 absolute barotropic-history pair

Date: 2026-09-13. Frozen after the user answered Decision 37 YES and after
reading the compiled oracle and the existing shared implementation, but before
running a new model measurement or editing the carried-state arithmetic.

## Authorization, magnitude rank, and compiled statement

The authorized change is narrow: replace legoESM's six deviation-form
barotropic histories with NEMO's six absolute histories, only together with the
already-landed upstream prognostic barotropic mean and stage-reconciliation
owner. The round-51 raw-NEMO-history injection remains **REFUTED** and held; it
mixed NEMO histories with legoESM's then-unequal live window boundary and is not
the candidate here.

The compiled GYRE branch initializes the six absolute arrays only for a cold
start, seeds the current external velocity directly from `puu_b/pvv_b`, and
zeros only the external-mode accumulators at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:339-380`.
It consumes the absolute `un_e/ub_e/ubb_e` arrays in the written midpoint
association at the same compiled source's `:481-509`, rotates absolute arrays
after every substep at `:783-795`, and reads/writes those six absolute arrays in
restart I/O at `:991-1018`. The post-window primary external velocity is divided
and committed independently at `:835-883`; the 3-D stage then replaces its
reference-thickness depth mean with that separately prognostic barotropic value
at `GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/stprk3_stg.f90:729-759`.

The current shared legoESM program already carries the separate `uu_b/vv_b`
window boundary and applies the stage-3 reconciliation. Round 78 measured the
resulting GYRE kt=2 substep-1 `un_e` and `ub_e` operands bit-exact; the first
non-bit statement is now `ubb_e`, 6/580 wet U faces with maximum
`8.470329472543003e-22`. This history representation is therefore the first
non-bit statement in source order. It is also the earliest eligible change that
can move the magnitude-ranked whole-step boundary: kt=2 U/V maxima
`2.7478404751243857e-12` / `3.305560306813421e-12`.

## Frozen candidate and restart contract

The one shared implementation will interpret and store `state.bt_hist` as the
absolute tuple `(Ub, Ubb, Vb, Vbb, etab, etabb)`. A continuation passes those
arrays directly to the external loop; a cold start remains `None` and keeps the
compiled two-substep initialization ramp. No configuration, selector,
coefficient, association, stabilizer, timestep, forcing, geometry, mask, or
scoring rule changes.

The pair is mechanical and inseparable: an AB3/AM4 continuation with histories
must also carry both `uu_b` and `vv_b`; otherwise the model must raise a named
error rather than reconstruct a window boundary from 3-D state. The existing
upstream mean/reconciliation arithmetic is not changed. It is gated by (a) the
round-78 exact `un_e` row, (b) direct two-window continuity through the shared
production solver, and (c) the existing stage-reconciliation tests.

The run-restart format will record the new absolute-history semantics. A legacy
format-3 checkpoint carrying deviation-form `bt_hist` may load only by an
explicit conversion using its persisted `uu_b`, `vv_b`, and `eta`, in the exact
old reconstruction order. If any required current field is absent, malformed,
or shape-incompatible, loading must fail with the named phrase
`cannot migrate deviation-form bt_hist`; silent zero fill or silent
reinterpretation is forbidden. New saves must round-trip absolute histories
bit-for-bit.

## Frozen predictions and falsifiers

1. On the admitted round-77 U-midpoint record, GYRE kt=2 substep-1 `ubb_e`
   becomes bit-exact (0/580), and the substep-1 `ua_e` result becomes bit-exact.
   The first live U non-bit statement moves later than substep-1 `ubb_e`; the
   expected next boundary is substep-2 `un_e`, already observed downstream in
   round 78. Any earlier mismatch, any remaining substep-1 U history/result bit,
   or any non-bit shared midpoint replay on NEMO operands falsifies the owner.
2. The ordinary GYRE ladder's kt=1 five state rows remain bit-identical to the
   committed before arm. The first moved whole-step rows are kt=2 U and V; every
   downstream kt=2--10 state row is registered as allowed to move. No row that
   was AT-BAR may leave the bar, and `first_over_bar` may not move earlier. The
   kt=2 U/V maxima must change at the bit level; an unchanged pair falsifies
   whole-step reachability. Direction is not predicted from source reading.
3. The recorded days-1--30 GYRE comparison uses
   `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_owners/lego_seed0_daily`
   as the before arm and the unchanged NEMO member under
   `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_owners`. Every day and
   every scored T/S/U/V/SSH row is registered as allowed to move after day 1;
   no AT-BAR row may leave the bar and the first-over-bar boundary may not move
   earlier. Candidate output must come from the production member driver, never
   a scratch toggle.
4. LOCK_EXCHANGE-zco executes the same absolute-history rotation with one
   external substep. Its kt=1--10 state rows may move only from kt=2 onward;
   no AT-BAR row may leave the bar and first-over-bar may not move earlier.
   OVERFLOW-zps runs the same split-explicit program but its resolved boxcar
   filter reinitializes histories every whole step; its kt=1--10 state rows are
   predicted bit-identical. Any changed OVERFLOW row falsifies value-inertness.
5. The direct restart tests must distinguish the semantics: substituting the
   old final-minus-history values into a new-format checkpoint must fail the
   absolute-value assertion, and removing the format-3 conversion or a required
   migration operand must make the compatibility test fail. Every plant exits
   nonzero.

A prediction that fails is retained as **REFUTED**. A candidate that moves an
unregistered row, makes an AT-BAR row debt, advances first-over-bar, silently
loads an ambiguous checkpoint, or receives a Codex `DO NOT SHIP` verdict does
not land.

## Rule 12 card table and exclusions

| card / lane | frozen disposition |
|---|---|
| GYRE-zco kt=1--10 | MEASURE before and after with the production ladder; kt=1 fixed, kt=2--10 registered downstream, first moved row expected kt=2 U/V |
| GYRE days 1--30 | MEASURE candidate with the production member driver and score against the unchanged NEMO root; compare every row with the recorded before arm |
| LOCK_EXCHANGE-zco | EXECUTES the shared AB3/AM4 absolute-history program; measure kt=1--10 against its records and the committed before JSON |
| OVERFLOW-zps | EXECUTES the shared split-explicit program but reinitializes boxcar histories; measure kt=1--10 and require bit-identical candidate output |
| DINO | SHARED-STATEMENT RISK: its leapfrog card carries its own before/history state and retains 96--98% per-row cancellation risk; no DINO neutrality is inferred from GYRE or band aggregates |
| ORCA2 | UNMEASURED-WITH-SPEC: resolve its compiled external-mode filter and restart branch; record current/before/twice-before U/V/SSH, coefficients, midpoint results, rotations, window-boundary mean, both 3-D reconciliations, and restart names for kt=1--10; replay in source order, preserve every AT-BAR row, and forbid an earlier first-over-bar boundary |

The year harness, card-reconciliation gate, freshwater pair, #1484 guard,
canonical NEMO source/build/run, held manifests, and all scientific
configuration remain untouched. No `makenemo` or `mpirun` is permitted.
