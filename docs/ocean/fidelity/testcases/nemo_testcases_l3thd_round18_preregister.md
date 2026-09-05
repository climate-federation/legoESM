# SI3 lane 3b round 18 preregistration — cross cards and NCAR bulk

Date: 2026-09-05

Tracker: `climate-federation/legoESM#1699`

Parent: `c7ca3866725b8e1856d18663d3abe8dbb8406870`

State: **PREREGISTERED; ROUND-18 MEASUREMENTS UNRUN**

## A. Cross-card semantics and frozen comparisons

NEMO 5.0.2 defines `ln_dynadv_OFF` as linear dynamics with no momentum
advection (`dynadv.F90:35,43`).  `dyn_adv` has executable cases only for vector,
flux-C2, and flux-UP3 (`:78-90`); initialization selects exactly one complete
program, with `ln_dynadv_OFF` mapping to `np_LIN_dyn` and
`ln_dynadv_up3` mapping to `np_FLX_up3` (`:128-134`).  Therefore the strict
legoESM validation that pairs horizontal OFF with vertical OFF is source
correct.  The shared LOCK_EXCHANGE/OVERFLOW card is the defect: it selects the
complete flux-UP3 program horizontally but inherited vertical OFF.  The
one-variable construction repair is to select the existing shared
`vertical_momentum_scheme="nemo_up3"` on those two cards.  The coupled C1D slab
will continue to select the complete OFF/OFF program explicitly; no validator
is weakened and no numerical operator is added.

After that construction repair, the frozen comparisons are:

1. `nemo_testcase_phase3_stage_sweep_gate.py` for LOCK_EXCHANGE-zco and
   OVERFLOW-zps against the pinned lane-1 kt1 stage frames;
2. `nemo_testcase_phase3_trajectory_gate.py` for both cards through kt=10;
3. the GYRE production-JIT CPU/fp64/scalar-libm kt1..10 register against its
   pinned lane-2 Oracle V2.

Every comparison is cellwise on its registered wet mask.  The reported pair is
the fixed campaign `1e-15` oracle-relative maximum and row-scale ULP distance,
with bit counts `unequal / n`.  A one-ULP live-cell plant must become the first
red row and exit nonzero.  Measurements will be retained outside git and only
their hashes and summary rows committed.

### Rule 8/12 predictions

The three shared changes under audit are the round-15 pre-stage-3 ZDF input
ordering, the round-16 `nemo_literal` QCO layer-thickness path, and the round-17
Kmm barotropic seed plus quadratic drag.  The first two are predicted not to
move the already registered LOCK/OVERFLOW rows except where their selected
paths execute.  The Kmm seed is predicted to move GYRE's kt=2 velocity toward
NEMO because GYRE has live non-rest Kmm flow, while the slab exposed that the
post-stage workspace is the wrong NEMO time level.  This direction is a
prediction, not a result.  Any worsened row is listed individually and remains
registered to the GYRE lane as owner-of-record for these shared operators.

## B. NCAR O1 bulk certification

### Frozen oracle and schema

The handoff is ORCA2 commit `c19944cea7e`, document
`nemo_testcases_l4_orca2_phase2g_handoff_receipt.md`.  The promoted scalar-math
canonical twins are retained at
`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2g_o1canon_{a,b}_10step_np2/`.
Their O1 records are byte-identical, SHA-256
`751b2d9181778e81f01bc5872d47100afc0fc3ad02c4ffcccc9613a0af2ad045`.
Frame 0 contains nine `fld_read` CORE inputs; frame 1 contains twenty slots on
the rank-0 90 by 148 interior.  `cd_du` and `qlwn` are source-unowned in the
provisional writer and zeroed only by the canonical WRITE-only writer; neither
is certified as a computed NCAR output.  The other eighteen frame-1 fields are
the coverage universe, with bulk scores restricted to the 8,794 wet T cells.

The pre-implementation search found one existing shared home,
`core/bulk_flux.py`: `validate_bulk_scheme`, the `large_yeager` neutral
coefficient helper, q-saturation helpers, MOST machinery, and the already
certified `nemo_si3_constant` selection.  It also found the coupler dispatch
that calls this core implementation.  Round 18 will extend that implementation
with a selector named for NEMO's resolved NCAR option; it will not create a
second bulk module or fork an ORCA2 card.

### Ranked literal-source walk

The old ORCA2 diagnostic, which is not a round-18 result, placed the first bulk
debt at `theta_air` (117/8,794 unequal), then `ssq` (3,964/8,794), with stress,
sensible heat, latent heat, and evaporation non-bit in all wet cells.  The
ranked source-literal walk is:

1. air-temperature height correction and potential-temperature statement;
2. water saturation specific humidity, including each scalar-libm EXP/LOG call;
3. Large & Yeager neutral 10 m drag/Stanton/Dalton coefficients;
4. the fixed NCAR stability iterations, their convergence/update order, and
   measurement-height corrections;
5. stress, sensible heat, latent heat, and evaporation assembly statements.

Each Fortran source statement will be mirrored in its written association and
rounded once with the one shared `nemo_source_round`.  Scalar libm is used only
where NEMO calls the scalar library.  Constants come from `legoesm.constants`;
compile-time folded values, if present, are pinned by exact bits with their NEMO
line.  The preregistered target is **0 / n non-bit values for every one of the
eighteen source-defined outputs**.  If a row remains non-bit, the gate stops at
the first operand in NEMO execution order and labels it without a downstream
claim.

One-variable private replay hooks replace one statement result at a time with
the oracle operand.  Ownership requires scale agreement and clearing the child
row.  A one-ULP plant is applied independently to every covered output; every
plant must exit nonzero at that row.  Mapping remains oracle-supplied and is not
re-certified here.

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| repair the genuine flux-UP3 card composition and run three cross-card gates | ASKED | preregistered |
| audit all three shared changes under Rule 8/12 | ASKED | preregistered |
| certify NCAR O1 using the ORCA2 handoff record | ASKED | preregistered |
| one shared NEMO-named bulk option, source rounding, libm, plants | ASKED | preregistered |
| alter the strict OFF/OFF validator | UNASKED and source-refuted | forbidden |
| certify `cd_du`, `qlwn`, mapping, RGB, SI3, or downstream ocean operators | UNASKED | outside this boundary |
| delete retained roots, modify shipped NEMO, add runtime output to git, GPU, push | UNASKED | forbidden |

## CONFIRMED / PLAUSIBLE

**CONFIRMED before measurement:** NEMO's complete OFF and flux-UP3 dispatch
semantics; the shared-card construction contradiction; the stable canonical O1
record identity and schema; and the location of the existing shared bulk
machinery.  **PLAUSIBLE / UNMEASURED:** every predicted cross-card movement,
every ranked NCAR arithmetic owner, and the 0/n target.  No round-18 numerical
result is claimed here.
