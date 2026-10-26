# Lane 3b SI3 thermodynamics Phase 5 scope preregistration

Tracker: `climate-federation/legoESM#1699`

Status: **PREREGISTERED before adding or evaluating the four Round-3 scope
ablations.**  The shared-Thomas deduplication was handled first and separately:
commit `7954ceae00a` proved the shared unnormalised mode bit-identical to the
then-private solver; commit `e7fba49e758` removed the private production copy
and routed SI3 through `legoesm.timestepping.tridiagonal.thomas_solve`.

## Fixed protocol

The oracle root, version-2 operand stream, forcing, one-hour step, 8,760-step
window, CPU backend, fp64 policy, and `1e-15` frame bar are unchanged from
Phase 4.  Every arm below is a private gate hook; no public selector or
Frankenstein identity is constructible.  The enabled production path remains
the ORCA1-resolved `jpl=1`, BL99 3+3/P07, `nn_icesal=2`, ponds/lateral-melt-off
identity.

For each arm, compare every exact-entry boundary/field over 8,760 steps and a
continuous 8,760-step trajectory.  Record first changed row, enabled and
disabled error against NEMO, over-bar count delta, requested growth samples,
and six phenomenology rows.  A plant replacing enabled output with disabled
output must fail the arm's stated discriminator.

## A — ZDF no-snow and melting-surface row ranges

NEMO separately selects snow-present cold/melting ranges and snow-free
cold/melting ranges.  The predicates are `h_s_1d>0` and `t_su_1d<rt0`; the
corresponding `jm_min/jm_max`, surface equation, and first material-layer row
are written at `icethd_zdf_bl99.F90:433-513`.  The reviewed legoESM rewrite is
the `_nemo_branch_ranges=True` path in `_si3_zdf_bl99_step`; its disabled arm
reproduces the earlier all-seven-row/cold-surface solve.

Hypothesis **HA-RANGES-ACTIVE**: the rewrite is faithful and active.  Confirm
only if at least one exact-entry row evaluates a different NEMO predicate,
enabled output improves its first changed POST_ZDF row by at least 100-fold,
and the branch census names the executed NEMO range.  Refute if no row changes
or the disabled arm is closer.  If no row changes, classify it INERT for this
column rather than faithful-by-inspection.

## B — bottom-up basal-melt layer loop

NEMO forms `zq_bot` at `icethd_dh.F90:107-120`, then loops
`jk=nlay_i..1`, skips layers already removed from above, distinguishes internal
from energy-limited basal melt, consumes the available energy, and updates each
layer/remap carrier (`icethd_dh.F90:309-315,366-424`).  The pre-review legoESM
implementation collapsed this to one bottom-layer decrement; the enabled path
transcribes the bottom-up loop.  The existing `_basal_melt=False` arm deletes
the entire physical term and is not an ablation of the rewrite.

Add a distinct private `_nemo_basal_layer_loop=False/True` arm holding the
basal heat input fixed.  Hypothesis **HB-BASAL-LOOP-OWNER**: the rewrite is a
faithful, active fix.  Confirm only if enabled output improves the first changed
POST_DH thickness/enthalpy row by at least 100-fold and the annual arm changes
at least one thickness/onset row.  Report the already-measured total-term
ablation (`_basal_melt=False`, melt onset day 176 versus production/NEMO day
133) separately as sensitivity, not ownership.

## C — snow-ice salinity contribution

For `nn_icesal=2`, NEMO computes snow-ice salinity and updates bulk salinity by
both snow-ice formation and bottom growth at `icethd_dh.F90:441-485,509-519`.
The reviewed addition is the `(zs_sni-s_i)*dh_snowice/h_i` contribution.

Hypothesis **HC-SNOW-ICE-SALINITY**: if the C1D year produces nonzero flooding,
the term is an active faithful fix and its enabled arm must improve the first
changed POST_DH `S_bulk`/POST_SAL `sz_i` row by at least 100-fold.  If flooding
is exactly zero in all exact-entry and continuous steps, the enabled/disabled
arms must be bit-identical at every registered row and it is classified
**INERT IN C1D**, without a broader fidelity claim.

## D — three EOS operation associations

The reviewed changes follow NEMO's written operation order for ice enthalpy
(`icevar.F90:938-946`), ice temperature inversion
(`icethd.F90:233-243`), and snow temperature inversion
(`icethd_dh.F90:498-503`; entry bounds at `icevar.F90:404-416`).  The formulas
are algebraically identical to their predecessors but can move binary64 ULPs.

Add one private `_nemo_eos_order=False/True` arm that changes only these three
associations.  Hypothesis **HD-EOS-REASSOCIATION**: the arm changes only the
roundoff class and the NEMO-written order is no worse at the first changed row.
Confirm as **FLOAT RE-ASSOCIATION** only if an operand-order replay closes the
changed row to at most two ULP and no branch predicate changes.  If all outputs
are bit-identical, classify it INERT IN C1D.  A material or branch-changing
row refutes the hypothesis and remains debt.

## Remaining-debt classification

Partition the 36,852 Phase-4 over-bar exact-entry rows by sub-call, variable,
and step.  Separately count rows whose normalisation denominator is exactly
one because every NEMO value in that row is zero, versus rows with a genuine
oracle magnitude (`denominator>1`); report the largest genuine-relative row.

For each requested continuous growth sample, inspect the exact-entry injection
at that same step.  Classify threshold amplification only if the continuous
error jump coexists with an exact-entry injection in the `1e-15` class; any
material same-step injection prevents that classification.  The terminal
statement must name the first counterexample or state that bit-exact whole-step
summation order is the remaining requirement.

## Choice register

- ASKED — remove the private Thomas implementation after a bit-agreement proof
  and route SI3 through the shared solver.
- ASKED — preregister and independently ablate row ranges, basal loop,
  snow-ice salinity, and three EOS associations.
- ASKED — correct the owner count and basal-melt description in the receipt.
- ASKED — histogram all remaining exact-entry debt and split zero-oracle/
  denominator-one rows from genuine relative rows.
- ASKED — compare exact-entry injection with continuous-error jumps and give an
  honest terminal classification.
- ASKED — CPU/fp64, explicit-path recovery-git commits/bundle, no push, and no
  shipped-NEMO changes.
- UNASKED — new public physics selectors, alternate SI3 identities, changed
  forcing/bar/timestep, coupled bulk-flux certification, GPU/MPI, or deletion
  of any source/run root.
