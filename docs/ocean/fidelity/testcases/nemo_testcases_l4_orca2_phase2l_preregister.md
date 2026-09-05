# NEMO testcase Lane 4 — ORCA2 Phase-2l preregistration

Date: 2026-09-06

Parent: `3e425e24ded84e9bd1cbba7af492aaec43e99170`

Status: **PREREGISTERED BEFORE MEASUREMENT.**  This round adjudicates the
Phase-2k stage-1 WZV clock claim on the compiled production step, measures the
cross-card runoff non-regression required by Rule 12, pins the resolved
vector/C2 momentum selection, and then resumes the ocean ladder.  It does not
modify shared physics and does not enter SI3.

## P2L-A — production stage-1 tracer `ww`

The frozen target is
`variant_icebergs_off_phase2j_wzv_a_10step_np2/oracle_stage1_wzv_operands_kt00000001.bin`
(SHA-256 `245be2ea348002b93358e5dd723f0b84198af47c3d38f0bc0bb1acb0fc3100fe`).
The comparison scores raw binary64 equality on the record's live rank-zero T
cells for which every horizontal input needed by the full-global card is
present.  The MPI partition edge and tripolar fold support row are excluded
and counted explicitly; this is a strict subset of the already registered
rank-zero interior, not a changed tolerance.

The candidate must come from `LatLonCGridOceanModel.step` under production
JIT, CPU, fp64 and scalar-libm.  A private WRITE-only hook may substitute the
already registered `ORACLE_SUPPLIED_EXTERNAL_MODE` endpoint (final SSH and
time-mean U/V transports) after the shared external solver, then expose the
actual `_g0[2]` array that the production tracer stage receives after the
ordinary compiled step completes.  The hook may not change a public card or
any arithmetic between the external endpoint and `tra_adv`.

The ordinary ORCA2 card is presently stopped by the shared geometric-depth
EOS allow-list even though the existing EOS-80 operand gate is exact.  This
round will not widen that shared operator guard.  The same private WZV hook may
therefore bypass only this constructor guard: EOS still executes, but its
result cannot reach the scored value after the external endpoint substitution.
The public card remains fail-closed and the guard is registered to its shared
owner.

The next constructor boundary is likewise shared and downstream-owned:
EOS-80 TKE/EVD `bn2` is not implemented.  If it prevents the compiled step
from reaching `_g0`, the diagnostic may substitute the already constructible
TEOS-10 `bn2` only in the upstream tendency whose external result is then
replaced.  No proxy result may enter `_g0`; the receipt must keep the EOS-80
TKE/EVD entry `UNMEASURED` and name its owner.

The same rule applies if the unimplemented ORCA2 tripolar `nemo_avg4` EEN
momentum fold stops the upstream tendency: a constructible EEN thickness proxy
may be used only before the substituted external endpoint.  The public card
selection remains `nemo_avg4`; the fold stays `UNMEASURED` and no proxy value
may enter `_g0`.

The selected spatial `nemo_div_curl` viscosity is also not implemented on a
tripolar grid.  If reached before the same substitution, its coefficient may
be zeroed in the discarded diagnostic tendency only.  This remains an
`UNMEASURED` shared/ORCA2-grid boundary; the public card is unchanged.

The discriminator is frozen:

- **AT_BAR, 0 / n:** retract `GYRE_OWNER_SHARED_WZV_STAGE_CLOCK` under Rule
  11.  The Phase-2k ablation mixed NEMO's one-third-stage `r3t(Kaa)-r3t(Kbb)`
  with the full 10,800 s denominator; production instead uses its full-step
  SSH delta with 10,800 s, which represents the same rate up to the source
  interpolation/rounding.
- **DEBT near `7e-5 m s-1`:** retain the shared-clock handoff and report the
  exact production path; GYRE's prior T/S result cannot certify `ww`, because
  no GYRE gate scored vertical velocity and vector
  `stprk3_stg.F90:284-290` skips momentum `ww` at stage 1.
- Any other over-bar result is not assigned to the clock without a
  one-variable decomposition; it is registered at its first causal boundary.

A one-ULP plant changes one scored production `ww` cell and must exit nonzero
through the same scorer.

## P2L-B — resolved momentum program

The ORCA2 card must resolve `momentum_advection="vector_invariant"` and
`ke_gradient_scheme="c2"`, the legoESM spellings of
`ln_dynadv_vec=.true.` and `nn_dynkeg=0`.  This is checked against the variant
namelist and `ocean.output`, not inferred from the card name.  A selector plant
changes only `ke_gradient_scheme` and must be rejected by card validation.

## P2L-C — Rule-12 runoff cross-card check

The shared runoff refactor must leave the GYRE, LOCK_EXCHANGE and OVERFLOW
kt=1 gates at 0 ULP.  Those cards pass `runoff_mass_flux=None`, so the new
branch is skipped and every existing optimization barrier must remain in the
same order.  The exact committed kt=1 gate commands and their actual counts
will be recorded.  A nonzero result is a regression and stops this lane; a
gate that cannot be run is `UNMEASURED` with its concrete prerequisite.

## P2L-D — downstream ocean ladder

Only after P2L-A through C are dispositioned, continue in NEMO execution order:

1. stage tracer advection (`traadv.F90`/`traadv_fct.F90`) with the recorded
   stage transport, and oracle-supplied stage `ww` only if P2L-A retains a
   real blocker;
2. BBL (`trabbl.F90`) on the already certified census-round geometry;
3. TKE entry (`zdftke.F90`), EVD entry (`zdfevd.F90`) and IWM entry
   (`zdfiwm.F90`).

Every boundary is cellwise with an explicit `n`, one-variable arms and a
binding plant.  ORCA2-specific north-fold, runoff, BBL, geothermal and IWM
debt may be repaired here only in the shared NEMO identity.  FCT internals,
TKE, ZDF and other shared debt is registered with its first boundary and
`GYRE_OWNER_*`; it is not repaired in Lane 4.  SI3 exchanges stay
`ORACLE_SUPPLIED` and SI3 operators stay `UNMEASURED_PENDING_ICE_MERGE`.

## ASKED / UNASKED at preregistration

| item | status | disposition |
|---|---|---|
| compiled production `ww` extraction and WZV rescore | ASKED | P2L-A |
| external-mode operand substitution | ASKED standing pattern | explicit, does not certify external mode |
| cross-card runoff measurement | ASKED | P2L-C |
| vector/C2 selector pin | ASKED | P2L-B |
| repair shared external mode, FCT, TKE or ZDF | UNASKED and forbidden | owner handoff |
| NEMO/MPI run | UNASKED | none is started in the sandbox |
| SI3 execution | UNASKED pending merge | remains oracle-supplied |
