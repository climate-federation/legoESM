# NEMO testcase Lane 4 — ORCA2 Phase-2t preregistration

Date: 2026-09-06

Starting parent: `608b50e4b04b`

Status: **PREREGISTERED BEFORE MEASUREMENT.**  This round admits the combined
Phase-2s oracle twins, resolves the executed RK3 SH2 time-level alias, walks
the non-vacuous kt=2 SH2/TKE entry, and independently repeats the three BBL
gates.  Shared SH2 and TKE arithmetic remain GYRE-owned and will not be
changed in Lane 4.

## P2T-1 — combined oracle admission

Candidate A and B are the user-shell roots
`variant_icebergs_off_phase2s_zdf_een_{a,b}_10step_np2`.  Admission requires:

- success logs, `time.step=10`, and exactly 100 `oracle_*.bin` streams;
- raw identity of all 100 twin streams;
- self-describing ZDF kt=1 and kt=2 headers whose allocation triples derive
  every field extent and payload count, with a field-by-field walk to exact
  EOF; the pre-existing stream schemas must likewise walk to exact EOF;
- raw identity of the 94 unchanged Phase-2q streams, a deliberate format-only
  difference for the kt=1 SH2 stream, and raw identity of the four restored EEN
  streams with the Phase-2m witness;
- exact ordinary restart/history identity against the accepted uninstrumented
  10-step variant control; and
- binding magic/extent/count/truncation/trailing/canonical-zero/non-vacuity and
  ordinary-output plants, each exiting nonzero through the production
  validator.

Only if all conditions pass is twin A pinned as the single
`VARIANT_ORACLE_V2` root and twin B retained as its independent witness.
Phase-2m/2n/2p/2q roots remain retained witnesses and are flagged superseded
as primary roots.

## P2T-2 — executed SH2 identity and kt=2 score

The source expression in `zdfsh2.F90:80-89` formally multiplies a Kmm
velocity difference by a Kbb velocity difference, and divides by the matching
Kmm and Kbb face thicknesses.  The executed RK3 caller, however, invokes
`zdf_phy(kstp,Nbb,Nbb,Nrhs)` at `stprk3.F90:163-165`; `zdfphy.F90:264-269`
passes those aliased values to `zdf_sh2`.  Therefore this ORCA2 executable
uses Nbb×Nbb at whole-step entry, not distinct Kmm/Kbb values.  The apparent
conflict is preregistered as formal-argument semantics versus the executed
call-site alias, and the receipt will preserve both citations.

The admitted kt=2 frame must be non-vacuous.  Under production JIT, CPU,
binary64, and scalar-libm policy, the gate will score on owned wet rank-zero
cells:

1. the ORCA2 production card's restored face-native Nbb×Nbb selector; and
2. the shared source-literal face-native routine called directly with the
   record's Nbb operands.

Both are scored cellwise with row-scale ULP and one-bit plants.  The first
non-bit intermediate/statement and its fold/partial-bottom/coast/interior class
are reported.  A shared departure is registered as
`GYRE_OWNER_SHARED_SH2`; an ORCA2-only input departure remains Lane 4.  No
shared SH2 arithmetic is changed here.

## P2T-3 — bottom TKE and ordered TKE walk

The NEMO deck resolves `ln_drg_OFF=.false.` through the reference namelist;
therefore `zdftke.F90:279-288` executes the bottom-friction Dirichlet
assignment at `mbkt+1`, using Kbb face velocities, masks, `rCdU_bot`, and
`MAX(zebot,rn_emin)`.  `zdfiwm` forces `rn_emin=1e-10` at
`zdftke.F90:841-844`.  The `nn_bc_bot=1` namelist value is read but unused by
the TKE routine outside the documented wave-coupling setting; it does not
guard this friction boundary assignment.

The instantiated ORCA2 card currently has `bottom_tke_bc=False`, while the
shared implementation already accepts the exact per-column bottom index and
Dirichlet operand.  Unless source/production inspection refutes that wiring,
this is preregistered as a Lane-4 card-selector defect and will be restored
`False -> True` without changing defaults or shared arithmetic.  Admission
requires the source-literal bottom operand to score `0 / 8,794` against a
target derived only from recorded NEMO operands, plus a binding one-bit plant.
If production cannot consume that existing arm, the result is instead a
shared TKE gap handed to GYRE, not silently emulated.

After the selector boundary, the ordered TKE walk is surface input, bottom
input, ice attenuation/Langmuir, shear and buoyancy production, dissipation,
tridiagonal assembly/solve, `nn_mxl=3`, and avm/avt assembly.  It stops at the
first non-bit statement.  Existing kt=2 frames do not contain post-solve TKE
or mixing-length outputs; any such boundary is reported
`UNMEASURED_NEEDS_WRITE_ONLY_FRAME` rather than inferred.  Shared arithmetic
routes to GYRE; ORCA2 forcing operands (`taum`, `fr_i`, `rCdU_bot`) remain
Lane 4.

## P2T-4 — BBL independent repeat

The diffusive-BBL, advective-BBL, and cross-card gates are run as three
independent commands with their respective plants.  Expected decisive rows
remain `0 / 8,489`, `0 / 8,554`, and `0 / 231,519`; any movement stops the
round.  This is a second execution record, not a new implementation.

## Controls and stop conditions

- Production JIT, CPU, fp64, and the explicit scalar-libm policy are required.
- Every numerical score has a same-path plant that exits nonzero.
- Claims use CONFIRMED/PLAUSIBLE and unequal / n; Rule 8/11/12 tables remain
  explicit.
- No MPI/NEMO process runs in the sandbox.  No shipped NEMO file, SI3 shared
  operator, or shared SH2/TKE arithmetic is edited.
- Any malformed stream, ordinary-output mismatch, nonbinding plant, or first
  shared over-bar boundary stops the walk and is handed to its owner.

## ASKED / UNASKED at preregistration

| action | classification | disposition |
|---|---|---|
| admit and pin Phase-2s twins | ASKED | gated as P2T-1; no pin before PASS |
| re-verify SH2 Kmm/Kbb levels | ASKED | source expression plus actual Nbb/Nbb call alias both cited |
| restore `bottom_tke_bc` if false is wrong | ASKED identity restoration | preregistered `False -> True`, conditional on production wiring audit |
| walk TKE to first non-bit | ASKED | ordered, fail-closed, no proxy targets |
| repeat three BBL gates/plants | ASKED | independent executions |
| keep BBL defaults 0 / 0.0 | ASKED Decision 10 | resolved; unchanged |
| change shared SH2/TKE arithmetic | forbidden in Lane 4 | not planned |
| execute MPI/NEMO, edit shipped NEMO, delete, push | forbidden | not planned |
