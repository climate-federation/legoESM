# NEMO testcase Lane 4 — ORCA2 Phase-2i preregistration

Date: 2026-09-06

Parent: `c8be7befc8c730f80ab72a952a7c8f9104e6203a`

Decision owner: user Phase-2i resume (ASKED).  This document freezes the
replacement acquisition and the ORCA2-owned RGB boundary before either writer
or numerical implementation is changed.

## 1. RHS reproducibility diagnosis and frozen acceptance

The completed systematic twins are
`variant_icebergs_off_phase2h_systematic_a_10step_np2` and
`variant_icebergs_off_phase2h_systematic_b_10step_np2`.  Both reached step 10
with 92 records.  The existing raw-byte gate reports 91 / 92 exact; the sole
difference is `oracle_rhs_kt00000001.bin` (7,086,892 bytes):

| arm | SHA-256 |
|---|---|
| A | `18a961621997f0f52d1293fec6aaeeb4424b24fb79cb469c6d75c73bfdfd6e5b` |
| B | `006fca37d1a3e337f7be3ce47a528e423266d03c2debf9baa915fb0ac47c4979` |

Both headers are valid: magic `NEMO_L1_RHS___1` and
`(version,kt,Krhs,jpi,jpj,jpk,bits)=(1,1,3,94,152,31,64)`.  The payload is two
full-domain rank-3 arrays, `uu(:,:,:,Krhs)` and `vv(:,:,:,Krhs)`.  The writer
already routes both arrays through the zero-first `l4_canon_3d` WRITE-only
view (`cfgs/ORCA2_OMIP_L4/MY_SRC/stprk3.F90:387-400`), so another mask does not
address the observation.  Unlike the adjacent stage and ZDF writers, however,
`l1_dump_rhs` has no `.NOT.lwp` return.  Both MPI ranks can therefore open the
same `STATUS='REPLACE'` pathname and concurrently write different local
subdomains.  The changing payload is a concurrent-write artifact, not evidence
that the stage-1 RHS used by physics is undefined.

The frozen repair is WRITE-only: retain the existing zero-first canonical
temporaries and add the standard rank-zero visibility guard before `OPEN`.
No model array, arithmetic expression, call position, header, or payload order
may change.  Rebuild with `conda-scalarmath`, require zero dynamic `_ZGV*`
symbols, and prepare two independent 10-step, two-rank (`jpni=2,jpnj=1`) run
directories.  VARIANT V2 is pinned only if each arm passes all schemas,
ordinary-output identity and plants and the twin gate reports **92 / 92 raw
byte-identical records**.  Earlier candidates remain preserved and flagged.

## 2. Frozen source-literal RGB identity

The accepted post-bulk operand substitution supplies the oracle's own
post-SI3 `qsr`.  The current shared `nemo_qsr_rgb` path is over bar at
149,647 / 233,341 rank-0 interior wet T cells (maximum absolute difference
`6.341827702190138e-06 K s-1`).  Lane 4 owns the RGB branch; the generic
`rgb_chl` behavior is not the oracle identity and must remain available.

The shared shortwave module will implement one source-named
`nemo_qsr_rgb` identity, with no ORCA2 conditional.  Its ordered alignment to
`qsr_RGBc` in `MY_SRC/traqsr.F90:255-496` is frozen as follows:

1. use the `nn_chldta=1` chlorophyll already read/interpolated by `fld_read`
   before the dump (`traqsr.F90:298-302`);
2. clamp surface chlorophyll, evaluate the `nn_chlprfl=1` Morel-Berthon
   coefficient statements and the resolved live `gdepw(jk+1,Kmm)` profile;
3. form `NINT(41 + 20*LOG10(zchl) + 1.e-15)` and index the source's 61-row
   blue/green/red coefficient table;
4. form `zz0=rn_abs`, `zz1=(1-rn_abs)/3`, then the four surface fluxes;
5. walk levels in the resolved extinction ranges `nk0=2`, `nkR=8`, `nkG=19`,
   `nkB=22`, evaluating the per-band `EXP` calls through the active scalar-libm
   precision policy and preserving the source addition order;
6. preserve the `zc0/zc1/zc2`-style previous/current flux recurrence, the
   `wmask(jk+1)` bottom no-flux behavior, and the exact
   `r1_rho0_rcp*(zeT-zzeT)/ze3t` increment.

Every arithmetic source statement is materialized with `nemo_source_round`;
the transcendental inputs and results are materialized at their source
boundaries.  Explicit live `e3t`, `gdepw`, and W-mask operands may be added to
the shared API because they are NEMO operands, not ORCA2 switches.  Production
JIT, CPU, fp64 and scalar-libm remain mandatory.

The target is bit exact: 0 / 233,341 source-defined rank-0 interior wet cells.
A one-variable one-ULP mutation of the oracle tendency must travel through the
same validator and exit nonzero.  If exactness is reached, continue in NEMO
execution order to EOS/HPG, external mode/north fold, stage transports, FCT,
BBL, and TKE/EVD/IWM entry.  Stop at the first over-bar boundary, name its
owner, and make no repair to a shared GYRE-owned numerical operator.  Bulk and
SI3 outputs remain visibly `ORACLE_SUPPLIED`; SI3 remains
`UNMEASURED_PENDING_ICE_MERGE`.

## 3. ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| diagnose and canonicalize RHS record | ASKED | existing zero-first views plus missing rank-zero writer guard |
| rebuild and prepare replacement twins | ASKED | exact binary/deck, two independent directories; user executes MPI |
| pin VARIANT V2 at 91 / 92 | UNASKED and forbidden | require 92 / 92 raw identity |
| make shared RGB source-literal | ASKED | shared source selector, no ORCA2 arm |
| retain generic RGB behavior | ASKED by one-implementation rule | no card-specific fork or duplicated module |
| continue ocean ladder | ASKED | stop at first debt needing a decision |
| repair shared-operator debt | UNASKED and forbidden | register boundary and hand to GYRE owner |
| execute MPI | UNASKED and prohibited | prepared launchers are executed by user shell |
| delete prior evidence | UNASKED and forbidden | preserve and provenance-label every earlier run |
