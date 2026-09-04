# NEMO testcase lane 2 GYRE — Phase 3 round-8 boundary receipt

**Verdict: DEBT — bottom drag, every external-mode frame, stage-2 momentum,
and the complete explicit stage-3 tracer accumulator are AT-BAR.  The first
remaining observed transition is the implicit ZDF solve; its internal matrix
and solution operands were not dumped, so the ordered register is exhausted
there as UNMEASURED.  Whole-step kt=2 does not certify.**

Date: 2026-09-03  
Implementation commit: `0da38492ea930789d0e7209897fff48bd434c19c`  
Starting reconciled tip: `57429ecf5f377ce2bf220f36bc05313cd29e0dfd`  
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`

Before measurement I reread the GYRE reconciliation receipt, the requested
lane-1 stage/SSH/phantom/census/face-thickness receipts, and the branch-
isomorphism map.  The unmerged round-7 WIP was not trusted: every live result
below was rederived from NEMO 5.0.2 source and a newly built config-local
instrument.  The run remained CPU-only, fp64, and used the lane-1 binary
record format, central time-level registry, immutable pointwise `1e-15` bar,
one-variable arms, planted controls, and shared log-log growth instrument.

## Ordered walk

### Bottom drag closes the external-mode register

GYRE resolves `nn_drg=np_non_lin`, `ln_drgimp=.true.`, and `rn_Cd0=1.0e-3`
(`rn_Cdmax=0.1`, `rn_ke0=2.5e-3`, `rn_z0=3.0e-3`).  NEMO constructs one Kmm
quadratic coefficient in `zdfdrg.F90:138-190`, freezes that coefficient and
the Kmm baroclinic residual in `dynspg_ts.F90:1584-1643`, applies drag to the
entry external velocity inside every substep at `dynspg_ts.F90:699-705`, and
uses the same coefficient in the bottom-cell implicit diagonal at
`dynzdf.F90:148-160,293-305`.

The canonical DINO bottom-drag implementation is now consumed as one
inseparable WS-RK3 identity; no GYRE-only operator was added.  At substep 2
the direct drag operands agree to `3.79e-29` absolute and combined
`trd_u/trd_v` agree to `2.07e-25`.  Omitting only in-substep drag produces
`6.73e-14` and `6.76e-14`, the exact residual scale, so the boundary label is
**CONFIRMED_CAUSAL_OWNER_AT_SUBSTEP2**.  All 800 external-mode rows (50
substeps × 16 boundaries) are AT-BAR.

For continuity with the accepted review redirect: before the literal ENE
operand landing, substep-2 `trd_u` was `8.60e-12`, which is **1.59%** of the
`5.41e-10` oracle magnitude—not tiny in relative terms.  Literal ENE reduced
it 128× to `6.73e-14` (`0.0124%`) and bottom drag then reduced the combined row
to `2.07e-25`.  This scaling sequence precedes every owner label.

The freshwater discriminator carried from the reviewed reversal remains
source- and runtime-backed.  NEMO's first SSH increment selects the
`8.50091456831154e-6` owned-domain EMP mean; the naive 704-cell value
`9.166209883944753e-6` would cause `1.86749562283012e-7` error.
`lib_fortran_generic.h90:92,144-148` multiplies each 2-D reduction operand by
`smask0_i`, and `dommsk.F90:200-205` constructs that unique interior mask.
Thus the masked numerator is faithful even though `usrdef_sbc.F90:122-140`
passes an unmasked array expression to `glob_2Dsum`.

### Stage-2 momentum closes to the bar

The WRITE-only HPG operands identify NEMO's live stage state.  The production
path now preserves `eosbn2.F90:265-288`'s `prd=zn*r1_rho0-1` association and
the `dynhpg.F90:340-390` `hpg_sco` recurrence.  Projected HPG errors are
`8.08e-28` (u) and `3.23e-27` (v); vorticity is `1.32e-23` and advection is
`1.28e-26`/`4.59e-26`.  The latter two terms are themselves near-null at this
state, so their AT-BAR absolute results are consistency evidence, not strong
exoneration.

The completed stage-2 Kaa state is AT-BAR at `2.71e-19` u and `3.25e-19` v.
The legacy pre-projection arm moves `8.13e-20`; both sides remain AT-BAR, so it
has no remaining owner claim.  The shared raw-`e3w_0` resolver preserves the
GYRE oracle mesh and its validated midpoint fallback keeps all lane-1 WS-RK3
cards operational; the no-scheme-duplication tripwire covers this single
implementation.

### Stage-3 transport and WZV

NEMO forms `zFu/zFv` from the retained Kmm stage velocity and barotropic mean
at `stprk3_stg.F90:257-278`, then passes those already materialized arrays to
`wzv(...,np_transport)` at `traadv.F90:220-226`.  Production now reuses that
same pair rather than recomputing an algebraically equivalent transport.

| boundary | normalized max | disposition |
|---|---:|---|
| `zFu` | `8.736385768894321e-16` | AT-BAR |
| `zFv` | `1.0966867098887257e-15` | DEBT; first transport row |
| `zFw` / shared-transport production | `2.3530283179984255e-13` | DEBT |
| `zFw` / legacy rederived transport | `2.4980249777374853e-13` | DEBT |

The private stored-`zvb` association arm worsens `zFv` to
`1.2388498019113382e-15`; it is **NOT_SOLE_OWNER** and production retains the
reduction-based value.  Sharing the materialized transport moves
`1.0741014697174985e-13` and improves `zFw`, but does not clear it, so that
landing is **CONFIRMED_CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER**.

A new config-local WRITE-only `MY_SRC/traadv.F90` records `ww` after WZV,
after the adaptive partition, and final `pFw`.  GYRE resolves
`ln_zad_Aimp=.false.`, so `wi` is a zero-extent NEMO array and the parser
records that logical boundary as exact zero.  The WZV velocity differs by
only `8.44e-21` absolute—AT-BAR—but that is `2.3487e-13` relative to the
`3.59e-8` oracle magnitude; multiplying by cell area yields the visible
`pFw=2.3488e-13` normalized residual.

Scaling-first arms assign no false sole owner:

- replacing only `pFu/pFv` moves `2.3045e-13` and leaves `1.0118e-14`;
- replacing only the `Kbb/Kmm/Kaa` SSH operand triplet moves `1.0127e-14`
  and leaves `2.3038e-13`;
- the planted SSH-arm control writes exactly `1.0` into a wet `ww` cell and
  fails closed.

The horizontal transport is the dominant upstream contributor; SSH is a
smaller contributor.  The literal WZV recurrence itself is not assigned an
owner label beyond those measured operands.

### First unmeasured boundary: implicit ZDF

The source-ordered explicit stage-3 accumulator now clears completely:
immediately before `tra_zdf`, T is `6.053921299982747e-16` and S is
`5.786348021897527e-16`, both AT-BAR.  Immediately after the production
literal solve, T is `1.3614736849003888e-12` and S is
`2.198822215627748e-14`; these are exactly the whole-step kt=2 tracer rows.
The transition therefore occurs inside the `tra_zdf` matrix construction or
ordered solve.  The analogous `dyn_zdf` internal boundary remains unavailable
for the stage-3 momentum jump.  No matrix/solution oracle record exists, so
both are honestly **UNMEASURED** and the ordered register is exhausted here.

## Re-pinned kt=1 and kt=2…10 sweep

At kt=1 T/S are bit-exact.  At-rest u/v/SSH are still **UNINFORMATIVE** because
their oracle and candidate values are identically zero.  The first whole-step
DEBT remains kt=2:

| field | kt=2 normalized max | status |
|---|---:|---|
| T | `1.3614736849003888e-12` | DEBT |
| S | `2.198822215627748e-14` | DEBT |
| u | `9.484089938412545e-7` | DEBT |
| v | `8.987992610401006e-7` | DEBT |
| SSH | `5.204170427930421e-18` | AT-BAR |

Against the resumed `57429d7` baseline sweep, 29 of 50 whole-step rows
improve, five are identical, and 16 worsen.  The kt=2 T residual improves
about 142× (`1.93e-10` to `1.36e-12`); the momentum debt is essentially
unchanged.  At kt=10 the normalized maxima are T `5.636638587568597e-3`, S
`1.5483121735284942e-4`, u `5.624987258866925e-2`, v
`1.1173366025086486e-2`, and SSH `2.1255475813273736e-4`.

The lane-1 log-log instrument labels the short tail for T and u
POLYNOMIAL_FIT_PREFERRED (exponents `0.60445` and `0.95631`) and SSH
BOUNDED_OR_DECAYING_NO_AMPLIFYING_MODE.  These are characterization labels,
not acceptance evidence.

## Provenance, controls, and review

The oracle executable was rebuilt only from config-local `MY_SRC`; no shipped
NEMO source was modified.  Stage-3 state (`3703a9f…`), kt=2 entry
(`887f3bb…`), and restart (`3271da1…`) reproduce the registered hashes.
The post-transport record's owned payload is bit-identical to the prior run;
its whole-file hash is intentionally not used as a control because NEMO leaves
unowned/halo bytes uninitialized.  The WZV record, executable, MY_SRC sources,
all gates, and implementation Git SHA are pinned in
`nemo_testcases_l2_gyre_phase3_round8_artifacts.sha256`.

Normal JIT and a constrained LLVM retry both exhausted host memory.  The
authoritative reruns therefore set `JAX_DISABLE_JIT=1` while retaining CPU,
JAX x64, and the explicit fp64 policy; no arithmetic selector or acceptance
bar changed.  Verification is 79 focused gate/registry tests plus 25 lane-1
WS-RK3 tests, all passing; an additional 148-test broad run first exposed the
shared-`e3w_0` regression, after which the complete failed WS set passed.

This is a Codex-internal measurement round.  Independent Claude and GLM
review of these new findings remains outstanding; no dual-review claim is
made.  Per the accepted process note, the earlier EMP reversal returned
through review with both source and runtime evidence before it landed.
