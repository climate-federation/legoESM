# NEMO testcase lane 2 GYRE — Phase 3 round-8 boundary receipt

**Verdict: DEBT — round-9 production-JIT recertification retracts round 8's
eager-only stage-2 and explicit-stage-3 AT-BAR claims.  Bottom drag and all
800 external-mode frames remain AT-BAR under JIT, but stage-2 momentum is
already DEBT (`4.67e-13` u, `5.19e-13` v), so the first production-regime
owner remains open.  Whole-step kt=2 does not certify.**

Date: 2026-09-03  
Implementation commit: `0da38492ea930789d0e7209897fff48bd434c19c`  
Starting reconciled tip: `57429ecf5f377ce2bf220f36bc05313cd29e0dfd`  
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`

Before measurement I reread the GYRE reconciliation receipt, the requested
lane-1 stage/SSH/phantom/census/face-thickness receipts, and the branch-
isomorphism map.  The unmerged round-7 WIP was not trusted: every live result
below was rederived from NEMO 5.0.2 source and a newly built config-local
instrument.  The run remained CPU-only, production-JIT fp64, and used the lane-1 binary
record format, central time-level registry, immutable pointwise `1e-15` bar,
one-variable arms, planted controls, and shared log-log growth instrument.

## Round-8 card-choice disclosure and one-variable audit

Round 9 searched the existing DINO/lane-1 option surface before changing the
card.  All six round-8 additions already existed as canonical shared options;
none is a GYRE implementation.  Each is selected by the resolved GYRE program,
so no flip is reverted.  The TKE/EVD selectors share an Nbb input but retain
independent consumers; `stprk3.F90:154-165` makes the RK3 whole-step entry Nbb
for both.

| choice | resolved namelist and selecting source | one-variable JIT move at kt=2 (T/S/u/v/SSH) | worst JIT move at kt=10 | round-8 / round-9 disposition |
|---|---|---|---:|---|
| `tke_n2_time_level=nemo_before` | `EXP00/namelist_cfg:204,215-217`; `stprk3.F90:154-165` | `0/0/0/0/0` | `0` | UNASKED addition; ASKED audit, retained (RK3-equivalent timing) |
| `evd_n2_time_level=nemo_now_before` | `EXP00/namelist_cfg:205-207`; `stprk3.F90:154-165` | `1.77e-3/9.34e-5/2.53e-2/2.53e-2/0` | `8.26e-3` | UNASKED addition; ASKED audit, retained |
| `een_e3f_scheme=nemo_avg4` | `EXPREF/namelist_ref:1072-1073`; `dynvor.F90:918-950` | `1.51e-16/1.93e-16/5.12e-9/2.07e-9/0` | `1.20e-2` | UNASKED addition; ASKED audit, retained |
| `een_metric_weighting=nemo` | `EXP00/namelist_cfg:163-165`; `dynvor.F90:518-531` | `1.51e-16/1.93e-16/6.94e-18/6.94e-18/0` | `3.97e-11` | UNASKED addition; ASKED audit, retained |
| `een_q_boundary=nemo_live` | `EXPREF/namelist_ref:1069`; `dynvor.F90:450-490` | `0/0/0/0/0` | `0` | UNASKED addition; ASKED audit, retained (uninformative through kt=10) |
| `nemo_two_band_full_shortwave=true` | `EXP00/namelist_cfg:73,76-79`; `traqsr.F90:665-712,1274-1276` | `2.20e-3/1.97e-14/0/0/0` | `9.01e-2` | UNASKED addition; ASKED audit, retained pending selector consolidation |

The earlier “142× kt=2 T improvement” is attributed to this named selector
set, not to “round 8”: the direct ablations show the two live tracer-scale
members are full two-band shortwave and EVD N2 timing, while `nemo_avg4`
becomes trajectory-scale by kt=10.  Their effects are nonlinear and partly
cancelling, so the audit does not invent an additive per-arm share.  The
production-JIT arm artifact is `selector_arms.json`, SHA-256
`487879f62bb95fa4f5f22d8eab9edd4fc70eb27ab632e9eab07a24d4b41aee3f`.
The `--plant` two-choice mutation exits 2, proving the one-variable manifest
fails closed.

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
the direct drag operands agree to `4.08e-27` absolute and combined
`trd_u/trd_v` agree to `3.83e-23`.  Omitting only in-substep drag produces
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

### Stage-2 momentum does not close under production JIT

The round-8 values in this subsection were produced with `JAX_DISABLE_JIT=1`
and are withdrawn as certification evidence.  The production-JIT completed
stage-2 Kaa state is DEBT at `4.674608410863007e-13` u and
`5.187571089381761e-13` v.  Replacing the stage-1 thermodynamic bundle moves
only `7.93e-17` (`1.53e-4` of the residual), so that arm is
**NEAR-NULL_NO_DISCRIMINATING_POWER**, not exoneration.  The ordered JIT source
terms themselves remain AT-BAR: HPG `1.07e-16`/`1.22e-16`, vorticity
`2.46e-21`/`2.45e-21`, and advection `1.65e-24`/`1.63e-24` (u/v).  The
composition/update between those terms and Kaa is therefore the first
observed JIT boundary; no owner is assigned.

### Stage-3 transport and WZV

NEMO forms `zFu/zFv` from the retained Kmm stage velocity and barotropic mean
at `stprk3_stg.F90:257-278`, then passes those already materialized arrays to
`wzv(...,np_transport)` at `traadv.F90:220-226`.  Production now reuses that
same pair rather than recomputing an algebraically equivalent transport.

| boundary | normalized max | disposition |
|---|---:|---|
| `zFu` | `2.268967008217296e-9` | DEBT; production-JIT |
| `zFv` | `2.4393178638950564e-9` | DEBT; production-JIT |
| `zFw` / shared-transport production | `4.367606308796769e-7` | DEBT; production-JIT |
| `zFw` / legacy rederived transport | `4.367606359827147e-7` | DEBT; production-JIT |

The private stored-barotropic-mean association arm moves only
`2.1235675846249628e-13` against the JIT `4.367606308796769e-7` residual; it is
**NOT_SOLE_OWNER**.  Sharing the materialized transport moves only
`1.6533842413934496e-13`, also **NOT_SOLE_OWNER** under JIT.  The eager owner
labels in the previous version of this receipt are withdrawn.

A new config-local WRITE-only `MY_SRC/traadv.F90` records `ww` after WZV,
after the adaptive partition, and final `pFw`.  GYRE resolves
`ln_zad_Aimp=.false.`, so `wi` is a zero-extent NEMO array and the parser
records that logical boundary as exact zero.  Under production JIT the WZV
velocity is already DEBT at `1.569607877847196e-14`; `pFw` is
`4.3676062860970496e-7`.  The eager-only `8.44e-21`/`2.3488e-13` statement is
withdrawn.

Scaling-first arms assign no false sole owner:

- replacing only `pFu/pFv` moves `4.367606243337112e-7` and leaves
  `9.221013305395866e-13`;
- replacing only the `Kbb/Kmm/Kaa` SSH operand triplet moves
  `9.221013305395866e-13` and leaves `4.367606243337112e-7`;
- the planted SSH-arm control writes exactly `1.0` into a wet `ww` cell and
  fails closed.

The horizontal transport is the dominant upstream contributor but does not
clear the bar; SSH is near-null at this scale.  The literal WZV recurrence is
not assigned a sole-owner label.

### First unmeasured boundary: implicit ZDF

The eager-only claim that the explicit stage-3 accumulator cleared is
withdrawn.  Its production-JIT rerun is DEBT before ZDF: T
`7.325244772979124e-14`, S `5.786348021897527e-15`.  Injecting the oracle
pre-ZDF tracer does not move the compiled final state, so the honest label is
**PRE_ZDF_ACCUMULATOR_DEBT_NO_ZDF_OWNER**.  The JIT whole-step kt=2 tracer rows
are T `1.3608682491316726e-12` and S `2.2181101297999213e-14`.
`tra_zdf`/`dyn_zdf` internals remain **UNMEASURED** and this round does not
start their matrix walk.

## Re-pinned kt=1 and kt=2…10 sweep

At kt=1 T/S are bit-exact.  At-rest u/v/SSH are still **UNINFORMATIVE** because
their oracle and candidate values are identically zero.  The first whole-step
DEBT remains kt=2:

| field | kt=2 normalized max | status |
|---|---:|---|
| T | `1.3608682491316726e-12` | DEBT |
| S | `2.2181101297999213e-14` | DEBT |
| u | `9.484089544036715e-7` | DEBT |
| v | `8.987995890362757e-7` | DEBT |
| SSH | `4.77048955893622e-16` | AT-BAR |

The review requested enumeration of the receipt's 16 worsened rows, but that
count was itself eager-only and is retracted rather than laundered into the
certificate.  Like-for-like production-JIT sweeps at `57429ecf5f3` and this
tip give 41 improved, five identical, and four worsened rows:

| field | kt | `57429ecf5f3` normalized max (JIT) | current normalized max (JIT) | current / baseline |
|---|---:|---:|---:|---:|
| u | 5 | `1.7229656645948714e-2` | `2.0733890538774537e-2` | `1.203383849419297` |
| u | 8 | `4.5264965475388851e-2` | `4.7180060361621853e-2` | `1.0423085462701671` |
| u | 9 | `4.3327556604067334e-2` | `5.2560399634250458e-2` | `1.2130940157681636` |
| u | 10 | `4.1745453128264214e-2` | `5.6249872588443578e-2` | `1.3474490842299418` |

The baseline and current production-JIT sweep artifacts have SHA-256
`c77bca9cf568c08423004944d0dca0dbb2a77973b26bc982e8353754387f0219`
and `17b1d103bd6e6871ddeecbfe59d9ccc9b7a5551041c4dd13e2c1f1699cc66489`,
respectively.  The kt=2 T residual improves about 142× in this named selector
set comparison; the one-variable attribution is in the preceding disclosure.
At kt=10 the production-JIT normalized maxima are T
`5.6366385875689e-3`, S `1.5483121735304233e-4`, u
`5.624987258844358e-2`, v `1.1173366024295098e-2`, and SSH
`2.12554757616743e-4`.

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

Round 8's host exhaustion was caused by retaining more than a dozen full GYRE
executables, one per static private hook, in a single process—not by the
30×20×31 production step or the 50-substep `lax.scan`.  A lone compiled gate
completed in 34 seconds.  The gate now obtains internal frames through
`_NEMOWSRK3TestHooks.expose_barotropic_substeps` on `.step()`, materializes
each diagnostic result to host, and clears that obsolete executable before
the next static variant.  The complete kt=1…10 run then completed CPU-only
with `execution_regime=production_jit`; `JAX_DISABLE_JIT` is now a hard gate
failure.  Artifact `/tmp/gyre_r9_jit_full.json` has SHA-256
`9567f31c4840440a185b0b0cce56c35a07e9c81ce04236d8097caf197e1eedf1`.

The fp64 regression executes the new EOS, literal SCO HPG, and QCO transport
geometry paths both eagerly and under `jax.jit`, then requires
`np.array_equal` for every returned array.  Removing the live optimization
barriers changes `r3u/r3v` bits and triggers the planted inequality control;
the focused run passed (`1 passed`).

This is a Codex-internal measurement round.  Independent Claude and GLM
review of these new findings remains outstanding; no dual-review claim is
made.  Per the accepted process note, the earlier EMP reversal returned
through review with both source and runtime evidence before it landed.
