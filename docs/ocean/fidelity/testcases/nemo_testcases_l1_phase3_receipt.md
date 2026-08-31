# NEMO testcase lane 1 — phase-3 first-divergence receipt

Date: 2026-08-31

Session: `01a0591f-a335-7060-9257-6c47bf2149ee`

Preregistration: `a6ff36fec1f`

## Verdict

**EOS and the selector implementations are complete; trajectory parity remains
DEBT at the first evolved LOCK state, kt=2.**  The exact state prefix ends
at kt=1.  In accordance with the preregistered first-divergence rule, neither
kt=3 nor the OVERFLOW-zps trajectory was scored.  This is a stopped, owned
first divergence—not a trajectory-match claim.

| dependency | result | disposition |
|---|---:|---|
| NEMO TEOS-10 polynomial, both cases | max `4.4321e-16` normalized | AT-BAR |
| stage-1 LOCK full u RHS (at-rest HPG owner) | `3.6863e-17` | AT-BAR |
| LOCK time-level ladder | Kaa `3,2,3`; Kmm `1,3,2` | VERIFIED |
| LOCK kt=1 entry | T/S/u/SSH exact; v structurally absent | exact prefix |
| LOCK kt=2 T | `1.2970e-13` (`6.93e6` improvement) | DEBT |
| LOCK kt=2 u | `5.7155e-10` | DEBT |
| LOCK kt=2 S | exact, but oracle `n_unique=1` | UNINFORMATIVE |
| LOCK kt=2 SSH | `4.7797e-28`, but oracle identically zero | UNINFORMATIVE |
| LOCK kt=2 v | no active meridional face; stored zeros checked | UNMEASURED |
| kt>=3 and OVERFLOW trajectory | stopped behind LOCK kt=2 | UNMEASURED |

Every candidate floating array in both gates is `float64`, the entry point
sets and verifies `PrecisionPolicy.fp64()`, and the reported JAX backend was
CPU.  A requested GPU was unavailable to JAX (`CUDA_ERROR_NO_DEVICE`), so the
recorded science invocations used `JAX_PLATFORMS=cpu`; no numerical result is
silently attributed to a GPU.

## Pre-implementation search and canonical options

The preregistered search covered `ocean/recipes.py`, the shared lat-lon C-grid
model, `ocean/fidelity/`, the DINO harness, Kamm twin machinery, EOS tables,
barotropic filters, vertical momentum operators, and tracer integrators.  It
found the full Roquet/NEMO TEOS-10 coefficient table already canonical in
`ocean/eos.py`, but no selectable `make_eos_fn` arm; the new
`eos="nemo_teos10"` dispatch reuses that table.  It found NEMO's Demange
filter but not filter 1, no active-UP3 vertical momentum selector, and no NEMO
Wicker-Skamarock tracer stage program.  Those gaps were implemented as shared,
selectable options:

- `nemo_boxcar1_ab3` for OVERFLOW and existing `nemo_ab3am4` for LOCK;
- `nemo_up3` vertical momentum advection;
- `rk3_ws` for momentum and tracer stage coefficients;
- `nemo_kmm` stage-resolved tracer transports;
- `nemo_teos10` with `eos_depth="geometric"`.

The default configuration of every unrelated card remains unchanged.  The
testcase recipes remain pure configuration over shared canonical blocks; no
testcase solver was added.  Fidelity-only code reads artifacts, aligns NEMO
levels, scores rows, and writes receipts.

## EOS result and reconciliation

The old `veros_gsw` card differed from the independent NEMO literal polynomial
by `8.89e-5`–`8.93e-5` normalized in LOCK and `8.824e-3`–`8.831e-3` in
OVERFLOW.  After selecting the existing Roquet table, all six actual-state
rows are at bar: LOCK's maximum is `2.213e-16`; OVERFLOW's is `4.432e-16`.
The 52 parsed coefficients match the independent source literal exactly.

One assembly disagreement was found and reconciled.  The EOS probe supplied
geometric depth, while the cards initially retained the generic
`eos_depth="insitu"` default.  NEMO passes live geometric `gdept` to the
polynomial (`src/OCE/TRA/eosbn2.F90:253-288`).  Pinning
`eos_depth="geometric"` reduced the actual LOCK stage-1 u RHS error from
`5.6854e-9` to `3.6863e-17`; the gate records the latter.  Thus EOS/HPG is
exonerated at the first active stage, not merely in an isolated EOS probe.

## Selector resolution

The cards now resolve the oracle programs per case:

| selector | LOCK_EXCHANGE-zco | OVERFLOW-zps | source |
|---|---|---|---|
| barotropic filter | `nemo_ab3am4`, alpha `.07` | `nemo_boxcar1_ab3`, alpha `0` | `dynspg_ts.F90:1068-1094,1265-1273` |
| runtime external substeps | `1` | `3` | `dynspg_ts.F90:1223-1240` |
| vertical momentum | `nemo_up3` | `nemo_up3` | `dynadv.F90:78-90`; `dynadv_up3.F90:250-358` |
| outer/tracer stages | `rk3_ws` | `rk3_ws` | `stprk3.F90:184-207`; `stprk3_stg.F90:535-570` |
| tracer transport levels | `nemo_kmm` | `nemo_kmm` | `stprk3.F90:194-207`; `stprk3_stg.F90:160-168,195-213,225-303,456-519` |

The phase-3 preregistration's statement that the cases "resolve nn_e=30" is
retracted: 30 is the input value, but both namelists enable `ln_bt_auto`.
NEMO recomputes the executed count from depth, metric, gravity, timestep, and
`rn_bt_cmax`.  The run receipt reports LOCK `nn_e=1`
(`lock_kt1_3/ocean.output:763-764`); the certified OVERFLOW output reports
`nn_e=3`.  `nemo_auto_substeps` and both cards now reproduce those executed
values and accept NEMO's valid one-substep result.

Direct tests cover the filter recurrence/window, UP3 literal recurrence,
rest/zero-flux controls, differentiability, WS stage polynomial, EOS literal,
card construction, and invalid auto-substep inputs.  These implementations
close the former **selector availability** waivers; the coupled RK3 program is
separately red below.

## First divergence and ownership

The finer-cadence NEMO run used the certified phase-1 binary and configuration,
changing only `nn_itend` and `nn_stock` from 61200 to 3.  It completed on CPU
and wrote entry dumps at kt=1,2,3.  The gate consumed only kt=1 and kt=2, then
stopped.  The certified DINO config/binary and phase-1 testcase configs were
not modified.

The separate `LOCK_EXCHANGE_OMIP_L1_P3` build adds read-only stage/RHS/
transport dumps in its own MY_SRC.  Its one-step run has the same geometry and
physics namelist and completed on CPU.  `ocean.output:880-964` records the
instrumented ladder: stage Kaa `3,2,3`, transport Kmm `1,3,2`.

The first-divergence gate now assigns every ownership statement an explicit
evidence class:

1. **EOS/HPG — CONFIRMED EXONERATED.**  At rest, momentum advection,
   viscosity, and `f=0` vorticity are zero.  The complete stage-1 u RHS agrees
   at `3.6863e-17`.
2. **Per-stage barotropic correction — CONFIRMED EXONERATED; prior owner label
   RETRACTED.**  The old gate compared NEMO's measured stage mean against
   `np.zeros_like`, not a computed legoESM stage, and the alleged
   `1.1354e-3 m/s` owner was incompatible in scale with the evolved-state
   discrepancy.  The replacement gate computes legoESM's actual pre-solve
   stage-1 mean with the production thickness weights.  Its direct control
   confirms that a simple level mean happens to equal the thickness-weighted
   mean exactly on LOCK's twenty uniform 1 m layers; that shortcut is neither
   assumed nor used on other grids.  The causal option then installs the
   post-solve external-mode mean at each Kaa, following
   `stprk3_stg.F90:44,143-144,206-207,225,433-446`.  It moves the registered
   wet-point L-infinity u error from `5.7155454103e-10` to
   `5.9807602992e-10 m/s`, a worsening of `2.6521488885e-11 m/s`.
3. **Kmm-consistent stage transports — CONFIRMED OWNER of the large T
   divergence.**  NEMO constructs `zFu/zFv/zFw` from Kmm before every tracer
   stage (`stprk3.F90:194-207`; `stprk3_stg.F90:160-168,195-213,225-303,
   456-519`).  The new selectable canonical path carries Kbb, stage-1 Kaa, and
   stage-2 Kaa velocities with their qco stage thicknesses through the real
   flux/FCT path.  Against the same kt=2 dump, selecting it moves T from
   `8.9924674545e-7` to `1.2970365522e-13`, a `6.933087e6` improvement.  The
   remaining T value is still DEBT against the registered `1e-15` bar.
4. **FCT two-step program — UNMEASURED.**  NEMO dispatches `fct_up1_2stp`
   under `key_RK3` (`traadv_fct.F90:153-161,470-641`), while legoESM still has
   its generic FCT staging.  Source difference is not causal ownership:
   limiter coefficients, antidiffusive fluxes, and a two-model intervention
   have not been measured.

Rule-1e reconciliation of the velocity scale is explicit rather than an
average.  The review quoted `1.27e-12 m/s` without a statistic definition.  A
fresh read of the whole registered wet u field does not reproduce that scalar:
the gate's declared statistic is absolute L-infinity (`5.7155454103e-10
m/s`), with wet-point mean absolute error `3.9333948945e-12 m/s`, RMS
`3.1472972252e-11 m/s`, and reference maximum `2.2144535390e-3 m/s`.  These
are different reductions and are therefore not interchangeable.  Both the
review value and every reproduced field norm remain at least six orders below
the `1.1354e-3 m/s` stage diagnostic, so the scaling refutation and the
exoneration are unchanged.  Ownership of the remaining u residual is
UNMEASURED.

## Gates, controls, test reconciliation, and artifacts

`nemo_testcase_phase3_trajectory_gate.py` is fail-closed on dump headers,
shapes, finite values, masks, dtype, and the first over-bar step.  Its planted
wet-T control is red.  `nemo_testcase_phase3_first_divergence_gate.py` also
fails on an invalid Kaa/Kmm ladder, a planted RHS value, and malformed magic.
It now computes the candidate stage mean and reports the causal Kmm and
per-stage-correction arms; the old zero-substitution control was removed with
its retracted claim.

The review-round-2 residual count remains the historical phase-2 count stated
in that receipt: **41 tests across the four files at review input, 42 after r1**;
the earlier 61 included 20 separately named supporting regressions.  For this
phase-3 review fix, the exact executed set is **93 tests across eight explicitly
listed files**, broken down as `3 EOS gate + 6 trajectory gate + 6 first-
divergence gate + 8 Roquet EOS + 11 card + 3 UP3 + 3 WS tracer + 53
barotropic-common`.  This is a new, explicitly scoped test run, not an average
or substitution for the phase-2 count.  The WS set includes an
un-monkeypatched model step through the real mass-flux, diagnosed-w, and FCT
path; changing the velocity/time-level selector produces a `>2e-5 K`
difference.  The Roquet set hash-pins all 52 TEOS-10 density coefficients from
local IEEE-754 bytes and proves that a planted `1e-11` coefficient perturbation
changes the digest, without reading the NEMO checkout.

Full artifact hashes are committed in
`nemo_testcases_l1_phase3_artifacts.sha256`.  The trajectory and diagnostic
JSONs are external run products under
`/data/abyssal/dbalwada/nemo-testcases-l1/phase3/`; their hashes make the
receipt reproducible without committing quota-heavy binary dumps.

## Loud UNMEASURED register

- owner of the final `5.7155e-10` wet-point L-infinity u residual; the
  per-stage barotropic correction is causally exonerated;
- individual NEMO/legoESM FCT limiter coefficients and antidiffusive fluxes;
- every LOCK state at kt>=3, by the stop rule;
- every OVERFLOW-zps trajectory state and its BBL transport, because LOCK is
  still red;
- long-trajectory phenomenology and statistical equivalence, which are beyond
  this first-divergence dispatch.
