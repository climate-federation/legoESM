# NEMO testcase lane 2 — GYRE causal and barotropic-boundary receipt

Date: 2026-09-01

Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`

This receipt freezes the `efdef59b28af` boundary round.  The subsequent ENE
operand walk, corrected selector, and kt=10 rerun are authoritative in
`nemo_testcases_l2_gyre_phase3_ene_operand_receipt.md`; numbers below are
retained as the historical generic-coefficient control.

Preregistration commit: `34ca9a913`

Parent resolved-program commit: `fbeef5d4d3e0513f4b1b8bedbd2e8bc7c1a74e91`

## Verdict

**DEBT, with the first barotropic divergence localized inside substep 2 and
TKE/EVD ordering established as a causal contributor.**  Substep 1 matches all
registered entry, AB3-midpoint, continuity, PGF, ENE, slow-forcing, and exit
frames at the `1e-15` bar.  At substep 2, `trd_u` is the first over-bar frame:
absolute error `8.602761661688124e-12` against oracle magnitude
`5.409032045634095e-10`, i.e. `1.5904%` relative; `trd_v` differs by
`8.706079133797117e-12`.  The
following velocity exits differ by `2.477595358568228e-9` and
`2.507350790534324e-9`.  This identifies the live barotropic ENE tendency
boundary, not a sole coefficient owner.

The stage-entry shortwave arm is causal but non-primary at `kt=2`.  Replacing
the effective vertical-mixing profiles at their actual legoESM solve boundary
moves the u/v residual by `1.0003` times its faithful magnitude and leaves
about `1.3e-5`; it is therefore
**CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER**, not an exact-match claim.  The whole-step owner remains
`UNMEASURED_AFTER_REGISTERED_ARMS`, and trajectories remain DEBT.

## Claude SHIP notes closed first

The resolved-program gate now parses the run's `output.namelist.dyn` at
runtime, enumerating every `NAMDYN*`, `NAMZDF*`, and `NAMTRA*` group, then
compares that set with the card's explicit disposition map.  The renamed test
`test_resolved_program_coverage_parses_runtime_namelist` plants a future
fourteenth group and requires a DEBT row because the static disposition has no
matching entry.  This is runtime block discovery plus a static scientific
checklist, not runtime parsing of every scalar selector.

The two completed-step momentum scaling arms are marked
`NEAR-NULL_AT_KT2`: vector/ENE/C2 changes u/v by only
`8.067732246803613e-9`/`9.505427630093375e-9`, and the level-Laplacian arm by
`1.3245560308674294e-8`/`1.329978956872706e-8`, millions of times below the
`2.5256e-2` residual in the corrected run.  They have no discriminating power
at this rung and are explicitly **not exonerated**.

## Oracle program and pre-implementation search

The resolved run prints `nn_bt_flt=3`, `rn_bt_alpha=.07`, `nn_e=50`, and
`ln_bt_fw=T` at `ocean.output:852-861`; `rDt_e=288 s` follows from the pinned
14,400 s whole step and 50 substeps.  In the executing source:

- `stprk3.F90:174-205` computes BN2 and `zdf_phy` from Nbb, then calls `stp_2D`
  once before the three RK stages at `:213-230`;
- `dynspg_ts.F90:274-305` installs the RK3 forcing and removes the Kmm
  barotropic Coriolis;
- `:552-572` applies the forward/AB3 predictor, `:647-713` applies continuity,
  AM4 pressure and live `dyn_cor_2D`, and `:828-846` writes and rotates each
  substep frame;
- `stprk3_stg.F90:602-619` adds two-band `tra_qsr` only at stage 3.

The search found DINO's canonical AB3/AM4 seed/history, QCO face-depth,
continuity, transport accumulation, PGF and filter-3 machinery, and the
canonical 3-D `pv_flux_ene` recurrence.  GYRE reuses those pieces.  It bypasses
the MLF-only post-loop trend update and instead composes the one external-mode
solve into all three RK stages.  The existing EEN barotropic adapter could not
stand in for ENE, so the new `ene_metric` sibling reuses `pv_flux_ene` with the
same frozen Kmm thickness/metric bundle.  The card now fails closed unless
`barotropic_coriolis_split="live"` and
`barotropic_coriolis="ene_metric"` are selected together.

## Instrumentation and bit identity

The NEMO extension is WRITE-only:

- `stprk3.F90` writes stage-entry `avm/avt` after `zdf_phy`;
- `stprk3_stg.F90` writes `qsr` and the isolated stage-3 temperature-RHS
  increment;
- `dynspg_ts.F90` writes 18 registered arrays for each of 50 substeps.

Seven pre-existing artifacts — kt=1/2 entry, all three stage dumps, entering
RHS, and the first external-mode frame — reproduce their registered SHA256
values bit-for-bit.  The transport reader also records one non-finite
unowned-halo sentinel in stage 1 while requiring every owned A2D value to be
finite.  No scored value is sanitized.

The executed `--plant-barotropic --max-step 1` control moves only substep-1
`eta_entry` by `1.0`; the gate reports that exact planted value as the first
over-bar boundary.  The control report SHA256 is
`810ac8c36f0b27363acb9f05a0255417ac33172229ef3102fa13249c03cd4438`.

## Forcing corrections resolved by the trace

Two composition errors were corrected before assigning a barotropic owner.
First, `usrdef_sbc` already returns stress in native i/j directions; treating
it as geographic rotated the pair a second time.  The gate now converts that
native pair back to the public east/north forcing channel, after which the
registered `zu_frc/zv_frc` slow terms match to `9.926167350636332e-24` at all
50 substeps.

Second, the phase-2 “unmasked numerator includes 104 land cells” reading is
retracted.  Although `usrdef_sbc.F90:122-140` passes the unmasked array to
`glob_2Dsum`, the first oracle SSH increment pins an effective EMP mean of
`8.50091456831154e-6`, exactly the 600-owned-cell numerator.  The naive
704-cell value `9.166209883944753e-6` would create the measured
`1.86749562283012e-7` first-substep SSH error.  The global reduction excludes
the non-owned boundary ring.  This is also the source-defined behavior:
`lib_fortran_generic.h90:92,144-148` multiplies each 2-D reduction operand by
`smask0_i`, and `dommsk.F90:200-205` constructs `smask0_i` from the unique
interior-domain mask.  Card, independent phase-2 gate, and tests now encode
that source-and-runtime ownership result.

Shortwave surface flux is AT-BAR (`2.842170943040401e-14` absolute,
`1.7112048084759902e-16` normalized).  The original penetrative mismatch was
traced to `combined.py` calling the kernel with generic constants; it now
passes the card's pinned NEMO `rho_0` and `c_sw`.  The remaining direct
temperature-rate difference is `5.257960831729332e-13`, over the absolute
bar but only `1.4199163798245935e-7` of the kt=2 T residual under the oracle
injection arm.  Its honest label is `CAUSAL_NONPRIMARY_AT_KT2`.

## Vertical-profile causal arm and sampling-boundary retraction

An initial diagnostic compared NEMO's pre-`stp_2D` `avm/avt` with legoESM
profiles evaluated on the pre-explicit state and found roundoff agreement.
That was the wrong legoESM consumption boundary and is retracted.  legoESM's
implicit solve consumes profiles after its explicit RK3 update; the diagnostic
now runs the production explicit path first and samples the effective pair at
the solve boundary.  There, enhanced diffusion produces about `100 m2/s`:

| direct operand | max error | oracle max | gate |
|---|---:|---:|---|
| `avt` | `99.999988` | `0.008188154756539789` | DEBT |
| `avm` | `99.99988` | `0.08188154756539788` | DEBT |

The private causal hook replaces only this effective pair after
TKE/EVD/background composition; it does not surface the pair before closure or
double-count TKE.  Scaling precedes labels:

| field | faithful residual | causal movement | arm residual | movement / faithful |
|---|---:|---:|---:|---:|
| T | `2.1993973308250254e-3` | `1.4169946267228702e-3` | `2.1993973308250254e-3` | `0.644265` |
| S | `1.0953611759641099e-4` | `1.1261742293745038e-4` | `5.018260134584912e-5` | `1.02813` |
| u | `2.525640044682423e-2` | `2.526465626258189e-2` | `1.2630858531885794e-5` | `1.000327` |
| v | `2.5256765852261283e-2` | `2.5264656333664842e-2` | `1.2982574012910045e-5` | `1.000312` |
| SSH | `2.3364335452164746e-5` | `0` | unchanged | `0` |

This establishes vertical-closure **ordering/consumption** as a causal
contributor.  It does not establish that EVD itself is wrong: the same
canonical closure agrees before the explicit update, and the registered arm
does not clear every field.

## Within-substep barotropic walk

Substep 1 is AT-BAR throughout: entry is exact or below `4.73e-24`, slow
forcing below `9.93e-24`, SSH exit below `6.78e-21`, and u/v exits below
`3.39e-21`.  Substep-2 entry and AB3 midpoint remain AT-BAR.  The first
over-bar rows are the live ENE tendencies:

| substep/frame | u absolute error | v absolute error | disposition |
|---|---:|---:|---|
| 2 `trd` | `8.602761661688124e-12` (`1.5904%` of oracle `5.409032045634095e-10`) | `8.706079133797117e-12` | first DEBT boundary |
| 2 velocity exit | `2.477595358568228e-9` | `2.507350790534324e-9` | downstream DEBT |
| 50 `trd` | `4.989158868354881e-10` | `5.863222494344802e-10` | accumulated DEBT |
| 50 velocity exit | `5.636317706097865e-7` | `2.1332474430218844e-7` | accumulated DEBT |

The completed barotropic frame then feeds stage 1, whose u/v errors are
`5.636317706097594e-7` and `2.1332474430221554e-7`.  The stage-3 correction
still jumps to `2.525640044682423e-2`/`2.5256765852261283e-2`.  Because the
metric-weighted `zhU/zhV` comparison lacks dumped oracle metrics and no
one-variable ENE coefficient replacement clears the frame, the coefficient
owner remains **UNMEASURED**.  What is measured is the exact first boundary:
substep 2, live `dyn_cor_2D`/ENE tendency.

## Artifacts and stopping boundary

| artifact | SHA256 |
|---|---|
| full causal/barotropic gate | `ee30165c83e211c427ab534cd9d6e5171f6c1e870489093c5147217c96cc93ed` |
| `oracle_zdf_entry_kt00000001.bin` | `c5fa72f40fd39983e5de77d28ef2839cf19f27a8a0f239ed3854fa231f31121c` |
| `oracle_qsr_stage3_kt00000001.bin` | `9af5a9ab8f6594cbf42976b098553b8c72eb69ac5481f645f097bfa711610e52` |
| `oracle_bt_substeps_kt00000001.bin` | `49cd61df3753dae967721167cfa2ba33e81242062c05fcfc69aac5b0443fb938` |
| instrumented `nemo.exe` | `25cadd80cb0c4797392fcd42f2bdfcb1e805ae511038146cc0c1c7d1e03d2e7e` |
| `MY_SRC/dynspg_ts.F90` | `8fcf36cfd4991cb33cfdc01d9c2017f05343d329683d0aeccac3d463cba09752` |
| `MY_SRC/stprk3.F90` | `11c6a66407eda69f3c8ceb57a37ed5f6565cefe4d8e32c0dee453ad021906fee` |
| `MY_SRC/stprk3_stg.F90` | `0e715b15a7540e1bc79da3b4a7f53ff12d1629496f9f8661509859fc8987487a` |

The gate stops at the preregistered boundary.  Whole-step `kt=2` remains DEBT;
the coefficient-level ENE owner, metric-weighted transport parity, internal
tracer stages, and an exact trajectory are honestly UNMEASURED.  Claude's
review of the previous round was SHIP (“strongest round of the campaign”);
this receipt closes its two minor notes.  GLM review remains outstanding, so
no dual-review claim is made.
