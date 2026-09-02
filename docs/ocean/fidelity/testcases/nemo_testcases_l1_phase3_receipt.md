# NEMO testcase lane 1 — phase-3 first-divergence receipt

Date: 2026-08-31

Session: `01a0591f-a335-7060-9257-6c47bf2149ee`

Preregistration: `a6ff36fec1f`

## Verdict

**EOS and the selector implementations are complete; trajectory parity remains
DEBT at the first evolved state of both cases, kt=2.**  The exact state prefix
ends at kt=1.  After the currently measurable LOCK kt=2 owners were exhausted,
the explicit owner-exhausted sweep scored kt=3 and OVERFLOW-zps while retaining
kt=2 as the registered first-over-bar step.  These are red measurements, not
trajectory-match claims.

| dependency | result | disposition |
|---|---:|---|
| NEMO TEOS-10 polynomial, both cases | max `4.4321e-16` normalized | AT-BAR |
| stage-1 LOCK full u RHS (at-rest HPG owner) | `3.6863e-17` | AT-BAR |
| LOCK time-level ladder | Kaa `3,2,3`; Kmm `1,3,2` | VERIFIED |
| LOCK kt=1 entry | T/S/u/SSH exact; v structurally absent | exact prefix |
| LOCK kt=2 T | `1.5821e-13` normalized (`4.7464e-12 K` absolute) | DEBT |
| LOCK kt=2 instantaneous u (`uu(Nbb)` vs prognostic u) | `1.7120868e-10 m/s` | DEBT |
| LOCK kt=1→2 time-mean u transport (`un_adv` vs `Hu_avg`) | `8.6736e-19 m2/s` | AT-BAR |
| LOCK kt=2 S | exact, but oracle `n_unique=1` | UNINFORMATIVE |
| LOCK kt=2 SSH | `4.7797e-28`, but oracle identically zero | UNINFORMATIVE |
| LOCK kt=2 v | no active meridional face; stored zeros checked | UNMEASURED |
| LOCK kt=3 T/S/u/SSH | `9.1634e-12 / 2.0301e-16 / 2.0540e-9 / 1.3553e-18`; S uninformative, SSH at bar | DEBT |
| OVERFLOW kt=2 T/S/u/SSH | `2.40035e-8 / 2.0301e-16 / 3.95605e-6 / 1.23723e-7`; S uninformative | DEBT |
| OVERFLOW kt=3 T/S/u/SSH | `2.86507e-7 / 2.0301e-16 / 1.48997e-5 / 5.12822e-6`; S at bar | DEBT |

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
- `nemo_rk3_two_step` FCT low-order prediction;
- WS external-mode and distinct advecting-transport reconciliation;
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
| FCT low-order predictor | `nemo_rk3_two_step` | `nemo_rk3_two_step` | `traadv_fct.F90:153-161,470-641` |
| stage velocity/transport reconcile | Kaa external mean + `un_adv/hu` | same | `stprk3_stg.F90:257-274,433-446` |

**Resolved default decision (standing Rule 3):** the stored selector sentinel
resolves at model validation to `nemo_kmm` only when
`tracer_time_integrator="rk3_ws"`; every other integrator resolves to the
previous `frozen_final`, preserving its prior validation and numerical path.
An explicit `frozen_final` remains the documented legacy WS arm.  The two
testcase cards continue to pin `nemo_kmm` explicitly in their resolved run
configuration.

The phase-3 preregistration's statement that the cases "resolve nn_e=30" is
retracted: 30 is the input value, but both namelists enable `ln_bt_auto`.
NEMO recomputes the executed count from depth, metric, gravity, timestep, and
`rn_bt_cmax`.  The run receipt reports LOCK `nn_e=1`
(`lock_kt1_3/ocean.output:763-764`); the certified OVERFLOW output reports
`nn_e=3`.  `nemo_auto_substeps` and both cards now reproduce those executed
values and accept NEMO's valid one-substep result.

## Standing rule: no Frankenstein compositions

**Binding user decision (2026-08-31):** every selectable arm must name the
reference configuration that exercises it: either a NEMO source/namelist
citation or **legoESM legacy, pre-existing behavior**.  Provenance of two arms
separately does not attest their composition.  A combination that no reference
model runs must either (a) be rejected by validation or (b) require an explicit
experimental flag and warning; it may never be a silent third model.  This rule
applies to this and every later round.

This audit instantiated both cards through `LatLonCGridOceanModel` under the
fp64 policy, rather than reading recipe declarations alone.  It inventories the
explicit card selectors, inherited selectors that are numerically live on the
cards, and every arm introduced by this lane.  `Lcfg` and `Ocfg` below mean the
executed phase-3 LOCK and OVERFLOW `namelist_cfg` files under
`/data/abyssal/dbalwada/nemo-testcases-l1/phase3/{lock,overflow}_kt1_3/`.
No validation semantics changed in this audit; the dispositions below are
proposals to resolve before more trajectory claims.

### Instantiated card-arm provenance

| selector arm | card(s) | reference that exercises the arm |
|---|---|---|
| `metric_convention="exact"` | both | NEMO LOCK/OVERFLOW `usrdef_hgr.F90:88-97` (constant Cartesian scale factors) |
| `vface_zonal_metric_evaluation="nemo_vpoint"` | both | NEMO LOCK/OVERFLOW `usrdef_hgr.F90:75-97` (separate T/U/V/F coordinates and metrics) |
| full-step z coordinate | LOCK | NEMO `tests/LOCK_EXCHANGE/MY_SRC/usrdef_zgr.F90:92-108,146-155`; shipped `key_vco_1d` configuration |
| `bottom_index_rule="nemo_tpoint"` | OVERFLOW | NEMO `tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:157-186`; `tools/DOMAINcfg/src/domzgr.F90:1163-1169` |
| `eos="nemo_teos10"` | both | executed `ocean.output:157` (LOCK), `:159` (OVERFLOW); `src/OCE/TRA/eosbn2.F90:1920-2108` |
| `eos_depth="geometric"` | both | NEMO `src/OCE/TRA/eosbn2.F90:253-288`, which passes live `gdept` |
| `tracer_advection="fct2"` | both | `Lcfg:68-71`; `Ocfg:69-72` (`ln_traadv_fct=T`, `nn_fct_h=nn_fct_v=2`) |
| `tracer_time_integrator="rk3_ws"` | both | shipped case `cpp_*.fcm:1` (`key_RK3`); `src/OCE/stprk3.F90:184-207` and `stprk3_stg.F90:535-559` |
| `momentum_time_integrator="rk3_ws"` | both | same `key_RK3` configurations; `stprk3_stg.F90:112-238,306-446` |
| `tracer_fct_low_order_predictor="nemo_rk3_two_step"` | both | `src/OCE/TRA/traadv_fct.F90:153-161,470-641` under `key_RK3` |
| `tracer_rk3_transport_time_levels="nemo_kmm"` | both | `stprk3.F90:194-207`; `stprk3_stg.F90:250-303,456-519` |
| `rk3_ws_stage_barotropic_correction=True` | both | NEMO HYB stage program, `stprk3_stg.F90:118-168,204-234,433-446` |
| `rk3_ws_momentum_transport_reconcile=True` | both | NEMO `un_adv/hu` correction, `stprk3_stg.F90:257-274,315-333` |
| `momentum_advection="flux_form"` | both | `Lcfg:82-85`; `Ocfg:83-86`; `src/OCE/DYN/dynadv.F90:78-90` |
| `momentum_flux_scheme="upwind3"` | both | same namelist rows; `src/OCE/DYN/dynadv_up3.F90:141-365` |
| `vertical_momentum_scheme="nemo_up3"` | both | same active UP3 call; `dynadv.F90:87-89`, `dynadv_up3.F90:239-365` |
| `pgf_scheme="nemo_sco"` | both | `Lcfg:95-98`; `Ocfg:96-99`; `src/OCE/DYN/dynhpg.F90:117-123,340-390` |
| `pgf_quadrature="nemo_trapezoid"` | both | NEMO `dynhpg.F90:340-380` surface/interior recurrence |
| `barotropic_solver="explicit_substep"` | both | `Lcfg:100-106`; `Ocfg:101-104` (`ln_dynspg_ts=T`) |
| `barotropic_time_filter="nemo_ab3am4"` | LOCK | `Lcfg:105-106`; executed `ocean.output:758-760`; `dynspg_ts.F90:1068-1094` |
| `barotropic_time_filter="nemo_boxcar1_ab3"` | OVERFLOW | `Ocfg:101-104` plus reference defaults; executed `ocean.output:892`; `dynspg_ts.F90:1265-1273` |
| auto substeps `1` / `3` | LOCK / OVERFLOW | both `namelist_ref:1099-1100`; executed outputs `:764` / `:897`; `dynspg_ts.F90:1223-1240` |
| `adaptive_implicit_vertadv=True` | both | `Lcfg:126-134`; `Ocfg:123-131`; stage-3 partition at `stprk3_stg.F90:281-303` |
| `implicit_vertical_mixing=True`, `A_v=1e-4`, `K_v=0` | both | same `namzdf` rows; stage-3 `dyn_zdf` call at `stprk3_stg.F90:428-430` |
| explicit lateral tracer/momentum mixing off | both | `Lcfg:74-78,109-123`; `Ocfg:75-79,106-120` |
| `lateral_side_bc="free_slip"` | both | `Lcfg:39`; `Ocfg:40` (`rn_shlat=0`) |
| BBL off | LOCK metadata | executed `ocean.output:647-650` (`ln_trabbl=F`) |
| BBL advective option 2, diffusive option 0, gamma 20 s | OVERFLOW metadata | `Ocfg:134-140`; executed `ocean.output:776-783` |
| `outer_integrator="forward_euler"` | both | **legoESM legacy, pre-existing behavior**; here it is the no-extra-wrapper sentinel around the selected inner WS-RK3 program, not an Euler replacement for that program |
| `vorticity_scheme="al81"` | both | **legoESM legacy, pre-existing behavior**; inherited, while NEMO selects ENS at `Lcfg:88-93` / `Ocfg:89-94` |
| `coriolis_scheme="matsuno_split"` | both | **legoESM legacy, pre-existing behavior**; inherited; both oracle grids set `f=0` at `usrdef_hgr.F90:100-104`, so the mismatch is currently a dead-arm fact, not an attestation |
| `tracer_wall_neumann_fill=True` | both | **legoESM legacy, pre-existing behavior**; no NEMO case citation presently attests the halo-fill equivalence |
| `barotropic_face_depth="min_rule"` | both | **legoESM legacy, pre-existing behavior** |
| `barotropic_continuity_evaluation="generic"` | both | **legoESM legacy, pre-existing behavior** |
| `barotropic_transport_accumulation_evaluation="generic"` | both | **legoESM legacy, pre-existing behavior** |
| `barotropic_seed_evaluation="generic"` | both | **legoESM legacy, pre-existing behavior** |
| `barotropic_seed_face_depth="min_rule"` | both | **legoESM legacy, pre-existing behavior** |
| `barotropic_pgf_evaluation="generic"` | both | **legoESM legacy, pre-existing behavior** |
| `barotropic_reconcile_target="velocity_avg"` | both | NEMO `src/OCE/DYN/dynspg_ts.F90:1170-1172`; the arm is attested, but its present composition with the preceding generic arithmetic is not |
| `barotropic_diffusion_alpha=0.01`, reference dt 60 s | both | **legoESM legacy, pre-existing behavior**; this positive coefficient is a live extra eta-diffusion operator, not NEMO's `rn_bt_alpha` filter parameter |
| `zdf_implicit_solver_evaluation="shared_thomas"` | both | **legoESM legacy, pre-existing behavior**; the available `nemo_literal` arithmetic is not selected |
| `implicit_vmix_dzw_slot=False`, `implicit_vmix_e3t_now_divisor=False` | both | **legoESM legacy, pre-existing behavior**; these stored time-level arms enter the active implicit-mixing program |

Selectors that are dispatch-inactive under these cards (WENO, EEN, leapfrog,
AB2, lateral-viscosity, forcing, GM/Redi, and MLF-only evaluation arms) are not
misrepresented as reference components.  Their stored defaults remain legacy,
but they do not enter the realized testcase program.  The nonrotating
vorticity/Coriolis pair is listed despite being dead because it is a visible
card default and a future geometry change would make it live.

### Lane-added arm provenance

| added selector arm | reference |
|---|---|
| `bottom_index_rule="nemo_tpoint"` | NEMO OVERFLOW `usrdef_zgr.F90:157-186` |
| `barotropic_time_filter="nemo_boxcar1_ab3"` | NEMO OVERFLOW `nn_bt_flt=1`, executed output `:892`; `dynspg_ts.F90:1265-1273` |
| tracer `rk3_ws` | NEMO `key_RK3`; `stprk3.F90:184-207`, `stprk3_stg.F90:535-559` |
| momentum `rk3_ws` | NEMO `key_RK3`; `stprk3_stg.F90:112-238,306-446` |
| `nemo_up3` | NEMO `ln_dynadv_up3=T`; `dynadv.F90:87-89`, `dynadv_up3.F90:141-365` |
| `nemo_teos10` | NEMO executed `ln_TEOS10=T`; `eosbn2.F90:1920-2108` |
| `nemo_rk3_two_step` | NEMO `key_RK3` dispatch to `fct_up1_2stp`, `traadv_fct.F90:153-161,470-641` |
| `one_step` | **legoESM legacy, pre-existing behavior** |
| transport level `nemo_kmm` | NEMO `stprk3.F90:194-207`; `stprk3_stg.F90:250-303` |
| transport level `frozen_final` | **legoESM legacy, pre-existing behavior** |
| transport-level sentinel `None` | no numerical arm: validation resolves it to `nemo_kmm` only for tracer `rk3_ws`, otherwise the legacy `frozen_final` arm |
| stage barotropic correction `True` | NEMO `stprk3_stg.F90:433-446` |
| stage barotropic correction `False` | **legoESM legacy, pre-existing behavior** (one post-stage split correction) |
| momentum transport reconcile `True` | NEMO `stprk3_stg.F90:257-274,315-333` |
| momentum transport reconcile `False` | **legoESM legacy, pre-existing behavior** |

### Presently constructible, unattested compositions

The direct construction audit proved every family below currently passes model
construction unless explicitly described as metadata-only.  These are
composition families, not an attempt to enumerate their Cartesian product.

| # | presently constructible family | why unattested | proposed resolution |
|---:|---|---|---|
| 1 | tracer `rk3_ws` + `frozen_final` transports | NEMO advances Kmm between stages; legacy freezes one final transport | **(b)** require a specific experimental flag/warning; it remains useful as the registered causal control |
| 2 | tracer `rk3_ws` FCT + `one_step` predictor | NEMO `key_RK3` calls `fct_up1_2stp`; one-step is legacy | **(b)** experimental flag/warning for the causal control |
| 3 | WS-RK3 + stage correction `True` + transport reconcile `False` | keeps NEMO's Kaa mean but omits NEMO's distinct `un_adv/hu` transport | **(b)** experimental flag/warning for the measured stage-mean control |
| 4 | WS-RK3 + both stage correction and transport reconcile `False` | embeds legoESM's one-post-solve split inside NEMO's stage loop | **(b)** experimental flag/warning; never an ordinary card |
| 5 | only one of tracer/momentum integrators is `rk3_ws` | NEMO `key_RK3` owns one coupled momentum/tracer stage program | **(a)** reject; no reference or required control needs the decoupled program |
| 6 | `nemo_rk3_two_step` + `ppm_fct` | validation permits it, but the certified cases and literal tests exercise only FCT2 (`nn_fct_h=nn_fct_v=2`) | **(a)** reject until an exact NEMO FCT4/PPM mapping and reference configuration are certified |
| 7 | `vertical_momentum_scheme="nemo_up3"` with horizontal momentum other than flux-form/upwind3 | NEMO `dynadv` selects horizontal and vertical UP3 as one routine | **(a)** require the coupled flux-form/upwind3 selections |
| 8 | `eos="nemo_teos10"` + non-geometric (`insitu`) depth | NEMO passes geometric `gdept`; this pairing caused the measured phase-3 EOS/HPG error | **(a)** require geometric depth |
| 9 | `eos_depth="geometric"` + a non-NEMO EOS | the new depth arm has no cited non-NEMO reference | **(a)** restrict the arm to cited EOS configurations |
| 10 | `pgf_quadrature="nemo_trapezoid"` + PGF other than `nemo_sco` | NEMO trapezoid was transcribed as part of `hpg_sco`; validation enforces only the forward implication | **(a)** enforce the reverse pairing too |
| 11 | swapping LOCK/OVERFLOW NEMO filters or overriding their auto-substep counts | each arm is NEMO-attested, but the mutated testcase card is not the executed reference configuration | **(a)** case-card validation must pin filter, alpha, and resolved count together |
| 12 | certified cards with inherited `al81`/`matsuno_split` | these are legacy arms while the oracle namelists select ENS and the RK3 source owns Coriolis staging; `f=0` only makes the present mismatch inert | **(a)** select/implement the NEMO ENS/stage arm or validate a formally eliminated dead operator; do not call the legacy pair a NEMO selection |
| 13 | NEMO RK3/filter cards with legacy tracer wall fill, positive generic eta diffusion, generic barotropic face-depth/continuity/accumulation/seed/PGF arithmetic, and generic implicit-mixing arithmetic/time levels | every legacy arm has provenance, but no cited model runs this mixture; several canonical `nemo_literal` arms exist, while filter-1/filter-3 transport accumulation and the exact RK3 ZDF composition are not yet covered | **(a)** certified-card validation must reject the legacy group after the missing literal coverage is implemented and pinned; the extra eta diffusion must be disabled unless a reference is supplied |
| 14 | OVERFLOW metadata selects BBL option 2 while `model_config` has no consumed BBL selector | the NEMO arm is cited, but the legoESM solver does not execute it | **(a)** reject an “oracle-complete” OVERFLOW trajectory card until the canonical BBL selector is wired; retain the present card only as loudly incomplete/UNMEASURED |

Thus the current cards are source-coherent for the explicitly pinned
EOS/PGF/RK3/FCT/UP3/filter selections, but they do **not** yet satisfy the new
standing composition rule as oracle-complete cards because families 12--14 are
present in the resolved configuration.  No further trajectory match should be
promoted until those current-card families are resolved.  Families 1--4 are
diagnostic controls, not candidate reference models.

Direct tests cover the filter recurrence/window, UP3 literal recurrence,
rest/zero-flux controls, differentiability, WS stage polynomial, EOS literal,
card construction, and invalid auto-substep inputs.  These implementations
close the former **selector availability** waivers; the coupled RK3 program is
separately red below.

## Structural-collapse and BBL wiring round

**Standing rule refinement (binding for this and future rounds): where NEMO
has no switch, legoESM gets no switch.** The four owner-hunt fields
`tracer_rk3_transport_time_levels`, `tracer_fct_low_order_predictor`,
`rk3_ws_stage_barotropic_correction`, and
`rk3_ws_momentum_transport_reconcile` were branch-only and had no external
users. They are removed from `LatLonCGridOceanConfig`. Selecting the coupled
tracer/momentum `rk3_ws` program now unconditionally means Kmm stage
transports, the RK3 two-step FCT predictor, the per-stage external-mode
correction, and the distinct `un_adv/hu` transport reconciliation. NEMO runs
the last two together at `src/OCE/stprk3_stg.F90:433-446` and `:257-274`;
the prior nested arms were not a reference model. Causal controls now enter
only through `_NEMOWSRK3TestHooks`. Families 1--4 in the earlier audit are
therefore structurally unrepresentable in public configuration.

The remaining accepted reject dispositions are executable guards:

- family 5: tracer and momentum WS-RK3 must be selected together;
- family 6: this certified scheme identity accepts FCT2, not the unattested
  FCT4/PPM composition;
- family 7: `nemo_up3` requires the single flux-form/upwind3 momentum program;
- families 8--10: `nemo_teos10` and geometric depth are bidirectionally paired,
  as are `nemo_sco` and `nemo_trapezoid`;
- family 11: `validate_nemo_testcase_card` pins each named oracle's filter,
  auto-substep count, and BBL tuple and rejects mutations;
- family 12: validation proves `f=0` and exactly one wet meridional row, so the
  inherited rotation operator is structurally absent; a geometry that makes it
  live is rejected;
- family 13: the cards select NEMO-literal face-depth, continuity,
  transport-accumulation, seed, surface-PGF, and implicit-ZDF programs.
  Literal raw `wgtbtp2` accumulation was extended to the executed filter-1 and
  filter-3 windows from `dynspg_ts.F90:1058-1102,999-1000`. The unmatched live
  eta diffusion is zero, explicitly enforcing Rule 9.
  The stage-end closed-wall tracer fill is pinned to NEMO's `lbc_lnk` call
  (`stprk3_stg.F90:614-626`), and final 3-D momentum reconciliation selects
  `un_adv/hu` transport (`dynspg_ts.F90:1170-1172`).

Pre-implementation search found the shared canonical operator in
`packages/ocean/legoesm/ocean/physics/bbl_adv.py` and analytic tests in
`tests/ocean/unit/test_bbl_adv.py`; solver integration was missing. The search
also found that its option-2 density gate used Wright derivatives while NEMO
calls `eos_rab`. It now uses the literal Roquet TEOS-10 alpha/beta polynomial
on Kbb bottom T/S at each column's Kmm geometric depth
(`src/OCE/TRA/trabbl.F90:342-353,415-454`). Static slope indices and the
minimum bottom-cell face thickness follow `tra_bbl_init:507-533`; the closed
three-leg tendency follows `tra_bbl_adv:243-284`. Solver wiring is at RK stage
3 only, matching `stprk3_stg.F90:468,588`, with Kbb tracers and Kmm volume.
The capped host-Euler wrapper was not used because its cap has no oracle
analogue.

The BBL scaling check is deliberately one-sided.  On legoESM's shipped kt=1
state, `max|utr_bbl|=max|vtr_bbl|=0`; its faithful and private BBL-disabled
candidates are bit-identical in kt=2 temperature (`max|delta T|=0 K`).  This
is **CONFIRMED INACTIVE on the legoESM side**.  NEMO's `utr_bbl` was not read,
so the corresponding oracle conclusion is only **PLAUSIBLE from the common
geometry**, not confirmed and not a two-model exoneration.  The former blanket
`CONFIRMED_EXONERATED_AT_KT2` label is retracted in both gate JSON and this
receipt.  A synthetic dense-shelf violation still proves that legoESM's real
stage-3 BBL path fires and conserves closed-exchange tracer content to its fp64
accumulation floor.

The structural-collapse round originally reported OVERFLOW kt=2 T at
`2.1810780026e-8` and u at `2.4321816013e-2`.  Those values came from replacing
the prognostic primary velocity with the distinct time-mean transport.  After
the frame correction, the regenerated faithful scores are T
`2.4003458243e-8`, u `4.1210339357e-6`, and SSH `1.2372313177e-7`.  The
`2.1810780026e-8` T value is retained only as the explicitly named
`wrong_prognostic_transport_frame` control; it is not the faithful card.
This is a real, previously uncredited improvement: the correctly framed
OVERFLOW instantaneous-u score moved from `2.4321816013e-2` to
`4.1210339357e-6`, a factor of about `5901`.

Focused verification for this round is **113 tests across seven explicitly
listed files**: 15 card + 6 WS-RK3 + 8 BBL + 54 barotropic-common + 17 config
footguns + 7 trajectory-gate + 6 first-divergence-gate tests. The first run
found two stale test expectations (a conservation tolerance that omitted the
large content scale and an EOS test that constructed the now-rejected
NEMO-polynomial/insitu pairing); both were corrected, then the identical
113-test scope passed. This count is separate from every historical count
above and is neither averaged nor merged with them.

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
4. **Distinct WS advecting transport — PLAUSIBLE PARTIAL OWNER of the
   remaining instantaneous residual, not owner of the frame regression.**
   NEMO replaces the live Kmm depth mean by `un_adv/hu` in `zFu/zFv` while
   retaining Kmm as the advected velocity
   (`stprk3_stg.F90:257-274,316,326-331`).  The private, one-sided control moves
   instantaneous u from `5.9807602992e-10` to `1.7120868406e-10 m/s`; this is
   scale-compatible but cannot establish confirmed two-model ownership.  It
   did not cause the later `1.135e-3` regression, which is fully assigned to
   the scoring-frame substitution below.
5. **Horizontal UP3 spatial recurrence — CONFIRMED EXONERATED.**  A local
   hypothesis initially read `dynadv_up3`'s formal `Kbb` name without
   reconciling its caller.  The live RK3 caller passes `Kmm` into both slots.
   Replaying `dynadv_up3.F90:141-212` against the baseline/no-advection NEMO
   pair matches the causal stage-2 tendency at `4.7314e-19` absolute.
6. **FCT two-step program — PLAUSIBLE PARTIAL OWNER.**  The shared selectable
   implementation transcribes the half-step upstream guess and averaged
   full-step low-order flux (`traadv_fct.F90:470-641`).  Within the otherwise
   identical faithful stage-reconcile arm it reduces LOCK T from
   `7.4358e-12` to `4.7464e-12 K`; it remains over bar.
7. **Vertical viscosity — CONFIRMED EXONERATED.**  NEMO's `rn_avm0=0` control
   changes stage 3 by `1.1363e-8 m/s`, but the paired legoESM `A_v=0` arm leaves
   the residual at `1.7120953e-10`, unchanged at the relevant scale.

### Scoring-frame retraction and Rule-1e reconciliation

**LOCK kt=2 u regressed `1.7121e-10 -> 1.135e-3 m/s` under the collapsed
scoring frame.**  This was a real regression in the gate, not a physical-model
owner.  `dynspg_ts.F90:509,641-642,843-847` proves that `un_adv` is a
secondary-weight, substep-time-mean transport, while `puu_b(Kaa)` is the
separately normalized primary velocity.  `stprk3_stg.F90:433-446` installs
that primary velocity in the RK3 after state.  The gate had compared NEMO
`uu(:,:,:,Nbb)` against a legoESM field whose depth mean had instead been
replaced by `Hu_avg/H`.

The corrected gate now has two frame-labelled rows:

- instantaneous: oracle `uu(:,:,:,Nbb)` and legoESM prognostic `u`, both 3-D
  C-grid U-face fields, same wet-face mask, elementwise L-infinity, no depth or
  time reduction;
- transport: oracle `un_adv` and legoESM `Hu_avg`, both 2-D C-grid U-face,
  vertically integrated substep-time-mean transports, same wet-face mask,
  elementwise L-infinity, no further reduction.

Restoring the primary velocity frame moves LOCK kt=2 u
`1.135e-3 -> 1.7120868406e-10 m/s`, reproducing the pre-regression value.  The
separate transport row is AT-BAR at `8.6736173799e-19 m2/s`.  The corrected
instantaneous baroclinic-anomaly residual is `1.7120866476e-10 m/s`; it is
reported independently and remains DEBT.  In the deliberately mismatched
control, the vertical spread is `8.7839880e-10 m/s` against the
`1.1353673e-3 m/s` maximum (ratio `7.7367e-7`), reproducing the reviewer's
depth-uniform signature.  NEMO's raw `puu_b(Kaa)` also matches the next-entry
thickness-weighted `uu(Nbb)` depth mean at `1.5510e-26 m/s`.  The raw NEMO
frame dump comes from a read-only `stprk3.F90` call immediately after
`stp_2D`; its rerun reproduced
all three prior step-entry dumps byte-for-byte, so the new diagnostic did not
alter the trajectory.

## Owner-exhausted trajectory continuation

The continuation flag is explicit and defaults off.  It preserves kt=2 as the
first-over-bar record and marks later rows as entering without an exact prefix.
LOCK grows to the regenerated values in the verdict table at kt=3.  OVERFLOW-zps
is exact at kt=1 and first diverges at kt=2 in T, u, and SSH; uniform S is
explicitly UNINFORMATIVE rather than corroborating alignment.  The OVERFLOW finer-
cadence oracle used the certified zps binary/configuration on one CPU process,
changing only `nn_itend=3` and `nn_stock=1`; hashes pin both namelists, output,
and all three dumps.  No certified configuration or binary was modified.

The OVERFLOW kt=2 T owner hunt registered the faithful absolute error first
(`4.8006916487e-7 K`), then ran one-variable private controls.  No control is
promoted to confirmed ownership because these are legoESM-side ablations only:

| one changed operand | candidate movement / faithful T error | resulting absolute T error | disposition |
|---|---:|---:|---|
| one-step rather than NEMO two-step FCT predictor | `0.0762` | `4.8007318e-7 K` | scale present, gap unchanged |
| frozen-final rather than Kmm tracer transport | `5360.85` | `2.5731008e-3 K` | decisively worse; Kmm remains required |
| omit per-stage primary-velocity correction | `0.2478` | `3.6109159e-7 K` | next scale-compatible suspect; ownership UNMEASURED |
| omit `un_adv/H` tracer-transport reconcile | `0.1217` | `5.3851318e-7 K` | worsens the gap |
| substitute time-mean transport as prognostic state | `0.0913` | `4.3621560e-7 K` | wrong-frame control; explains old `2.1811e-8` normalized score |

Thus the sweep stops honestly at OVERFLOW kt=2 T with the stage primary-
velocity correction ranked for the next internal stage dump.  BBL is only
legoESM-side CONFIRMED inactive and NEMO-side PLAUSIBLE inactive, as stated
above.  LOCK kt=3 was walked after the kt=2 owner register was exhausted; its
first continuation debts are T `9.1634e-12` and u `2.0540e-9`, with no new
owner label from the already-divergent prefix.

## Stage-primary control and contiguous kt=10 sweep

This round continues the evidence ledger on issue **#1699**.  It creates no
new public selector and changes no certified DINO or phase-1 testcase
configuration.  A private read-only return seam exposes WS-RK3 stage-1 or
stage-2 instantaneous velocity only after the full legoESM step has completed;
therefore the diagnostic cannot perturb any later stage.  Stage 3 is the
ordinary returned prognostic velocity.  NEMO's Kaa register is stage 1 = 3,
stage 2 = 2, stage 3 = 3 at the dumps immediately following
`src/OCE/stprk3_stg.F90:433-446`; the next entry is kt=2/Nbb=3 from the
`stprk3.F90` entry instrument.  Every stage row names both operands as
instantaneous 3-D C-grid U-face velocity and applies the identical wet-face
elementwise L-infinity reduction with no depth or substep-time average.

The registered OVERFLOW one-variable arm changes only
`stage_barotropic_correction`.  The faithful kt=2 T absolute error is
`4.8006916487e-7 K`; omission moves the candidate by `1.1897757091e-7 K`
(`0.247834` of the faithful residual) and reduces the error to
`3.6109159396e-7 K`, but does not clear the bar.  This is
**PLAUSIBLE_CONTRIBUTOR_NOT_OWNER** on scale alone.  It is not a proposal to
remove the correction: the direct stage-1 u error worsens from
`6.3674137757e-8` to `4.5029698611e-2 m/s`.  The gate records the correction as
**CONFIRMED_REQUIRED** and carries an explicit cancellation warning.  The
remaining OVERFLOW T owner is UNMEASURED.

LOCK's instantaneous-u tail was tested with exactly two one-variable arms, as
registered.  Omission of the stage primary-velocity correction moves the
candidate by `4.2989715926e-10 m/s` and worsens the residual from
`1.7120868406e-10` to `5.7155454103e-10 m/s`.  Omission of momentum-transport
reconciliation moves it by `6.1720004452e-10 m/s` and worsens it to
`5.9807602992e-10 m/s`.  Both movements exceed the faithful tail, so the scale
test is sensitive; neither arm improves or clears the bar.  Both are
**REFUTED_AS_PRIMARY_OWNER**, and after the required two-arm limit the owner is
registered **UNMEASURED_AFTER_TWO_ARMS**.  No third arm was run.

The new NEMO configurations are isolated
`tests/LOCK_EXCHANGE_OMIP_L1_P3` and `tests/OVERFLOW_OMIP_L1_P3`, compiled with
the same conda/gfortran toolchain and pinned `key_qco`, vertical-coordinate,
and `key_RK3` keys.  The latter was cloned from OVERFLOW solely to carry the
same stage/entry instrument; the first generated cpp file incorrectly picked
up `key_xios`, so that key was removed before the successful build.  Both CPU
runs set `nn_itend=10`, write the final restart at 10, and completed normally.
Their kt=1--3 entry dumps are byte-identical to the earlier certified dumps.
The copied instrument writes all ten entry states and only the registered kt=1
stage/RHS/transport records.  The legoESM harness requested a pinned GPU, but
CUDA device discovery returned `CUDA_ERROR_NO_DEVICE` in this session; it ran
fp64 on CPU instead.  Candidate T/u dtypes are printed as `float64` in both
stage and trajectory JSONs.

The table reports normalized wet-point L-infinity error against the exact
NEMO entry state.  `v` is omitted because both tanks have no active
meridional face and every row is UNMEASURED.  LOCK S at kt=2 and OVERFLOW S at
kt=2 are UNINFORMATIVE because the oracle field is uniform; later S values are
measured.  This explicit continuation does not restore an exact prefix after
kt=2.

| case | kt | T | S | instantaneous u | SSH |
|---|---:|---:|---:|---:|---:|
| LOCK | 1 | `0` | `0` | `0` | `0` |
| LOCK | 2 | `1.58214e-13` | `0` (UNINFORMATIVE) | `1.71209e-10` | `4.78214e-28` (UNINFORMATIVE) |
| LOCK | 3 | `9.16340e-12` | `2.03012e-16` (UNINFORMATIVE) | `2.05400e-9` | `1.35525e-18` |
| LOCK | 4 | `4.84606e-11` | `2.03012e-16` | `4.85319e-9` | `4.92999e-14` |
| LOCK | 5 | `1.50836e-10` | `4.06024e-16` | `8.77056e-9` | `5.03171e-13` |
| LOCK | 6 | `3.61405e-10` | `4.06024e-16` | `1.37782e-8` | `2.44491e-12` |
| LOCK | 7 | `7.37110e-10` | `2.03012e-16` | `1.98316e-8` | `8.23068e-12` |
| LOCK | 8 | `1.34649e-9` | `4.06024e-16` | `2.68666e-8` | `2.21790e-11` |
| LOCK | 9 | `2.26941e-9` | `4.06024e-16` | `3.47956e-8` | `5.14107e-11` |
| LOCK | 10 | `3.59674e-9` | `4.06024e-16` | `4.35051e-8` | `1.06780e-10` |
| OVERFLOW | 1 | `0` | `0` | `0` | `0` |
| OVERFLOW | 2 | `2.40035e-8` | `2.03012e-16` (UNINFORMATIVE) | `4.12103e-6` | `1.23723e-7` |
| OVERFLOW | 3 | `2.86506e-7` | `4.06024e-16` | `1.54030e-5` | `4.47222e-6` |
| OVERFLOW | 4 | `9.69750e-7` | `4.06024e-16` | `3.15067e-5` | `2.40002e-5` |
| OVERFLOW | 5 | `1.98970e-6` | `4.06024e-16` | `5.69749e-5` | `4.45039e-5` |
| OVERFLOW | 6 | `3.25520e-6` | `4.06024e-16` | `8.63738e-5` | `4.47386e-5` |
| OVERFLOW | 7 | `4.92497e-6` | `6.09037e-16` | `1.08575e-4` | `6.65274e-5` |
| OVERFLOW | 8 | `7.22005e-6` | `1.01506e-15` | `1.27563e-4` | `8.33689e-5` |
| OVERFLOW | 9 | `1.01233e-5` | `1.01506e-15` | `1.56818e-4` | `8.07337e-5` |
| OVERFLOW | 10 | `1.34856e-5` | `8.12049e-16` | `1.97687e-4` | `8.21267e-5` |

The first-over-bar entry remains kt=2: LOCK in T/u, OVERFLOW in T/u/SSH.
The continuation therefore measures growth from an already-divergent prefix;
it makes no trajectory-match claim.

## Primary-transport arm and kt=60 continuation

The next source read found a branch the card exercised but legoESM did not:
OVERFLOW resolves `ln_dynadv_vec=F` with variable volume, so NEMO accumulates
the primary external mode as `wgtbtp1*ua_e*hu_e` rather than velocity
(`src/OCE/DYN/dynspg_ts.F90:823-834`), then divides by the Kaa face depth only
after the primary SSH average is complete (`:956-979`).  This is not a NEMO
switch and therefore is not a legoESM selector: the canonical NEMO RK3
flux-form scheme identity now executes it unconditionally.  Its only ablation
is the private test hook used by the owner gate.

The stage-by-stage scaling result refutes this term as the remaining kt=2 T
owner.  Relative to the faithful `4.8007010989e-7 K` absolute T residual, the
velocity-average ablation moves T by only `9.4502184e-13 K` (`1.97e-6` of the
residual).  It is **REFUTED_AS_PRIMARY_OWNER**.  The source-required arm does,
however, improve the final instantaneous-u score from `4.1210339357e-6` to
`3.9560471201e-6 m/s`.  Stage-1/2/3 instantaneous-u errors are now
`2.2873001e-7`, `1.0007265e-6`, and `3.9560471e-6 m/s`; the legacy primary
arm gives `6.3674138e-8`, `8.3571138e-7`, and `4.1210339e-6 m/s`.  The required
stage correction remains a **PLAUSIBLE_CONTRIBUTOR_NOT_OWNER** for T: its
one-variable omission explains `24.7836%` but leaves `3.6109159e-7 K` and
simultaneously destroys stage-u alignment.  The other registered arms are
refuted as primary owners.  The remaining roughly 75% is therefore
**UNMEASURED_AFTER_REGISTERED_ARMS**; the owner hunt is exhausted, not cleared.

Both isolated NEMO configurations were rebuilt with the same conda/gfortran
toolchain after extending only the read-only entry-dump bound from 10 to 60.
Fresh CPU runs used `nn_itend=nn_stock=60`, completed normally, and produced
60 entry plus 60 barotropic-frame records per case.  Their kt=1--10 entry
records are byte-identical to the certified short runs.  The current kt=60
normalized wet-point L-infinity scores are:

| case | T | S | instantaneous u | SSH |
|---|---:|---:|---:|---:|
| LOCK | `3.95492e-6` | `2.84217e-15` | `7.41114e-6` | `3.84974e-6` |
| OVERFLOW-zps | `7.96523e-4` | `3.65422e-15` | `4.13339e-3` | `1.98816e-4` |

The gate keeps the reviewer-requested two-line computation explicit:
`ratios = errors[1:] / errors[:-1]`, followed by
`ratios_monotone_decreasing = all(diff(ratios) < 0)`.  Reconciliation changes
the earlier prose: full-history ratios decrease monotonically for LOCK T/SSH
and OVERFLOW T, but not for LOCK u or OVERFLOW u/SSH.  On the registered tail
window (kt=32--60), log-log fits are preferred over semilog fits for LOCK
T/u/SSH (`p=3.545/3.306/5.079`) and OVERFLOW T/u (`p=1.707/1.697`).
OVERFLOW SSH is bounded/oscillatory with negative fitted rate and `p=-0.675`.
No field selects the exponential-fit/open-mode label: the measured evidence is
polynomial accumulation or bounded oscillation, not an amplifying mode.
The earlier reviewer's `p≈1.5--3.9` range holds for four of the five growing
tail series; LOCK SSH is the measured exception and is reported, not averaged.

The pre-primary-transport kt=2--10 OVERFLOW rows immediately above are retained
as historical round output.  The current gate JSON and the kt=2/3 verdict rows
at the top of this receipt supersede them.

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

For this continuation, the exact focused invocation ran **58 tests across six
files**: `5 WS tracer + 11 card + 12 flux-form momentum + 17 config footguns +
6 first-divergence gate + 7 trajectory gate`; all 58 passed.  This count is
separate from, and does not revise or average, either historical count above.

For the frame-correction round, the final focused invocation ran **116 tests
across seven explicitly listed files**: `15 card + 6 WS-RK3 + 8 BBL + 54
barotropic-common + 17 config-footguns + 8 trajectory-gate + 8
first-divergence-gate`; all 116 passed.  This is the actual collected
breakdown, not an averaged historical count.

For this stage/kt=10 continuation, the focused invocation ran **43 tests across
six explicitly listed files**: `2 stage-sweep gate + 8 trajectory gate + 8
first-divergence gate + 3 EOS gate + 7 WS-RK3 + 15 testcase-card`; all 43
passed in the final post-edit run.  This is the collected breakdown for this
round, not an average or substitution for any historical count above.

For the primary-transport/kt=60 closure, the final focused invocation ran
**48 tests across five files**: `9 trajectory-gate + 2 stage-sweep-gate + 8
WS-RK3 + 15 testcase-card + 14 AB3/filter`; all 48 selected tests passed.  Two
pre-existing NEMO-gyre model-step tests in the AB3 file were explicitly
deselected because they construct the validation-rejected
`pgf_quadrature="nemo_trapezoid" + pgf_scheme!="nemo_sco"` Frankenstein
configuration and are outside these two certified cards.  The first run's one
in-scope failure was the new live-arm assertion using `>1e-12 K` against the
measured `9.4502e-13 K`; it was corrected to the non-vacuous `>0` assertion
without changing any science code, then the exact 48-test scope passed.

Full artifact hashes are committed in
`nemo_testcases_l1_phase3_artifacts.sha256`.  The trajectory and diagnostic
JSONs are external run products under
`/data/abyssal/dbalwada/nemo-testcases-l1/phase3/`; their hashes make the
receipt reproducible without committing quota-heavy binary dumps.

## Stage-composition round: cen2 tracer stages, live HPG operands, per-stage vertical UP3

Takeover of the codex session (out of credits mid-round) on branch
`fidelity/nemo-testcases-l1-codex`.  Commits `3a68e43338a4` (frame registry,
git-SHA stamps, planted-control exit status, Rule-8 disclosure of the literal
flux-form external update), `614bed818bb4` (the three WS-RK3 composition
fixes below + gate port), `c8f506a69545` (trajectory-gate SHA stamp).  Every
number below is fp64 on CPU with `set_policy(PrecisionPolicy.fp64())` and
printed `float64` candidate dtypes; every gate JSON carries
`legoesm_git_sha`.

### Rule 1e reconciliation: which implementation was wrong, and why

Two lines had converged on the same owner candidate (the momentum-stage HPG
operands frozen at Kbb), and codex's first Arm A had made things WORSE
(stage-2 baroclinic U `7.80e-7 -> 1.77e-6 m/s`, kt=2 `3.31e-6 -> 6.70e-6`).
Neither number was recorded until the disagreement was explained:

1. **A quantity both probes had to agree on exactly** (Rule 1e step 2): the
   stage-1 Kaa tracer that codex's Arm A handed to the stage-2 `eos+dyn_hpg`
   call, against NEMO's stage-1 dump.  `max|T_lego - T_nemo| = 3.452e-4 K`
   = 100 % of NEMO's own stage-1 change; the front cells' increments were
   `0x` and `2x` NEMO's (i=20: `+2.3e-11` vs `-3.45e-4`; i=21: `-6.90e-4`
   vs `-3.45e-4`).  That is the first-order-upwind signature of the FCT
   low-order predictor, not NEMO's centred stage.
2. **The oracle's source** (Rule 0): `traadv.F90:281-282` forces
   `ll_dofct = .FALSE.` for `kstg /= 3`, and `:361-364` dispatches `np_FCT`
   to `tra_adv_cen(nn_fct_h, nn_fct_v)` — 2nd-order centred
   (`traadv_cen.F90:140-149, 196-210`) — at stages 1 and 2.  legoESM's WS
   tracer program ran the full FCT limiter at every stage.
3. **The instrument** (the hunt's `hpg_sco` replay): legoESM's own momentum
   tendency on NEMO's stage-1 Kaa operands (`T_s1, S_s1, ssh_s1`, u=0)
   agrees with the `dynhpg.F90:340-390` replay to `1.7e-16 m/s^2`
   (rest control `9.2e-17`), i.e. Arm A's operand plumbing (density from
   stage T/S at the live `gdept(1+r3t)`, `e3w(1+r3t)`, `gdept_z0 = gdept -
   ssh`, HYB `ssh = ssha/3, ssha/2`) was right.  Fed codex's operands, the
   replay predicts a stage-2 error of `1.766e-6 m/s` — exactly the measured
   `1.766e-6`.

**Verdict: codex's implementation was the wrong one, through an upstream
defect it inherited rather than introduced** — legoESM's stage-1/2 tracer
operator (FCT instead of `tra_adv_cen`).  The hunter's source readings and
replay stand.  With `cen2` at stages 1-2 the stage-1 T operand is
bit-exact (`0.0 K`), S `7.1e-15`, ssh `3.5e-15 m`; the stage-2 T operand is
`2.83e-8 K` (normalized `1.4e-9`, DEBT), see the register below.

### What NEMO does at each stage vs what legoESM now does

| item | NEMO 5.0.2 (`key_RK3`, `key_qco`, flux-form UP3, FCT2) | legoESM `rk3_ws` after `614bed818bb4` |
|---|---|---|
| tracer stages 1-2 | `tra_adv_cen(2,2)` (`traadv.F90:281-282,361-364`) | `centered` flux pair, same stage transport (`_nemo_ws_rk3_tracer_pair_step`) — MATCH (stage-1 T exact) |
| tracer stage 3 | `tra_adv_fct(Kbb,Kmm,Kaa)` with the RK3 two-step predictor | unchanged FCT2 path |
| stage-2/3 EOS+HPG operands | `eos(ts,Kmm)`, `dyn_hpg(Kmm)`; Kmm = previous stage Kaa after the `stprk3.F90:218,224` swaps; ssh at HYB N+1/3, N+1/2 (`stprk3_stg.F90:146,209,317-320`) | stage tracers built interleaved with the momentum stages; `compute_frozen_geom_density` on the stage T/S/eta — MATCH (replay 1.7e-16) |
| dyn_adv vertical UP3 | every stage on `zFw` from the stage Kmm transport (`stprk3_stg.F90:315,331-334`; `dynadv_up3.F90:239-358`); stage-3 `ww` split by `wAimp` (`:298-302`) | `nemo_up3_vertical_momentum_advection` in every stage RHS on the stage Kmm velocity and explicit `ww`; implicit share to the one ZDF matrix — the once-per-step post-hoc application is deleted |
| stage transport triplet | one `(zFu,zFv,zFw)` per stage for dyn_adv and tra_adv (`stprk3_stg.F90:257-304`) | `_nemo_ws_stage_transport` (hoisted), shared by momentum and tracer stages |

No public selector was added; NEMO has no switch at any of the three sites.
The private hooks `freeze_stage_hpg_operands`, `omit_stage_vertical_up3`,
`expose_tracer_stage` exist for the gate's one-variable arms only.

### Arm outcomes against the frozen predictions

Predictions were frozen in
`nemo_testcases_l1_overflow_stage_composition_preregister.md` (Arm A
without B; Arm B with A).  The stage sweep gate now carries both arms and
labels them only after the scaling check:

| arm | stage 1 | stage 2 | stage 3 = kt=2 entry | kt=2 T (K) |
|---|---:|---:|---:|---:|
| faithful (A + B) | `3.7618e-12` | `6.9081e-11` | `2.5984e-07` | `2.5485e-07` |
| A only (omit per-stage vertical UP3) | `3.7618e-12` | `1.6478e-07` | `6.1803e-07` | `2.5485e-07` |
| B only (frozen Kbb HPG operands) | `3.7618e-12` | `9.4460e-07` | `3.6884e-06` | `2.3672e-07` |

- **Arm A (live stage HPG operands), measured on the A-only arm:** stage 2
  `7.798e-7 -> 1.648e-7 m/s` (predicted `~1.65e-7`, within factor 2), kt=2
  `3.311e-6 -> 6.180e-7` (predicted `<= 7e-7`): **CONFIRMED**.
- **Arm B (vertical UP3 in every stage RHS), measured on the faithful arm:**
  stage 2 `1.648e-7 -> 6.908e-11` (predicted `~1e-10`), stage 3 = kt=2
  `-> 2.598e-7` (predicted `~2.6e-7`): **CONFIRMED**.
- **B-only control (frozen Kbb operands, one variable from faithful):**
  stage 2 `9.446e-7`, kt=2 `3.688e-6` — the replay's `E_hpg` alone
  (`9.448e-7 / 3.68e-6`); the ported replay's predicted/measured stage-2
  movement is `1.0002` and, fed legoESM's own stage-1 operands, it
  predicts `5.20e-19 m/s` — the operand is no longer the owner.
- Stage 1 (`3.76e-12`) is the missing qco stage factor
  `(1+r3u(Kmm))/(1+r3u(Kaa))` (`stprk3_stg.F90:389-394`), measured by the
  hunt at `3.8e-12` — PLAUSIBLE, below the current stage-3 floor by 5
  orders, not touched.

The mechanism of the stage-3 share of Arm B is **PLAUSIBLE, not
decomposed**: the once-per-step post-hoc increment measured `1.18e-8 m/s^2`
max inside the real step (`7.5e-12` from the `dynadv_up3` transcription on
the same velocity with NEMO's dumped stage-3 `zFw`), so moving it into the
stage-3 RHS cannot by itself account for the `6.18e-7 -> 2.60e-7` stage-3
movement of the omit arm; the stage-2 term (`3.3e-8 m/s^2 x 5 s`) and its
downstream effect through the corrected stage-2 velocity are the measured
remainder.  The hunt's stage-3 `E_vert = 5.0e-7` attribution is therefore
REVISED to "the two per-stage vertical terms together", not the stage-3 term
alone.

### kt=2 entry and the kt=2..10 trajectory (normalized L-inf, both cases)

OVERFLOW-zps kt=2: T `2.40034479e-08 -> 1.27424627e-08`, U `3.31108168e-06
-> 2.59844026e-07`, SSH `1.04916076e-14 -> 1.04916076e-14`.  "Before" is
the round-4 literal-flux-form baseline (`overflow_kt1_10_flux_gate.json`,
same barotropic arm), so each column pair differs only in this round's
stage composition.

OVERFLOW-zps:

| kt | T before | T after | U before | U after | SSH before | SSH after |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | `2.40034479e-08` | `1.27424627e-08` | `3.31108168e-06` | `2.59844026e-07` | `1.04916076e-14` | `1.04916076e-14` |
| 3 | `2.86529500e-07` | `1.63310582e-07` | `9.22615336e-06` | `3.37673071e-06` | `3.66567551e-09` | `3.75492262e-09` |
| 4 | `9.70021745e-07` | `5.90530377e-07` | `1.50603834e-05` | `3.37636512e-06` | `1.07010798e-06` | `1.07119102e-06` |
| 5 | `1.99077587e-06` | `1.16912537e-06` | `2.93540699e-05` | `3.28700281e-06` | `1.48167577e-05` | `1.48122887e-05` |
| 6 | `3.25827731e-06` | `1.74603452e-06` | `5.59331066e-05` | `9.27687843e-06` | `4.00067365e-05` | `4.00019841e-05` |
| 7 | `4.93252050e-06` | `2.39742496e-06` | `7.99314090e-05` | `1.15106208e-05` | `4.55653806e-05` | `4.55757512e-05` |
| 8 | `7.23598252e-06` | `3.29132932e-06` | `9.16463001e-05` | `2.66872742e-05` | `7.97549653e-05` | `7.97135279e-05` |
| 9 | `1.01527667e-05` | `4.40403288e-06` | `1.07033146e-04` | `3.21886997e-05` | `7.29954633e-05` | `7.29592804e-05` |
| 10 | `1.35356455e-05` | `5.55186262e-06` | `1.40474350e-04` | `2.64522041e-05` | `9.24576527e-05` | `9.24110227e-05` |

LOCK_EXCHANGE-zco:

| kt | T before | T after | U before | U after | SSH before | SSH after |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | `1.58214182e-13` | `1.62773498e-13` | `1.71208363e-10` | `2.13880413e-10` | `4.78213792e-28` | `4.78213792e-28` |
| 3 | `9.16339597e-12` | `5.20626505e-12` | `2.05231841e-09` | `2.13879378e-10` | `1.38235777e-18` | `1.35525272e-18` |
| 4 | `4.84606725e-11` | `3.59465938e-11` | `4.84941135e-09` | `2.13851856e-10` | `2.41777084e-17` | `2.16027283e-17` |
| 5 | `1.50835907e-10` | `1.21776485e-10` | `8.77140353e-09` | `2.13746464e-10` | `6.21324131e-16` | `6.41580871e-16` |
| 6 | `3.61406312e-10` | `3.05194329e-10` | `1.38075895e-08` | `2.13440690e-10` | `3.66530956e-15` | `3.83130144e-15` |
| 7 | `7.37113008e-10` | `6.40530355e-10` | `1.99444732e-08` | `2.12733880e-10` | `1.42910857e-14` | `1.56074155e-14` |
| 8 | `1.34649660e-09` | `1.19372411e-09` | `2.71657579e-08` | `2.11363624e-10` | `4.41092475e-14` | `4.73124144e-14` |
| 9 | `2.26942305e-09` | `2.04205328e-09` | `3.54524968e-08` | `2.08895765e-10` | `1.18974275e-13` | `1.26009229e-13` |
| 10 | `3.59676472e-09` | `3.27381405e-09` | `4.47830601e-08` | `2.04777155e-10` | `2.71415052e-13` | `2.85172493e-13` |

LOCK kt=2 is faithful-but-slightly-worse (T `1.582e-13 -> 1.628e-13`, U
`1.712e-10 -> 2.139e-10`) while its kt=3..10 U error stops growing
(`4.478e-8 -> 2.048e-10` at kt=10, flat at `~2.1e-10` from kt=2 on) and T
improves at every later step; disclosed, not reverted (Rule 8).  The flat
`2.1e-10` LOCK U residual is the next LOCK owner and is UNMEASURED.

### Statistical scorer (6120 steps, fp64 and fp32, CPU)

Same scorer, same NEMO FCT2/FCT4 spread, same-revision fp32 floor, both arms
complete with every-step finite checks at `c8f506a69545` (fp64 `431.88 s`, fp32
`226.32 s` wall).  `OUTSIDE` count `4 -> 3`, `WITHIN-SCHEME-SPREAD` `2 -> 3`; the
final temperature histogram crosses into the NEMO scheme spread, `temperature_linf`
improves but stays `OUTSIDE`, the water-mass census and instantaneous-U rows worsen
slightly, and both plume rows move away from NEMO while staying far inside the
scheme spread.  The frozen prediction target ('the 4 OUTSIDE rows') is therefore
met for one row and NOT met for three: the kt=2 initiator is fixed, the 6120-step
statistics are owned mostly by something else (Rule 3: the early-step gain does
not survive the chaotic window).

| metric | before | after | fp32 floor before | fp32 floor after | NEMO spread | verdict before | verdict after |
|---|---:|---:|---:|---:|---:|---|---|
| final_temperature_histogram_tv | `0.0537796` | `0.0423046` | `0.0237045` | `0.0194058` | `0.044231` | OUTSIDE | WITHIN-SCHEME-SPREAD |
| final_water_mass_census | `0.0150395` | `0.0169753` | `0.00349422` | `0.00157409` | `0.00336209` | OUTSIDE | OUTSIDE |
| instantaneous_u_linf | `0.969889` | `0.999211` | `0.456889` | `0.402734` | `0.672694` | OUTSIDE | OUTSIDE |
| plume_descent_m | `0.904647` | `16.9554` | `0.100003` | `0.0087228` | `1499.62` | WITHIN-SCHEME-SPREAD | WITHIN-SCHEME-SPREAD |
| plume_front_km | `1.00947` | `4.04936` | `0.561019` | `0.0837761` | `121.931` | WITHIN-SCHEME-SPREAD | WITHIN-SCHEME-SPREAD |
| temperature_linf | `0.394129` | `0.379492` | `0.0809865` | `0.140318` | `0.356744` | OUTSIDE | OUTSIDE |

### Retractions and revisions (Rule 11)

- codex's first Arm-A implementation is RETIRED as a measurement of the
  operand time level: it measured the FCT-vs-cen2 tracer-stage defect
  through the HPG.  Its plumbing survives unchanged in the fix.
- The hunt's stage-3 `E_vert` attribution is revised as above (the term was
  present post hoc and measured `1.18e-8 m/s^2`, not `4.95e-8`); its
  stage-2 attribution and both source readings are CONFIRMED by the gate.
- The `stage_operand_ownership` block written by codex evaluated Arm A's
  frozen prediction on the faithful arm; with Arm B landed that arm is A+B,
  so the prediction is now evaluated on the omit-vertical arm (A only),
  exactly as preregistered.

### Next owner (PLAUSIBLE, UNMEASURED): kt=2 T `2.55e-7 K`

The T residual is untouched by both arms (faithful `2.548e-7 K`, B-only
`2.367e-7 K`, A-only `2.548e-7 K`), so it lives in the tracer program
itself.  The stage-2 tracer operand already carries `2.83e-8 K` with the
stage-1 velocity at `3.8e-12 m/s`; the remaining difference in that stage
is the stage face thickness: NEMO's `e3u(Kmm) = e3u_0 (1 + r3u)` with
`r3u` the `e1e2`-weighted mean of the two columns' `r3t`
(`domqco.F90:218-222`), while `_nemo_ws_stage_transport` takes the min of
the two stretched cell thicknesses.  At the OVERFLOW front (`r3t = +-3.3e-2
/ 500`) the two differ by `6.6e-5` relative, times the stage change
`9.5e-4 K` gives `~6e-8 K` — scale-compatible with the measured `2.8e-8 K`
operand error and with the kt=2 `2.5e-7 K`.  legoESM already has the
literal QCO face-thickness helper (`nemo_qco_live_face_thicknesses`, used
by the `wzv_call2` literal path); routing the stage transports through it
is the registered next arm.  Not run here (outside this round's two arms).

### Provenance

| artifact (under `/data/abyssal/dbalwada/nemo-testcases-l1/`) | SHA256 |
|---|---|
| `barotropic_walk/review_round/frame_gate_hold_round_ab6ca17b6.json` | `59c0a4e056e87d74af04937a8638dcc279840c6f7ae7e78abf4499e6c7d43e42` |
| `barotropic_walk/review_round/pytest_barotropic_gate_hold_round.log` | `64a815b7a0608d37ab43bcd3a7377d8a559ecd7692fd45baaf9818f0ce66b62a` |
| `stage_composition/legoesm/overflow_zps/fp32/metadata.json` | `dfaa8b66f927bd3e6a75bc9aae79771ad1390f29f00fbe91ad213c9fedc28207` |
| `stage_composition/legoesm/overflow_zps/fp32/states.npz` | `c73306a060b8cd74bdabc56465eefa4b6414d403ea0cc1d9187bf6b2d20de3f7` |
| `stage_composition/legoesm/overflow_zps/fp64/metadata.json` | `6470d575cca23987c94b73696bc2f811e3f65f099d9086931ace43247733150a` |
| `stage_composition/legoesm/overflow_zps/fp64/states.npz` | `0599482dd41dba259494cc277c4ad36dabbb2c7bdd370177d5430226038ccdff` |
| `stage_composition/lock_stage_sweep_gate_kt2.json` | `a3716cb258e33da4143d4da3fc5bd0fde2ebedc94872a4606a51c766809f5a4c` |
| `stage_composition/lock_trajectory_gate_kt10.json` | `4332b47664820b6a67769f17ea4de5c1d2efc449db7070d8a3f3564d350788c6` |
| `stage_composition/overflow_stage_sweep_gate_kt2.json` | `5bdcd995da66758199f3ad2db5cc44b97f42e093962476c2b86624d676c5dc9c` |
| `stage_composition/overflow_statistics.json` | `a9aca2d78dae62e8de4da4edd61233b101160637ed769a476254841f3c05df26` |
| `stage_composition/overflow_trajectory_gate_kt10.json` | `bcc8a68d26c918954329173fe955aec82155502db5b64743cac9ab963ab59377` |
| `stage_composition/overflow_trajectory_gate_kt60.json` | `086dbd6fc7a61ea328ecda692d7ef3491092f9fc6127db5a661d5f81f5967712` |

Focused CPU/fp64 tests: **161 passed across eleven explicitly listed files** at `c8f506a69545`: 7 OVERFLOW barotropic gate + 5 stage-sweep gate + 9 trajectory gate + 10 WS-RK3 tracer + 15 full statistics + 20 adaptive-implicit + 12 stability probe + 15 testcase card + 19 vertical momentum scheme + 10 rk3_ws/mxl3 + 39 leapfrog integrator (corrected from a miscounted 163/9: the barotropic-gate file collects 7 tests at that revision, not 9).  Four tests in the last two files fail identically at the pre-takeover revision `5104de943784` (verified in a throwaway worktree): two construct the validation-rejected `nemo_trapezoid` + non-`nemo_sco` configuration and two the decoupled `rk3_ws` momentum/tracer program; both are construction-time `ValueError`s outside these cards and are not touched by this round.

## Loud UNMEASURED register

- owner of the final LOCK `1.7121e-10` wet-point L-infinity u residual after
  two scale-sensitive one-variable arms; status is
  UNMEASURED_AFTER_TWO_ARMS, while live-Kmm UP3 and vertical viscosity remain
  exonerated;
- individual NEMO/legoESM FCT limiter coefficients and antidiffusive fluxes;
- NEMO `utr_bbl` at kt=1 (legoESM BBL is confirmed inactive; NEMO is only
  geometrically plausible inactive);
- OVERFLOW-zps kt=2 U is now MEASURED and owned (stage-composition round
  above: live stage HPG operands + per-stage vertical UP3 + cen2 tracer
  stages, `3.311e-6 -> 2.598e-7`); the remaining `2.598e-7` U (stage 3),
  the `3.8e-12` stage-1 qco factor, and the kt=2 T `2.55e-7 K` (registered
  next arm: NEMO `e3u(Kmm) = e3u_0(1+r3u)` stage face thickness,
  `domqco.F90:218-222`) stay UNMEASURED; the kt>=3 SSH walk is untouched;
- LOCK U after this round sits flat at `2.1e-10` from kt=2 to kt=10 (was
  growing to `4.5e-8`); its owner is UNMEASURED;
- term-level explanation of the measured kt=4--60 polynomial accumulation in
  either case;
- long-trajectory phenomenology and statistical equivalence beyond 60 steps.
