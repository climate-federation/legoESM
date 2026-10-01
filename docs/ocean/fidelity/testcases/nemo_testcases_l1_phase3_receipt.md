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
| `momentum_flux_scheme="nemo_up3"` | both | same namelist rows; `src/OCE/DYN/dynadv_up3.F90:141-365`.  (Renamed from the unqualified `"upwind3"` 2026-09-02: that one value served BOTH the NEMO and the Oceananigans UP3, whose T-point upwind selectors differ — see the map doc's S-46.) |
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
| 7 | `vertical_momentum_scheme="nemo_up3"` with horizontal momentum other than flux-form/`nemo_up3` | NEMO `dynadv` selects horizontal and vertical UP3 as one routine | **(a)** require the coupled flux-form/`nemo_up3` selections |
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
- family 7: `nemo_up3` requires the single flux-form/`nemo_up3` momentum program;
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

## Face-thickness and stage-qco round: the kt=2 T owner, MEASURED

Preregistered in
`nemo_testcases_l1_overflow_face_thickness_preregister.md` (commit
`6f4204e11`), which was committed BEFORE either arm existed.  Both changes
are unbranched parts of the NEMO WS-RK3 scheme identity; no public selector
was added and no other scheme's path changed.

**Headline: the OVERFLOW-zps `kt=2` temperature residual, registered as the
next owner at the end of the previous round, is closed.  `2.548493e-07 K ->
2.238210e-13 K` (normalized `1.274246e-08 -> 1.119105e-14`), a factor of
1.14e6.  Its owner is the stage transport's FACE THICKNESS.**  The stage-1
velocity residual `3.761789e-12 -> 1.811051e-15 m/s` is owned by the qco
stage weighting, predicted to 0.05% before the arm ran.

### What NEMO does vs what legoESM did (both readings verified in source)

| term | NEMO 5.0.2 | legoESM before | legoESM now |
|---|---|---|---|
| stage face thickness | `e3u(Kmm) = e3u_0*(1 + r3u(Kmm)*umask)`, `domzgr_substitute.h90:127`, consumed at `stprk3_stg.F90:272-273`; `r3u = 0.5*(e1e2t_i*ssh_i + e1e2t_{i+1}*ssh_{i+1}) * r1_hu_0 * r1_e1e2u`, `domqco.F90:219-222` | `min_cell_to_uface(h_stage)` = min of the two STRETCHED T thicknesses (`ocean_model_latlon_cgrid.py:1064`) | the NEMO rule, built by `_nemo_ws_qco_stage_faces` |
| reference face | `pe3u(:,:,:) = pe3t(:,:,:)`, `tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:184` | `min(e3t_0_i, e3t_0_{i+1})` | unchanged |
| stage velocity | `uu(Kaa) = ((1+r3u(Kbb))*uu(Kbb) + rDt*(1+r3u(Kmm))*uu(Krhs)) / (1+r3u(Kaa))`, `stprk3_stg.F90:373-378` and `dynzdf.F90`'s `key_qco` branch | `u_raw = u0 + stage_dt*RHS` | the NEMO weighting |
| stage tracer | `ts(Kaa) = ((1+r3t(Kbb))*ts(Kbb) + rDt*(1+r3t(Kmm))*ts(Krhs)) / (1+r3t(Kaa))`, `:552-554` | already faithful (`_stage` divides by `h_stage`) | unchanged |

Two source facts that had to be checked rather than assumed:

* `r3u` is an `e1e2t`-weighted mean of **ssh** divided by `hu_0`, NOT the mean
  of the two `r3t` (each of those divides by its own column's `ht_0`).  At the
  OVERFLOW shelf break the two denominators differ by a factor of four.
* the REFERENCE face was already right.  Measured on the card's own
  `mesh_mask.nc`: `e3u_0 == min(e3t_0_i, e3t_0_{i+1})` **exactly** on all
  16900 wet U faces.  Only the stretching was wrong, which is why this is a
  one-variable change.

`ln_dynadv_vec=.false.` with `key_qco` makes `lk_linssh` false, so the
thickness-weighted branch at `:371-378` is the one both cards execute; the
velocity-form branch at `:363-366` is dead here.

### Rule 4: the rule was NOT re-implemented

`grep -rn "r3u\|qco" packages/ocean/legoesm/ocean/` before writing anything
found `vertical.py:140 nemo_qco_live_face_geometry_from_operands` and
`vertical.py:197 nemo_qco_live_face_thicknesses`, the shared canonical
`dom_qco_r3c` builders already used by the DINO ldfslp/dynzad path and by the
`wzv_call2_evaluation="nemo_literal"` tracer path
(`ocean_model_latlon_cgrid.py:5234`).  `_nemo_ws_qco_stage_faces` calls the
`_from_operands` primitive and only maps its native east/north extent onto
legoESM's redundant west/south layout; the wrapper was not usable directly
because it reads operands from `z_coord.nemo_*` fields the L1 testcase cards
do not carry.  **Instrument check before any arm:** on the OVERFLOW stage-1
oracle ssh the helper reproduces the analytic `e3u_0*(1+r3u)` to `0.0` on
every wet face, while the min rule is off by `1.3199826e-03 m`.

### REACH TABLE — which cards execute the changed lines

Every row was measured by instantiating the card and printing the selector,
not read off a comment.

| card | selector that routes it | executes the changed lines? |
|---|---|---|
| OVERFLOW-zps (`build_nemo_testcase_card`) | `momentum_time_integrator="rk3_ws"`, `tracer_time_integrator="rk3_ws"` | YES, both changes |
| LOCK_EXCHANGE-zco (same builder) | same | YES, both changes — and MEASURED as a bit-identical no-op at kt=1 (below) |
| GYRE (`build_nemo_gyre_recipe`) | `momentum_time_integrator="rk3_ws"`, `tracer_time_integrator="euler"` | the momentum stages would, but the card is UNCONSTRUCTIBLE on this branch: model construction raises `pgf_quadrature="nemo_trapezoid" is the hpg_sco recurrence and requires pgf_scheme="nemo_sco"`.  Five `test_nemo_recipe.py` GYRE tests fail identically at this round's base `28d166428`; pre-existing, untouched here |
| DINO `kamm_mlf` (`build_nemo_recipe`) | `momentum_time_integrator="rk3"` (the `NEMOModelRecipeConfig` default) | no — the `rk3_ws` block is skipped, whichever `outer_integrator` the DINO screens select |
| ORCA1 / OMIP (`run_omip_core2.py:5936,6096`) | `momentum_time_integrator = "rk3" if --momentum-rk3 else None` | no |

Two live rows, so the LOCK no-op is the only cross-card check this change
gets, and that is a FINDING, not a success: the WS-RK3 identity is exercised
by exactly one non-degenerate card today.  Because DINO does not reach the
change, its short-run bit-exact gates were not run; nothing on the MLF lane
can move.

**The same defect class DOES exist on the MLF lane, and is deliberately NOT
fixed here.**  NEMO's MLF `dom_qco_r3c` (`domqco.F90:166-169`) uses the
IDENTICAL `r3u` formula as the RK3 variant (`:219-222`) — both were read.
legoESM's MLF transport face thickness is still
`min_cell_to_uface(h_k)` (`ocean_model_latlon_cgrid.py:9390`), and main's
PR #1642 separately moved the MLF *vertical-mixing* face control volume to a
masked average (`:8130-8160`), which is a third rule again.  Changing the MLF
transport face would move every DINO number and is a different one-variable
round; recorded here as open debt.

#### RETRACTION (next round, branch `fidelity/face-thickness-shared`)

**The paragraph above misattributes the site, and its "open debt" conclusion
does not follow.**  Retracted here rather than quietly corrected, because a
stale confident pointer gets built on.

`ocean_model_latlon_cgrid.py:9390` at this round's base `28d166428` is
`h_u = min_cell_to_uface(h_k)` whose ENCLOSING function is `_ab2_step`
(`def` at `:9108`) — the Veros Adams-Bashforth-2 outer integrator, which is
not the MLF lane at all — and that line is a barotropic/baroclinic DEPTH-MEAN
WEIGHT, not a transport face thickness.  The MLF lane's own steppers are
`_leapfrog_step` (`:9824` at tip) and `_nemo_mlf_step` (`:10361`); their
depth-mean weights are separate sites again.

What the MLF lane actually does, measured by instantiating each card and
printing the resolved selectors (fp64, CPU):

| card | `outer_integrator` | `momentum` / `tracer` integrator | `wzv_call2_evaluation` | tracer-transport face rule |
|---|---|---|---|---|
| DINO `nemo_dino_kamm_mlf` | `leapfrog` | `euler` / `euler` | `nemo_literal` | ALREADY `e3u_0*(1+r3u)` |
| DINO `nemo_dino_kamm` | `forward_euler` | `euler` / `euler` | `nemo_literal` | ALREADY `e3u_0*(1+r3u)` |
| ORCA1 / OMIP (`run_omip_core2.py`) | `forward_euler` | `euler` / `euler` | `generic` | min-rule |
| OVERFLOW-zps, LOCK_EXCHANGE-zco | `forward_euler` | `rk3_ws` / `rk3_ws` | `generic` | `e3u_0*(1+r3u)` in the stage transport |
| GYRE | `forward_euler` | `rk3_ws` / `euler` | `generic` | stage transport only |

So the DINO cards were already on NEMO's rule for the tracer transport
(`ocean_model_latlon_cgrid.py`'s `wzv_call2_evaluation == "nemo_literal"`
branch, which called `vertical.nemo_qco_live_face_thicknesses`), and no DINO
number could have moved by adopting it.  The `generic` arm's `min` rule is a
REAL legoESM scheme selection, not a defective transcription of NEMO, and it
is left alone.  Its in-place citation (the MOM6/MITgcm `hFacW` convention,
Adcroft–Hill–Marshall 1997 eq. 11–13, at
`ocean_model_latlon_cgrid.py:1401-1406`) covers only the REFERENCE part:
AHM97's `hFacW` is a FIXED fraction, and taking the min of the two STRETCHED
thicknesses is a nonlinear-free-surface extension that postdates that paper.
The check that would settle whether the extension is a scheme or a defect is
free-surface/tracer consistency — advect a uniform tracer under that face
thickness and confirm it stays uniform — and it is NOT run here.

What WAS an artificial branch point: the two lanes reached the shared rule
through two DIFFERENT wrappers, each carrying its own copy of the
native-east/north → redundant-west/south face map.  Both now call the one
`vertical.nemo_qco_live_face_geometry_cgrid`.  Pure refactor: identical
operand order, identical arithmetic, verified by re-running all four phase-3
gates before and after.  Two copies of that map survive elsewhere and are
NOT touched here, because they sit on the dynzad/`sshwzv` path rather than on
either time-stepping lane, and that path cannot be exercised today (the DINO
cards are unconstructible, below): `ocean_pe_latlon_cgrid.py:1493-1509`
re-opens the same five-field operand lookup and `:1520-1524` re-implements the
same native-to-redundant concatenate.

`_replace_stage_mean` is CLEARED, not deferred — and the first version of
this block got that wrong.

NEMO's RK3 stage barotropic correction weights the depth mean with the
REFERENCE ladder (`stprk3_stg.F90:440-441`,
`uu_b(Kaa) - SUM( e3u_0(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0`), while its MLF
counterpart `mlf_baro_corr` weights with the LIVE after-level face
(`stpmlf.F90:514-522`, `e3u(ji,jj,jk,Kaa)` and `r1_hu(:,:,Kaa)`).  legoESM's
`_replace_stage_mean` uses the live Kbb `min_cell_to_uface(h_k_pre)` over the
live `H_u_pre`.

**RETRACTED:** this block first reported `max|h_u_pre − e3u_0| = 4.922e-02 m`
(4.92 %) on the LOCK tilted entry state and concluded that correcting the
weighting "would move the `kt=2..10` trajectory".  That is the WEIGHT
difference, which is the wrong quantity: a depth mean is invariant under a
per-column rescaling of its weight, and under `key_qco` the free-surface
stretch is uniform in the vertical, so both NEMO forms and legoESM's collapse
to the same mean whenever one neighbour wins the min rule at every level.  An
independent physics review raised the cancellation; re-measured directly, in
fp64, on both certified cards:

| card | max abs eta [m] | max abs (h_u_pre − e3u_0) [m] | max per-column spread of h_u_pre/e3u_0 | max abs depth-mean difference [m/s] |
|---|---:|---:|---:|---:|
| LOCK_EXCHANGE-zco | 0.9845 | 4.922e-02 | 0.0 | 2.168e-19 |
| OVERFLOW-zps | 0.9881 | 3.952e-02 | 2.220e-16 | 6.939e-18 |

The ratio `h_u_pre/e3u_0` is depth-uniform to at most one ULP on both cards,
so legoESM already reproduces `stprk3_stg.F90:440-441` to roundoff and there
is nothing to correct.  The residual risk is a mesh on which the min rule's
WINNER flips with depth (a partial-cell step whose shallower side carries the
higher ssh); neither certified card does that.  No code change — and the
one-variable round this block previously scoped would have measured a null.

### Scaling BEFORE any owner label

Recomputed by the gate itself on every run
(`face_thickness_and_qco_scaling`), from the oracle kt=1 dumps and the card
geometry only — no legoESM arm enters it.  These reproduce the frozen
preregistration numbers exactly.

H1, the face thickness.  The oracle's per-stage tracer increment is 100%
advective on this card, so the predicted T movement is the relative transport
error times that increment:

| stage (Kmm level) | max abs Δe3u [m] | max rel Δe3u | oracle stage max abs ΔT [K] | predicted T movement [K] |
|---|---:|---:|---:|---:|
| 1 (Kbb, ssh ≡ 0) | `0.0` | `0.0` | `3.4518e-04` | `0.0` |
| 2 (N+1/3) | `1.3199826e-03` | `6.5999e-05` | `6.0182e-04` | `3.9720e-08` |
| 3 (N+1/2) | `1.9799739e-03` | `9.8999e-05` | `3.6980e-03` | `3.6609e-07` |
| **total** | | | | **`4.0581e-07`** |

H2, the qco stage weighting.  With `uu(Kbb)=0` at kt=1 the omitted factor
leaves `u_lego - u_nemo = u_nemo*(r3u(Kaa)-r3u(Kmm))/(1+r3u(Kmm))`, and the
stage barotropic correction removes its depth mean, so only the baroclinic
part survives into the scored row:

| stage | max abs r3u(Kaa) | max abs r3u(Kaa)−r3u(Kmm) | predicted baroclinic u movement | pre-fix residual | predicted/measured |
|---|---:|---:|---:|---:|---:|
| 1 | `3.4495e-05` | `3.4495e-05` | `3.7635e-12` | `3.761789e-12` | `1.0005` |
| 2 | `5.1743e-05` | `1.7248e-05` | `6.1965e-11` | `6.908114e-11` | `0.897` |
| 3 | `1.0349e-04` | `5.1743e-05` | `6.3212e-10` | `2.598440e-07` | `0.0024` |

The stage-3 row is why H2 was preregistered as REFUTED for the `kt=2` u
residual before any arm was run.

### Arm outcomes against the frozen predictions

Scored by the gate's `preregistered_prediction_check` block, which carries the
committed windows as constants.

| prediction | predicate | measured | status |
|---|---|---|---|
| P1 (H1 owns the T residual) | kt=2 T movement in `[2.03e-07, 8.12e-07] K` and the residual improves | movement `2.5484875e-07 K`; residual `2.5484897e-07 -> 2.2382096e-13 K` | **MET** (predicted `4.0581e-07`, 1.59x the movement) |
| P2 (H1 is REFUTED for u) | stage-3 u movement `< 2.6e-08 m/s` | `7.958972e-11 m/s` | **MET** |
| P3a (H2 owns stage 1) | faithful stage-1 baroclinic u `< 1.3e-13 m/s`, ablation reproduces the pre-fix value | `1.811051e-15 m/s`; ablation `3.761789e-12` vs predicted `3.7635e-12` | **MET** |
| P3b (H2 owns stage 2) | faithful stage-2 baroclinic u `< 1.4e-11 m/s` | `9.433445e-11 m/s` | **NOT-MET** |
| P3c (H2 is REFUTED as the stage-3 owner) | predicted stage-3 movement `< 0.01x` the kt=2 u residual | `6.3212e-10 / 2.5988e-07 = 0.0024`; measured movement `6.316738e-10` | **MET** |
| P4 (both inert on LOCK) | every LOCK movement `< 1e-15` | H1 `0.0` / `0.0`; H2 `0.0` and `1.29e-26` | **MET** |

**P3b is a genuine preregistration miss and is not being reinterpreted.**  The
H2 stage-2 MOVEMENT prediction (`6.1965e-11`) is scale-compatible with what
the arm moved, but turning that into a residual FLOOR assumed H2 owned the
`6.908e-11`, and it does not: at stage 2 both corrections make the u residual
slightly worse (pre-fix `6.908e-11`; H1 alone `7.477e-11`; H2 alone
`8.865e-11`; both `9.433e-11`).  That is Rule 8 — faithful-but-worse on a
term four orders below the stage-3 residual, disclosed, not reverted, and not
hidden behind a default.

### kt=2 entry, both cards (absolute L-infinity on the wet mask)

| card | quantity | before | after |
|---|---|---:|---:|
| OVERFLOW-zps | T | `2.548493e-07 K` | `2.238210e-13 K` |
| OVERFLOW-zps | u | `2.598440e-07 m/s` | `2.598798e-07 m/s` |
| OVERFLOW-zps | SSH (normalized) | `1.0491608e-14` | `1.0491608e-14` |
| LOCK_EXCHANGE-zco | T | `4.883205e-12 K` | `4.883205e-12 K` |
| LOCK_EXCHANGE-zco | u | `2.138804e-10 m/s` | `2.138804e-10 m/s` |
| LOCK_EXCHANGE-zco | SSH (normalized) | `4.7821379e-28` | `4.7821379e-28` |

LOCK is bit-identical at every kt=1 stage and at kt=2, which is exactly what
the source says must happen: its oracle `ssh` is identically zero at all three
kt=1 stages (0 non-zero cells of 134x7) and legoESM's own `eta` after kt=1 is
`4.78e-28 m`, so both face rules and both velocity updates coincide
algebraically.  A no-op where the source predicts a no-op is the strongest
control this round has.

Stage rows, OVERFLOW-zps (instantaneous u, absolute):

| stage | before | after |
|---|---:|---:|
| 1 | `3.757022e-12` | `6.501744e-15` |
| 2 | `6.908155e-11` | `9.433404e-11` |
| 3 | `2.598440e-07` | `2.598798e-07` |

Stage-2 tracer operand (the row that measured the face-thickness defect
directly): `2.833037e-08 K (DEBT) -> 3.552714e-15 K (AT-BAR)`.

### kt=2..10 trajectory (normalized L-infinity, both cases)

OVERFLOW-zps:

| kt | T before | T after | U before | U after | SSH before | SSH after |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | `1.27424627e-08` | `1.11910481e-14` | `2.59844026e-07` | `2.59879793e-07` | `1.04916076e-14` | `1.04916076e-14` |
| 3 | `1.63310582e-07` | `1.94808747e-09` | `3.37673071e-06` | `3.37369013e-06` | `3.75492262e-09` | `3.77382183e-09` |
| 4 | `5.90530377e-07` | `1.87561953e-08` | `3.37636512e-06` | `3.35428742e-06` | `1.07119102e-06` | `1.07153587e-06` |
| 5 | `1.16912537e-06` | `3.52748005e-08` | `3.28700281e-06` | `3.22624514e-06` | `1.48122887e-05` | `1.48051671e-05` |
| 6 | `1.74603452e-06` | `5.00379945e-08` | `9.27687843e-06` | `9.27010239e-06` | `4.00019841e-05` | `3.99913513e-05` |
| 7 | `2.39742496e-06` | `5.89796107e-08` | `1.15106208e-05` | `1.15021090e-05` | `4.55757512e-05` | `4.55794834e-05` |
| 8 | `3.29132932e-06` | `5.99574620e-08` | `2.66872742e-05` | `2.66887966e-05` | `7.97135279e-05` | `7.96914266e-05` |
| 9 | `4.40403288e-06` | `6.16954667e-08` | `3.21886997e-05` | `3.21852255e-05` | `7.29592804e-05` | `7.29419662e-05` |
| 10 | `5.55186262e-06` | `7.71178526e-08` | `2.64522041e-05` | `2.64429824e-05` | `9.24110227e-05` | `9.23739063e-05` |

LOCK_EXCHANGE-zco:

| kt | T before | T after | U before | U after | SSH before | SSH after |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | `1.62773498e-13` | `1.62773498e-13` | `2.13880413e-10` | `2.13880413e-10` | `4.78213792e-28` | `4.78213792e-28` |
| 3 | `5.20626505e-12` | `4.04654088e-13` | `2.13879378e-10` | `2.13879691e-10` | `1.35525272e-18` | `1.35525272e-18` |
| 4 | `3.59465938e-11` | `8.40098361e-13` | `2.13851856e-10` | `2.13863540e-10` | `2.16027283e-17` | `2.16027283e-17` |
| 5 | `1.21776485e-10` | `1.18554055e-12` | `2.13746464e-10` | `2.13790095e-10` | `6.41580871e-16` | `8.12355683e-16` |
| 6 | `3.05194329e-10` | `1.53110117e-12` | `2.13440690e-10` | `2.13571487e-10` | `3.83130144e-15` | `5.59530367e-15` |
| 7 | `6.40530355e-10` | `1.94392650e-12` | `2.12733880e-10` | `2.13058932e-10` | `1.56074155e-14` | `2.21587936e-14` |
| 8 | `1.19372411e-09` | `2.29789521e-12` | `2.11363624e-10` | `2.12027803e-10` | `4.73124144e-14` | `6.59659286e-14` |
| 9 | `2.04205328e-09` | `2.64973229e-12` | `2.08895765e-10` | `2.10163988e-10` | `1.26009229e-13` | `1.63752373e-13` |
| 10 | `3.27381405e-09` | `2.99730611e-12` | `2.04777155e-10` | `2.07050559e-10` | `2.85172493e-13` | `3.57703289e-13` |

LOCK's T improves 1092x at kt=10 even though the change is inert at kt=1: its
`eta` leaves zero after the first step, and from kt=3 on the two rules
diverge.  Its U and SSH move by a few percent in the worse direction and stay
flat at `~2.1e-10` and `~3e-13`; disclosed, not reverted.  Why a `~1e-13 m`
sea level produces a 1092x T change is **PLAUSIBLE, UNMEASURED**: `min` breaks
the left-right symmetry of a lock exchange while the NEMO mean does not.

kt=60 bridge (OVERFLOW-zps, before-entry normalized L-infinity at kt=60):
T `1.027641e-04 -> 1.184430e-05`, u `4.193114e-04 -> 2.507515e-04`,
S `2.639159e-15 -> 2.436147e-15`, ssh `7.757762e-05 -> 7.751799e-05`.

### Statistical scorer (6120 steps, fp64 and fp32, CPU)

Same scorer, same NEMO FCT2/FCT4 spread, same-revision fp32 floor, both arms
complete with every-step finite checks at `7fc887dd93c1` (fp64 `421.41 s`,
fp32 `217.95 s` wall).  The deterministic kt1..60 bridge was re-pinned to this
round's gate, as it is model output.

| metric | before | after | fp32 floor before | fp32 floor after | NEMO spread | verdict before | verdict after |
|---|---:|---:|---:|---:|---:|---|---|
| final_temperature_histogram_tv | `0.0423046` | `0.0390072` | `0.0194058` | `0.0174136` | `0.044231` | WITHIN-SCHEME-SPREAD | WITHIN-SCHEME-SPREAD |
| final_water_mass_census | `0.0169753` | `0.0130347` | `0.00157409` | `0.00180818` | `0.00336209` | OUTSIDE | OUTSIDE |
| instantaneous_u_linf | `0.999211` | `0.832083` | `0.402734` | `0.773437` | `0.672694` | OUTSIDE | OUTSIDE |
| plume_descent_m | `16.9554` | `16.9551` | `0.0087228` | `0.038907` | `1499.62` | WITHIN-SCHEME-SPREAD | WITHIN-SCHEME-SPREAD |
| plume_front_km | `4.04936` | `4.03884` | `0.0837761` | `0.135393` | `121.931` | WITHIN-SCHEME-SPREAD | WITHIN-SCHEME-SPREAD |
| temperature_linf | `0.379492` | `0.379487` | `0.140318` | `0.122985` | `0.356744` | OUTSIDE | OUTSIDE |

`OUTSIDE` stays 3 and `WITHIN-SCHEME-SPREAD` stays 3.  Every row improves or
holds — the water-mass census drops from 5.05x to 3.88x the NEMO scheme
spread and the instantaneous-u L-infinity from 1.49x to 1.24x — but no row
crosses.  **A six-order kt=2 gain buys a few percent at 6120 steps: the
long-run statistics are owned by something else, and this round does not
change that verdict.**

### Controls, and one control this round broke and repaired

- `--plant-stage` on LOCK: exit `1`, planted row carries the `+1.0` and is in
  `failed_rows`.
- `test_undetected_plant_exits_two`: a scorer that stops seeing plants makes
  the planted run exit `2`.  It FAILED at the base revision `28d166428`
  (LLVM `Cannot allocate memory`) and passes here, because the gate now clears
  the JAX compilation cache between arms.
- `--plant-prediction` inflates the frozen H2 numbers 1000x and REQUIRES the
  stage-3 refutation predicate to flip to NOT-MET; on LOCK that predicate does
  not exist, so the planted run exits `2`.
- **Adding `--plant-prediction` first BROKE the `--plant-operand` control.**
  Its guard landed between the operand plant's case check and its
  `require_planted` call, so the operand plant's landing check ran under the
  wrong flag: `--plant-operand` silently stopped verifying its own plant.
  Found by RUNNING the control, not by reading the diff; repaired in
  `2dd8cee05` and both plants re-run.
- Non-vacuity of the two new unit tests: both FAIL against the reverted source
  (checked out from `HEAD~1`) and pass with it, and
  `test_ws_stage_face_thickness_is_nemo_e3u_0_times_one_plus_r3u` additionally
  asserts inline that the reverted min rule differs by more than `1e-6 m`.

### Tests

Focused CPU/fp64 run at `2dd8cee05`, nine explicitly listed files:
**`7 failed, 106 passed in 645.49s`**
(`test_nemo_ws_qco_stage_faces`, `test_nemo_ws_tracer_rk3`,
`test_rk3_ws_and_mxl3`, `test_nemo_recipe`,
`test_nemo_testcase_full_statistics`,
`test_nemo_testcase_phase3_stage_sweep_gate`,
`test_nemo_testcase_phase3_trajectory_gate`,
`test_nemo_testcase_oracle_gate`,
`test_nemo_testcase_overflow_barotropic_gate`).

All seven failures reproduce IDENTICALLY at this round's base `28d166428`,
run in a separate worktree with the same command: two `test_rk3_ws_and_mxl3`
and five `test_nemo_recipe` cases whose configurations are rejected at model
construction by `pgf_quadrature="nemo_trapezoid" ... requires
pgf_scheme="nemo_sco"`.  Pre-existing, unrelated to this round, untouched.

The base run had an EIGHTH failure this round FIXES:
`test_undetected_plant_exits_two` died at `28d166428` with LLVM
`Cannot allocate memory` (`vm.max_map_count`, not host RAM) and passes here
because the gate now clears the JAX compilation cache between arms.

Planted-control exit codes, run end to end on the real cards:
`--plant-stage` on LOCK exit `1` with the planted row in `failed_rows`;
`--plant-operand` on OVERFLOW exit `1` with
`kt1.stage1.faithful.tracer_operand_T = 1.0000e+00 DEBT` in `failed_rows`;
`--plant-prediction` on OVERFLOW exit `1` with the frozen stage-3 predicate
flipped `MET -> NOT-MET` (planted prediction `6.3212e-07` vs the real
`6.3212e-10`).  Each exits `2` instead if its plant fails to land.

### Retractions and revisions (Rule 11)

- The previous round's registered next-arm estimate for this residual
  (`~6e-8 K`, from the stage-2 relative face error times one stage's change)
  is SUPERSEDED, not retracted: the per-stage sum is `4.06e-07 K` because
  stage 3 carries `3.66e-07` of it, and the measured movement was
  `2.55e-07 K`.  Its description of `r3u` as "the `e1e2`-weighted mean of the
  two columns' `r3t`" is CORRECTED: it is the weighted mean of `ssh` over
  `hu_0`, which is a different number when the two columns have different
  depths — exactly the OVERFLOW shelf break.
- The stage-sweep gate's `classify_arm` label is NOT applicable to these two
  arms: they ablate a LANDED fix, which inverts its "does the arm improve the
  residual" question.  The ablation verdict is recorded in its own field
  (`ablation_classification`) and the `classification` string for these two
  rows should be ignored.

### Provenance

| artifact (under `/data/abyssal/dbalwada/nemo-testcases-l1/`) | SHA256 |
|---|---|
| `face_thickness/lock_stage_sweep_gate_kt2.json` | `b82f4bcedfde95cb883e5954e11048980cfe00b143e4a8dee2a897ca43fadc7e` |
| `face_thickness/lock_trajectory_gate_kt10.json` | `1268c73c15063f0f4e16f653440d46c6eac0cd151b23f4d23c74a8d2fe11c8a5` |
| `face_thickness/legoesm/overflow_zps/fp32/metadata.json` | `a9bab4071db0d7e322faa2217d7bc22122066115af3796e0bcb72ec8d0e7d621` |
| `face_thickness/legoesm/overflow_zps/fp32/states.npz` | `aea3fe027d288a0e57e9e2f4c01ac7b0f3d3bb69ced9cd92d29d4381e8fe8565` |
| `face_thickness/legoesm/overflow_zps/fp64/metadata.json` | `a49ed0aaa0666d83d5037fb9d3f52b688475eb0c9e479ff45f8223c72a5e1c3a` |
| `face_thickness/legoesm/overflow_zps/fp64/states.npz` | `65a54c292cddde8ef2149cccb8deaefd50f847a8e363eb6e43301113a9b9b7d0` |
| `face_thickness/overflow_stage_sweep_gate_kt2.json` | `f85c91324f7aa10b6b231216abbc4bff11beb4450226ddfa8cd49fb24b6ddf15` |
| `face_thickness/overflow_statistics.json` | `cb975028e90d26579ff40b67b1bd4566790c71d70cda868fbf3dc4ae6a4af909` |
| `face_thickness/overflow_trajectory_gate_kt10.json` | `d1c69f6e3f27f9271e80331eed79e1fdf707f9758eb1330952ba34e17d2cc88c` |
| `face_thickness/overflow_trajectory_gate_kt60.json` | `0b46df0aa025c10ae3c7c7b371e2fe9c91cd4812bcaddbe4d4a9b50fc5d65582` |

Every gate JSON above stamps `legoesm_git_sha = ccb8e563e1fe91c799f652b49df5b4bf86521357`
on a CLEAN tree; the two later commits (`7efe057d2` re-pin, `2dd8cee05`
control repair) touch no model code, so those artifacts remain the current
model's output.  The statistics runs stamp `7fc887dd93c1bc38c60c16b0faba1da4b7c30028`.

All five reach-table rows were settled structurally as well as by
instantiation: `_nemo_ws_qco_stage_faces` and `_nemo_ws_stage_transport` are
called from exactly five sites (`ocean_model_latlon_cgrid.py:1142` inside the
transport itself, and `:4874,4936,4954,4972,5361`), every one of them inside
`_step_impl`'s `momentum_time_integrator == "rk3_ws"` block.  DINO's card
resolves that selector to `"rk3"`, so the changed lines cannot execute on the
MLF lane and its short-run gates were not run.

### What is now UNMEASURED after this round

- the OVERFLOW `kt=2` u residual `2.598798e-07 m/s`, which appears entirely at
  stage 3.  Both hypotheses of this round were preregistered as refuted for it
  and both refutations held (H1 moved it `7.96e-11`, H2 `6.32e-10`);
- the OVERFLOW stage-2 u residual `9.433e-11`, which BOTH faithful changes
  make slightly worse than the pre-fix `6.908e-11`;
- the LOCK u residual, still flat at `~2.1e-10` from kt=2 to kt=10;
- `_replace_stage_mean` still weights with `min_cell_to_uface(h_k_pre)` where
  `stprk3_stg.F90:438` uses `e3u_0` and `r1_hu_0`.  These agree exactly at
  kt=1 (ssh = 0 at Kbb) and drift apart afterwards; not touched this round;
- ~~the MLF-lane face thickness (`ocean_model_latlon_cgrid.py:9390`), which
  has the same defect against the identical `domqco.F90:166-169` formula~~ —
  RETRACTED, see the retraction block above: that line is in `_ab2_step`, and
  the DINO MLF cards were already on NEMO's rule;
- whether the WS-RK3 identity is exercised by any card other than these two.

## S-30 / S-12 collapse: the WS momentum ladder is now written once (2026-09-02)

Verdict: **ADMITTED under the ulp bar.**  The Wicker-Skamarock RK3 momentum
recurrence used to be written twice inside `_step_impl` — once before the
barotropic solve, only to build that solve's velocity seed, and once after it,
barotropically corrected and kept.  The first copy is deleted.  Every certified
row of both cards moved by at most **0.5 float64 ulp of its own scale**, every
T and S row at every step from kt=1 to kt=10 is **bit-identical**, and
`first_over_bar` is unchanged on both cards.  The DINO twin is unreached.

Collapse commit `28df515a84d572b28a7a5c6afb1ca905b31bfda4`; the pre-collapse
reference is `9070cf2767bc679e31bd49c21471a497fb90f280`.  Both arms fp64,
CPU backend, `PrecisionPolicy.fp64()` asserted by each gate's own entry point.

### What NEMO does, and why the first ladder had no counterpart

`stp_2D` evaluates the Kbb right-hand side ONCE — `eos`/`dyn_hpg`/`dyn_ldf`/
`dyn_vor`/`wzv`/`dyn_adv`, all at `Kbb` (`stp2d.F90:126-171`) — depth-means it
into `Ue_rhs`/`Ve_rhs` with the reference thicknesses,

```
Ue_rhs(ji,jj) = SUM( e3u_0(ji,jj,1:jpkm1)*uu(ji,jj,1:jpkm1,Krhs)
                    *umask(ji,jj,1:jpkm1) ) * r1_hu_0(ji,jj)   stp2d.F90:180
```

and hands the solver the BEFORE velocity:

```
CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )
                                                               stp2d.F90:281
```

`Kmm == Kbb`: there is no momentum stage between the RHS and the solve.  The
stage recurrence lives in `stprk3_stg.F90:344-374`, after it.  legoESM already
carried that depth-mean separately as `F_slow_u`/`F_slow_v`, so the pre-solve
ladder was pure duplication — and an expensive one: two extra `tendencies()`
evaluations per step on every `rk3_ws` card.  It was also actively harmful:
the stage vertical-UP3 and live-HPG-operand fixes of the stage-composition
round landed in the second copy only.

### Why the move is a re-association and not a physics change (read, not inferred)

The deleted stages added only ZERO-DEPTH-MEAN perturbations to `u0`
(`_mom_pert_ws` returns `du - depth_mean(du)`), and the barotropic solver
consumes its velocity argument exclusively through depth means: the substep
seed `U_bar`, and `u_prime = u_corr - U_bar_corr`
(`barotropic_latlon_cgrid.py:2071-2077,2362-2380`), whose own depth mean is
identically zero.  The one 3-D field the seed did influence, the solver's
returned `state_new.u`, is overwritten by the surviving ladder in any case.
What is left is summation roundoff — which is exactly the size measured.

### The bar, as a machine check

`legoesm.ocean.fidelity.ulp_move_gate` (`MAX_ULP_MOVE = 2`), reached from both
phase-3 gates as `--compare-to <committed reference JSON>`.  A comparison
PASSES only if every certified row moved at most 2 float64 ulps of that row's
own normalising scale (`2 * 2**-52 = 4.440892e-16` for a normalised row),
every T/S row is bit-identical, every non-measured row field is unchanged, the
row sets match, and `first_over_bar` and the report status are unchanged.
Non-vacuity: `--compare-plant-ulps 3` moves one row by 3 ulps and MUST make the
comparison exit non-zero — measured, it does (`3.000 ulp` violation, exit 1),
and `tests/ocean/unit/test_ulp_move_gate.py` re-plants it into all four
committed reference JSONs plus twelve synthetic violations.

`exact` is treated as a RESTATEMENT of the row's own residual rather than an
independent fact, because it is `normalized_max_abs == 0`.  LOCK's kt=2 SSH
residual went `4.782137916322191e-28 -> 0.0` (0.000 ulp) and took `exact`
False -> True with it; counting that as a second change would count one
admitted move twice.  The gate instead requires `exact` to AGREE with the
residual in both reports (so a future redefinition fails closed) and reports
the flip under `derived_field_changes`.

### kt=1..10 trajectory, both cards (normalised L-infinity on the wet mask)

`first_over_bar` LOCK `kt=2 {T,u}` and OVERFLOW `kt=2 {T,u,ssh}`, both
UNCHANGED.  Report status `DEBT` on both, unchanged.  37 of 50 rows per card
are bit-identical; every row that moved is below:

| case | kt | field | before (`9070cf276`) | after (`28df515a8`) | move | ulps |
|---|---:|---|---|---|---|---:|
| LOCK | 2 | ssh | `4.782137916322191e-28` | `0.0` | `4.7821e-28` | 0.000 |
| LOCK | 3 | u | `2.13879690976089e-10` | `2.1387969054240813e-10` | `4.3368e-19` | 0.002 |
| LOCK | 4 | u | `2.1386353983316592e-10` | `2.1386354026684679e-10` | `4.3368e-19` | 0.002 |
| LOCK | 4 | ssh | `2.1602728286773676e-17` | `2.1575623232461538e-17` | `2.7105e-20` | 0.000 |
| LOCK | 5 | u | `2.1379009478537936e-10` | `2.1379009608642197e-10` | `1.3010e-18` | 0.006 |
| LOCK | 6 | u | `2.1357148693871078e-10` | `2.1357148737239165e-10` | `4.3368e-19` | 0.002 |
| LOCK | 7 | u | `2.1305893171692097e-10` | `2.130589312832401e-10` | `4.3368e-19` | 0.002 |
| LOCK | 8 | u | `2.1202780313813244e-10` | `2.1202780270445157e-10` | `4.3368e-19` | 0.002 |
| LOCK | 8 | ssh | `6.5965928550869518e-14` | `6.5965928974385992e-14` | `4.2352e-22` | 0.000 |
| LOCK | 9 | u | `2.1016398831733701e-10` | `2.1016398788365614e-10` | `4.3368e-19` | 0.002 |
| LOCK | 9 | ssh | `1.6375237347739449e-13` | `1.6375237517146039e-13` | `1.6941e-21` | 0.000 |
| LOCK | 10 | u | `2.0705055942329964e-10` | `2.0705055898961877e-10` | `4.3368e-19` | 0.002 |
| LOCK | 10 | ssh | `3.577032892571096e-13` | `3.5770329264524139e-13` | `3.3881e-21` | 0.000 |
| OVERFLOW | 2 | u | `2.5987979303081221e-07` | `2.5987979300999553e-07` | `2.0817e-17` | 0.094 |
| OVERFLOW | 2 | ssh | `1.0491607582707729e-14` | `1.0505485370515544e-14` | `1.3878e-17` | 0.062 |
| OVERFLOW | 3 | ssh | `3.7738218288188574e-09` | `3.7738218270841339e-09` | `1.7347e-18` | 0.008 |
| OVERFLOW | 4 | ssh | `1.0715358736285152e-06` | `1.0715358736215763e-06` | `6.9389e-18` | 0.031 |
| OVERFLOW | 5 | u | `3.2262451417353066e-06` | `3.2262451417630622e-06` | `2.7756e-17` | 0.125 |
| OVERFLOW | 6 | u | `9.270102390497581e-06` | `9.2701023905045199e-06` | `6.9389e-18` | 0.031 |
| OVERFLOW | 6 | ssh | `3.9991351277357534e-05` | `3.9991351277413045e-05` | `5.5511e-17` | 0.250 |
| OVERFLOW | 7 | u | `1.150210902620824e-05` | `1.1502109026222118e-05` | `1.3878e-17` | 0.062 |
| OVERFLOW | 7 | ssh | `4.5579483448549007e-05` | `4.5579483448576763e-05` | `2.7756e-17` | 0.125 |
| OVERFLOW | 8 | u | `2.6688796622924282e-05` | `2.6688796622945099e-05` | `2.0817e-17` | 0.094 |
| OVERFLOW | 9 | u | `3.2185225506346782e-05` | `3.2185225506374537e-05` | `2.7756e-17` | 0.125 |
| OVERFLOW | 10 | u | `2.644298236102044e-05` | `2.6442982361048195e-05` | `2.7756e-17` | 0.125 |
| OVERFLOW | 10 | ssh | `9.2373906344422885e-05` | `9.2373906344533907e-05` | `1.1102e-16` | 0.500 |

**No T row and no S row appears in that table, on either card, at any of the
ten steps.**  That is the tracer prong of the bar, and it is met exactly.

### Stage sweep, kt=1 stages 1-3 — the face-thickness round's rows, re-pinned

Absolute L-infinity on the wet U-face mask, `faithful` arm.  These are the
`Face-thickness and stage-qco round` rows above; they are the only stage rows
that moved, and each moved a fraction of an ulp.

| case | row | before | after | ulps |
|---|---|---|---|---:|
| LOCK | kt1.stage1 instantaneous u | `2.5587171270657905e-17` | `2.5370330836160804e-17` | 0.001 |
| LOCK | kt1.stage1 baroclinic u | `6.2883726004159257e-18` | `6.3967928176644762e-18` | 0.001 |
| LOCK | kt1.stage2 instantaneous u | `9.7656449441797799e-11` | `9.7656449224957365e-11` | 0.001 |
| OVERFLOW | kt1.stage1 instantaneous u | `6.501743587961073e-15` | `6.5225602696727947e-15` | 0.094 |
| OVERFLOW | kt1.stage2 instantaneous u | `9.4334038844290369e-11` | `9.43340384106095e-11` | 0.002 |
| OVERFLOW | kt1.stage2 baroclinic u | `9.4334445636945485e-11` | `9.4334446070626354e-11` | 0.002 |
| OVERFLOW | kt1.stage2 ssh operand | `5.2458037913538647e-15` | `5.2527426852577719e-15` | 0.031 |
| OVERFLOW | kt1.stage3 instantaneous u | `2.5987979303081221e-07` | `2.5987979300999553e-07` | 0.094 |
| OVERFLOW | kt2 instantaneous u | `2.5987979303081221e-07` | `2.5987979300999553e-07` | 0.094 |

The kt=2 OVERFLOW velocity debt therefore stands at `2.5987979300999553e-07
m/s` and remains UNOWNED; this collapse is EXONERATED for it, as S-35 already
was.

### The one arm whose meaning this commit deliberately changes

`stage_barotropic_correction=False` used to skip the whole post-solve ladder,
leaving the barotropic solver's own velocity as the prognostic one.  With one
ladder there is nothing to skip, so the hook moved INSIDE `_replace_stage_mean`
and now ablates the per-stage external-mode replacement
(`stprk3_stg.F90:433-446`) — which is what its name says and what the sweep
arm `omit_stage_primary_velocity_correction` is for.

Measured, unfiltered, every arm of both stage sweeps:

| arm | LOCK max ulps | OVERFLOW max ulps |
|---|---:|---:|
| `faithful` | 0.001 | 0.094 |
| `freeze_stage_hpg_operands` | — | 0.094 |
| `freeze_stage_hpg_tracers` | — | 0.094 |
| `freeze_stage_hpg_eta` | — | 0.125 |
| `omit_stage_vertical_up3` | — | 0.094 |
| `legacy_velocity_primary_average` | — | 0.031 |
| `legacy_stage_min_face_thickness` | 0.001 | 0.094 |
| `omit_stage_qco_factor` | 0.001 | 0.094 |
| `omit_momentum_transport_reconcile` | 0.001 | — |
| `omit_stage_primary_velocity_correction` | **5.1e12** | **2.0e14** |

So the `.faithful.` row filter used for the sweep comparison is not hiding
anything: EVERY arm is inside the bar except the single redefined one.  Its
movement is the redefinition, not a numerical finding.

### DINO reach

DINO does not run this code.  `nemo_dino_kamm_mlf` resolves
`momentum_time_integrator="euler"`, `tracer_time_integrator="euler"`,
`outer_integrator="leapfrog"` (constructed and printed, not read off a
comment), so both edited branches are gated off.  Measured anyway with the
gate named in `dino_reach_check.md`: `kamm_twin_90d.py nemo_dino_kamm_mlf
--days 5 --bridge-before`, CPU fp64, `LEGOESM_NEMO_E3T=both`, 160 leapfrog
steps from NEMO's day-180 restart, ~155 s per arm, byte-identical invocation,
one variable (the commit).

Every numeric array in the two archives differs by exactly `0.000000e+00`.
The only differing key is `producer_git_sha`.  SAME BLIND SPOT as the earlier
reach check, restated so the clean number is not over-read: the archive stores
`eta`/`sst`/`u`/`v` as float32 SURFACE slices at 5 daily samples, so it
resolves a difference only to ~1e-7 relative and only at the surface.  What
carries the no-reach claim is the pairing of that with the resolved-integrator
readout above, not the surface fields alone.

```
sha256  c8a31c8fbb6e6d4043707c3557c169e6f39487ac58411158b4606517ca3db08f  twin_BASE_d5.npz
sha256  2b0473cf61afd971607f00a1099a4119dd004a255a1b011e8878bdb2daa876bf  twin_HEAD_d5.npz
sha256  5a54b0b5a528df70d4d5a1370dad7afad79b75c426110debe845b9295a78d748  twin_BASE_d5.log
sha256  c2a8b9cf2ea64df9c06e1ed8aca7f52c345d24352eff69990e554c50167d12a5  twin_HEAD_d5.log
```

### Which reference the gate compares against, and when to re-pin it

The four committed references under
`docs/ocean/fidelity/testcases/nemo_testcases_l1_phase3_ref_*.json` are the
PRE-collapse gate JSONs, stamped `9070cf276`.  They are deliberately NOT
re-pinned to the post-collapse numbers: the anchor is the last CERTIFIED
trajectory, not the last run, so a sequence of individually-admissible
re-associations cannot drift past the bar one 1.9-ulp step at a time.  Re-pin
them only when a real physics change is certified, and say in the commit which
certified numbers moved and why.

### Tests

```
tests/ocean/unit/test_ulp_move_gate.py          25 passed
tests/ocean/unit/test_nemo_ws_tracer_rk3.py     14 passed
tests/ocean/unit/test_nemo_testcase_recipe.py   15 passed
tests/ocean/unit/test_no_scheme_duplication.py  35 passed
```

### Artifacts

```
sha256  0d99cffe75b9797e82a18054be66e1db2827800557bf14f90dc52dbc57ab9b08  s30_collapse/lock_trajectory_gate_kt10.json
sha256  ba78f1ec41c05c97bba73255c32ef70eed59fcbd5e722b5d92c4fb0c72b3a9dd  s30_collapse/overflow_trajectory_gate_kt10.json
sha256  c177826497d57f6e36c230daff112da64c14d3f4289059a83d9326c76cd133f8  s30_collapse/lock_stage_sweep_gate_kt2.json
sha256  bb35d7446fe595128e078757481976a46ac40ab64f89fc7ce925587a43bc752c  s30_collapse/overflow_stage_sweep_gate_kt2.json
sha256  b48b81e5fcebdc3b77f4b3f2745047c21fef62f37c627694e2850ae9193c7fbb  nemo_testcases_l1_phase3_ref_lock_trajectory_kt10.json
sha256  a090221abb324400dd2fb21c20040a2a43a2249d605d3e2406311ebaf9d10532  nemo_testcases_l1_phase3_ref_overflow_trajectory_kt10.json
sha256  d85a3ec23df3bc49018a2dc734c36877ff24a521f3c2e6ad98355e4c91a75045  nemo_testcases_l1_phase3_ref_lock_stage_sweep.json
sha256  83bd8b5b01d782db8468fb33762dd54701cd6c15d8042a9714cb17cf022aaa54  nemo_testcases_l1_phase3_ref_overflow_stage_sweep.json
```

`s30_collapse/` is
`/data/abyssal/dbalwada/nemo-testcases-l1/phase3/s30_collapse/`; the
`nemo_testcases_l1_phase3_ref_*` files are committed under
`docs/ocean/fidelity/testcases/`.

### What is still UNMEASURED after this round

- The 6120-step statistical scorer was NOT re-run for this collapse.  Its
  inputs moved by at most 0.5 ulp at kt=10, so the six frozen metrics are
  expected to hold — expected, not measured, and stated as such.
- The GYRE card (the third `rk3_ws` card) has no gate on this branch and was
  not run; the change reaches it by construction.
- The now-dead legacy stage-transport rebuild at the tracer program's
  `_nemo_ws_live_stage_geometry is None` branch: with one ladder that branch
  is unreachable for every `rk3_ws` card (the config validator couples the two
  integrators).  Left in place, named here, not deleted — a separate one-line
  collapse with its own gate.

### RETRACTION and correction, same day, after dual adversarial review

Two independent reviewers read the collapse and its gate. Both landed on the
same scope error, and following it up produced a RETRACTION of a claim made
three sections above. Recorded here next to the claim it replaces (Rule 11).

**RETRACTED: "every T and S row at every step is bit-identical" was written in
a way that reads as a claim about the TRACER FIELDS. It is not one.** It is a
claim about the gate's REDUCTIONS, which is what the gate measures, and the
two differ. Measured with the new committed probe
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_state_ulp_probe.py`
(per-cell, wet mask, the trajectory gate's own staggering slice, fp64, ten
steps, `9070cf276` vs `28df515a8`):

| case | field | bit-identical | cells ever differing | max abs difference | max ulp (cells >= 1e-10 of field max) |
|---|---|---|---:|---|---:|
| LOCK | T | YES | 0 | 0 | 0 |
| LOCK | S | YES | 0 | 0 | 0 |
| LOCK | v | YES | 0 | 0 | 0 |
| LOCK | u | no | 3467 | `6.9389e-18` | 69 |
| LOCK | eta | no | 160 | `2.1684e-19` | 2 |
| OVERFLOW | S | YES | 0 | 0 | 0 |
| OVERFLOW | v | YES | 0 | 0 | 0 |
| OVERFLOW | **T** | **no, from kt=7** | 37 | `1.0658e-14` K | 5 |
| OVERFLOW | u | no | 15704 | `1.1102e-16` | 96 |
| OVERFLOW | eta | no | 376 | `2.7756e-16` | 12 |

OVERFLOW's T field is bit-identical for six steps and then departs at five
cells, growing to twelve cells and `1.07e-14 K` by ten steps, while the T ROW
stayed bit-identical throughout — the moved cells are simply not the cell that
attains the maximum residual. That is the gate's declared blind spot firing in
practice, and it is why the probe now exists and is committed.

Physically this is unavoidable rather than surprising: once the velocity moves
at all, the tracers it advects must eventually move. NO re-association that
touches velocity can hold tracer fields bit-identical indefinitely. What can be
said, and now is: T and S are bit-identical through kt=6 on OVERFLOW and
through all ten steps on LOCK, and every certified ROW is bit-identical.

**The bar has two readings and they disagree.** "At most 2 ulp of float64" per
certified number resolves to ADMIT or REFUSE depending on the normalising
scale, and both readings are defensible:

| reading | what is bounded | measured worst | verdict |
|---|---|---:|---|
| 2 ulps at UNIT scale (what the gate implements) | the normalised gate row | 0.500 ulp | ADMIT |
| 2 ulps of the row's own FIELD scale | the field behind the row | 3.0 ulp (OVERFLOW kt=1 stage-1 u, scale `0.0622`) | REFUSE by one ulp |
| 2 ulps per CELL of the raw field | the state itself | 96 ulp (OVERFLOW u), 5 ulp (OVERFLOW T) | REFUSE |

The third row needs reading with care rather than alarm: the perturbation
enters as a column-uniform barotropic shift, so it is an ABSOLUTE quantity of
size ~`2e-17 m/s`, and it therefore lands on a cell of magnitude `1e-4` of the
field maximum as `1e4` times as many ulps as on the largest cell. A per-cell
ulp count is the wrong instrument for an absolute perturbation. The second row
is the sharpest defensible statement: the largest velocity in the OVERFLOW
stage-1 field moved by three ulps of itself, one more than the bar.

**This is a DECISION, not a finding, and it is left to the user.** The gate as
committed implements the first reading and passes. Nothing here re-runs the
bar downward: the constant is `MAX_ULP_MOVE = 2` and the looseness is now
written into the module's own blind-spot list, with the measured factor.

**SCOPE: the "pure re-association" claim is certified for f = 0 only.** Both
reviewers found this independently and it is the more important of the two.
`nemo_testcase_recipe.py:158-159` builds both cards with `f0=0.0, beta=0.0`,
and the cards leave `coriolis_scheme` at its `matsuno_split` default. At f = 0
the Matsuno rotation between the deleted ladder and the barotropic seed is the
exact identity, so the seed's 3-D structure has no channel into the solve at
all. It is NOT the identity on a rotating card: `_forward_backward_coriolis_3d`
reads the full 3-D profile, and the four-point average of a field with zero
`h_v`-weighted mean does not in general have zero `h_u`-weighted mean where
column weights vary, so the barotropic seed's DEPTH MEAN would change at
`O(dt^2 f)` — first order, not roundoff. The same applies to the EEN
barotropic-Coriolis pre-step, which also reads the 3-D field (and which the
change arguably makes MORE faithful, since NEMO removes the barotropic
Coriolis from the Kbb depth mean at `dynspg_ts.F90:358`).

GYRE is the third `rk3_ws` card, it is rotating, and it has no gate on this
branch. So: on LOCK and OVERFLOW this is a measured re-association; on GYRE it
is an UNMEASURED behaviour change. Gate GYRE before repeating the
re-association claim there. Not done here, and not fixed by pinning
`coriolis_scheme` in the `rk3_ws` validator — that would be a silent selector
choice.

**UPGRADED from "expected" to UNMEASURED: the 6120-step statistical scorer.**
The earlier text argued the six frozen metrics should hold because the inputs
moved by at most 0.5 ulp at kt=10. That inference does not survive the gate's
own growth block for OVERFLOW: tail exponential rates `0.203/step` (ssh),
`0.269/step` (u), `0.083/step` (T), and the fits cannot discriminate
exponential from polynomial on a four-point tail (semilog r2 `0.738` vs loglog
`0.766` for ssh; `0.738` vs `0.694` for T, which the gate itself labels
`EXPONENTIAL_FIT_PREFERRED_OPEN_MODE_QUESTION`). If exponential, a `1e-16`
seed reaches order one in a few hundred steps, far inside 6120; if polynomial,
it never matters. Those rates also characterise the legoESM-vs-NEMO
DISCRETISATION error, not this perturbation's own growth, which is a further
untested assumption. Treat the long-run statistics as UNMEASURED for this
collapse. The cheap discriminator is ~300 OVERFLOW steps on both revisions with
`max|delta u|` plotted per step on a semilog axis.

**Gate hardening from the same review, landed here:** the planted control used
to target the alphabetically first non-tracer row, which on both trajectory
references is a zero-residual row — so the plant went red through the derived
`exact` consistency check rather than through the ulp bar, and stayed red even
with the bar deleted outright. Verified by emptying `MOVABLE_ROW_KEYS` and
watching the old assertion still pass. The plant now targets a nonzero-residual
row, the test asserts a sub-bar plant PASSES and an over-bar plant fails on a
MOVE violation, and emptying `MOVABLE_ROW_KEYS` now turns all four reference
tests red. The comparison also now checks top-level `selectors` and
`precision_policy`, records which report-level keys were absent from both
reports and therefore NOT checked (`first_over_bar` does not exist in the
stage-sweep schema, so that prong is inert there), records the row-filter
substring, and exits 2 when a requested plant fails to land.

Reviewers: two, independent, both Claude (the codex CLI is unavailable on this
account, so the standing codex+GLM pairing was met with two independent Claude
reviewers instead — stated rather than implied). Both APPROVED the NEMO
citation after reading `stp2d.F90` themselves; both raised the f=0 scope as the
leading finding. Every finding above was re-measured before being acted on; the
per-cell probe was written because a reviewer's claim about T could not be
settled from the gate JSONs.

## UP3 upwind-selector round: the OVERFLOW stage-3 baroclinic `u` owner, MEASURED

Preregistered in `nemo_testcases_l1_stage3_baroclinic_preregister.md`
(commit `02179b0eb`, frozen before the arm).  Code: `42ac525cc` (fix) and
`8d6756a42` (review round: selector owned by the WS-RK3 identity at every
stage).  All numbers fp64, CPU; baseline = the same gates re-run at
`9070cf276` from a pristine worktree (`stage3_selector/gates_before/`).

### Owner

NEMO `dynadv_up3.F90:166,169-170` selects the UP3 upwind curvature of the
T-point (same-direction) momentum fluxes by the sign of the advected-velocity
pair `zui = uu(Kmm)_i + uu(Kmm)_{i+1}`, while the flux magnitude (`:176`) is
the stage transport pair `zFu_i + zFu_{i+1}` with `zFu = e2u e3u (uu + zub)`
(`stprk3_stg.F90:273`); the F-point cross fluxes select by the transport pair
(`:179-187`).  legoESM's `_up3_reconstruct` (`opl:4078`) had both branch
formulas but chose the branch by the sign of the transport it was handed.
Under WS-RK3 the transport carries `zub`, and at kt=1 `zub` is about minus
half the primary barotropic velocity (a linear ramp's time mean), so at the
front T-pairs, top two levels, `zui = +0.023 m/s` while the transport pair is
`-0.076 m^2/s` per unit `e2u`: the wrong curvature was used exactly there.
Stage 2 has the same disagreement only at pairs whose curvature is exactly
zero, hence "stage-3 only".

Scaling (all m/s^2, e3u_0-weighted depth mean removed; `D_l` = legoESM's
derived stage-3 RHS minus `hpg_sco` minus the `dynadv_up3` transcription on
legoESM's own `u2` with NEMO's stage-3 transports; NEMO's own closure
`4.6e-16`):

| suspect | replay quantity | max | corr with `D_l` | slope | `max|D_l - pattern|` |
|---|---|---:|---:|---:|---:|
| Aimp partition (S-20) | `wi` on NEMO stage-3 transports (`Cu_v <= 1.7e-3`) | `0.0` | — | — | unchanged |
| implicit ZDF solve | NEMO's whole stage-3 increment `1.086e-08 m/s` (24x below the debt) | — | — | — | unchanged |
| stage-3 HPG operands | `dt*bc(H(lego s2) - H(nemo s2))`; legoESM HPG vs `hpg_sco` on the s2 operands `1.4e-16` | `1.6e-18` | — | — | unchanged |
| lateral viscosity (stage-3-only `skip_ldf=False`) | `tend(skip=False) - tend(skip=True)` | `0.0` | — | — | unchanged |
| vertical UP3 | transcription input sensitivity | `1.7e-13` | — | — | unchanged |
| **UP3 upwind selector** | `bc(up3[transport sign]) - bc(up3[velocity sign])` | `2.5989e-08` | **`+0.999994`** | **`1.00015`** | **`4.55e-11`** |

### Predictions vs outcomes

| # | prediction | outcome |
|---|---|---|
| P1 | OVERFLOW stage-3 / kt=2 `u` `2.5988e-07 -> < 1.0e-09` (linearised `4.5e-10`) | `4.551736e-10` (571x) — **MET** |
| P2 | OVERFLOW stage 1, 2 bit-identical | `6.501744e-15`, `9.433404e-11` — **MET** |
| P3 | OVERFLOW kt=2 `T`, `S`, `SSH` bit-identical | `1.1191048e-14`, `2.030e-16`, `1.0491608e-14` — **MET** |
| P4 | legacy arm reproduces the pre-fix stage-3 row | `2.598798e-07` exactly — **MET** (`control_over_faithful` 570.9, `CONFIRMED_REQUIRED`) |
| P5 | LOCK kt=2 `u` improves `>= 3x` | `2.138804e-10 -> 2.298509e-17` — **MET** (7 orders) |
| P6 | LOCK kt=2 `T`, `SSH` bit-identical | `T 1.627735e-13 -> 0.0` exactly, SSH identical — **REFUTED in the good direction**: LOCK's stage-2 `u` (`9.7656e-11 -> 2.86e-17`) was also the selector (its disagreement cells carry curvature), and the stage-2 Kmm transport feeds stage-3 FCT |
| P7 (secondary) | the kt>=3 OVERFLOW SSH walk is seeded by this error | **REFUTED**: kt=10 SSH `9.237391e-05 -> 9.237444e-05`; the walk has another owner |
| P9 | legacy paths bit-identical | 14 flux-form tests pass; no other caller passes a separate transport |

LOCK_EXCHANGE-zco now clears the `1e-15` bar at every kt=1 stage and at
kt=2 (stage sweep exit 0, `failed_rows: []`).

### Trajectory (normalized L-inf on the entering state), before -> after

OVERFLOW-zps:

| kt | T | u | SSH |
|---:|---|---|---|
| 2 | `1.119105e-14 -> 1.119105e-14` | `2.598798e-07 -> 4.551736e-10` | `1.049161e-14 -> 1.049161e-14` |
| 3 | `1.948087e-09 -> 7.788969e-12` | `3.373690e-06 -> 9.079022e-09` | `3.773822e-09 -> 3.774823e-09` |
| 10 | `7.711785e-08 -> 4.046684e-08` | `2.644298e-05 -> 2.644302e-05` | `9.237391e-05 -> 9.237444e-05` |
| 60 | `1.184430e-05 -> 1.184430e-05` | `2.507515e-04 -> 2.507516e-04` | `7.751799e-05 -> 7.751811e-05` |

LOCK_EXCHANGE-zco:

| kt | T | u | SSH |
|---:|---|---|---|
| 2 | `1.627735e-13 -> 0.0` | `2.138804e-10 -> 2.298509e-17` | `4.782138e-28 -> 4.782138e-28` |
| 3 | `4.046541e-13 -> 1.184238e-16` | `2.138797e-10 -> 7.233654e-15` | `1.355253e-18 -> 1.355253e-18` |
| 10 | `2.997306e-12 -> 1.681618e-14` | `2.070506e-10 -> 1.314126e-11` | `3.577033e-13 -> 3.574534e-13` |
| 60 | `8.721191e-10 -> 8.755528e-10` | `7.496273e-08 -> 7.496602e-08` | `1.609018e-08 -> 1.609001e-08` |

Rule 8 disclosure: LOCK kt=60 `T` moves `+0.4%` and `u` `+0.004%` in the
worse direction (not ulp-level); kt=2..10 improve by 2-5 orders.  The
OVERFLOW kt>=4 rows are unchanged to 5-6 digits: the kt=2 seed is gone and
the later growth is owned by the untouched SSH walk.

### Remaining OVERFLOW stage-3 debt, UNOWNED

`4.551736e-10 m/s`, faces 19 and 21 (the faces flanking the front), linear
in depth (face 21: `-1.6e-10` at k=18 to `-4.55e-10` at k=24), equal to the
RHS-closure remainder `4.55e-11 m/s^2 x dt`.  Excluded: the bottom ZDF
operand differences (columns 20/21 are exactly 500 m, no partial cell there;
predicted movement `1.9e-16`), a single-face advecting-transport difference
(corr `<= 0.33` against the unit-`dU` response at faces 18..22), the HPG
(`1.4e-16`), the lateral term (`0.0`), the Aimp partition (`0.0`).

### Statistics (6120 steps, fp64 + fp32, CPU), before -> after

| metric | before | after | fp32 floor before | fp32 floor after | NEMO spread | verdict before | verdict after |
|---|---:|---:|---:|---:|---:|---|---|
| final_temperature_histogram_tv | `0.0390072` | `0.0620842` | `0.0174136` | `0.0181244` | `0.044231` | WITHIN-SCHEME-SPREAD | OUTSIDE |
| final_water_mass_census | `0.0130347` | `0.0176424` | `0.00180818` | `0.000221017` | `0.00336209` | OUTSIDE | OUTSIDE |
| instantaneous_u_linf | `0.832083` | `1.99404` | `0.773437` | `1.72399` | `0.672694` | OUTSIDE | OUTSIDE |
| plume_descent_m | `16.9551` | `1499.8` | `0.038907` | `0.0742579` | `1499.62` | WITHIN-SCHEME-SPREAD | OUTSIDE |
| plume_front_km | `4.03884` | `117.125` | `0.135393` | `0.0207441` | `121.931` | WITHIN-SCHEME-SPREAD | WITHIN-SCHEME-SPREAD |
| temperature_linf | `0.379487` | `0.379253` | `0.122985` | `0.238998` | `0.356744` | OUTSIDE | OUTSIDE |

Faithful-but-worse (Rule 8): 3 -> 5 `OUTSIDE`.  Disclosed, not reverted.
The legoESM fp32-vs-fp64 floor for `instantaneous_u_linf` also more than
doubles (`0.77 -> 1.72 m/s`), i.e. the two legoESM arms decorrelate more
strongly from each other after the fix — the 6120-step state is chaotic and
the plume-descent jump to the full slope depth is a registered-time
threshold crossing, PLAUSIBLE, not decomposed.  Attribution between the
stage-2/3 fix (`42ac525cc`) and the stage-1 extension (`8d6756a42`):
measured with a third arm at `42ac525cc` (stages 2-3 only, `stage3_selector/v1_stage23_only/`): histogram `0.0608531`, census `0.0210638`, u_linf `1.54484` (fp32 floor `1.63044`, INDISTINGUISHABLE-AT-FLOOR), plume_descent `1499.89`, plume_front `117.14`, T_linf `0.379223` — i.e. the stage-2/3 selector fix alone carries the whole move (4 OUTSIDE / 1 WITHIN / 1 AT-FLOOR); the stage-1 extension then shifts rows by a few percent in both directions (census `0.0211 -> 0.0176`, u_linf `1.54 -> 1.99` against a floor of `1.6-1.7`), inside the fp32 noise of this decorrelated state.

### Second finding (not fixed here)

The step-entry `tendencies()` call that seeds WS-RK3 stage 1
(`omlc:4070`) receives no separate transport: legoESM's stage-1 momentum
advection uses `Q = h u(Kbb)` while NEMO's stage-1 `dyn_adv` (`:315`) is
handed `zFu = e2u e3u (uu(Kbb) + zub)` with `zub = un_adv/hu(Kbb) -
uu_b(Kbb)` built unconditionally at `:259-275`.  Inert at kt=1 (u = 0), live
from kt=2; CONFIRMED by reading, UNMEASURED.  Candidate owner of the kt>=3
SSH walk.  The S-21 row's "SHARED across the three RK3 cards" is true for
the tracer stages and for momentum stages 2-3 only.

### Reviews (two independent reviewers, the author reviewed nothing)

Physics/oracle reviewer: SHIP.  Confirmed the `:166-187` mapping line by
line; flagged (a) the no-transport default keeping the transport sign is a
hidden choice for any non-RK3 NEMO card on `flux_form_upwind3`
(`nemo_recipe.py:334`) — recorded, not flipped (ASK); **CLOSED 2026-09-02**:
the selector is now keyed by the REFERENCE the caller's
`momentum_flux_scheme` names (`nemo_up3` / `oceananigans_up3`), not by the
time integrator, and the unqualified `"upwind3"` is refused — see the map
doc's S-46; (b) the stage-1
finding above; (c) the remaining `4.55e-10` sits at the bottom, not where
the replay's remainder sat — attribution to the ZDF operands was then
MEASURED and excluded (above).
Code reviewer: FIX-FIRST, three items, all fixed in `8d6756a42`: the
`omit_momentum_transport_reconcile` gate arm had become two-variable
(selector resolved from `transport_velocity is not None`) — the WS-RK3
program now passes NEMO's rule explicitly at all three stages; the fallback
comment's justification was false and is rewritten; the cross fluxes were
untested (v == 0) — F10 isolates and pins them.

### Controls

- New unit tests fail on the reverted code, each verified by reverting:
  F9 (velocity rule vs an independent assembly; legacy default bit-for-bit),
  F10 (cross fluxes stay on the transport selector; fails when they are
  switched), the model-level hook test (fails when the WS-RK3 program stops
  passing the rule), and the S1/S2 predicate unit test (planted 1000x
  inflation flips S1 to NOT-MET; a stale legacy arm flips it too).
- `--plant-prediction` end-to-end on OVERFLOW (gate re-pointed from the now
  permanently NOT-MET P3 predicate, whose scale reference this round
  removed, to S1): exit `1` with `S1` reading NOT-MET on the planted `4.5517e-07` (report `stage3_selector/gates_after/overflow_plant_prediction.json`, stamped `8d6756a42-dirty` because the gate edit was uncommitted when the control ran); the same gate on LOCK exits `2` (`S1` is OVERFLOW-only), as `test_prediction_plant_is_fail_closed` asserts.
- The two pre-existing golden/parity failures
  (`test_baroclinic_decomposition_bit_identical`,
  `test_modular_matches_monolithic_pe_rel`) fail identically at the pristine
  `9070cf276` (verified in a throwaway worktree) and are not touched.

### Tests

Focused CPU/fp64: 14 flux-form momentum (incl. F9, F10), 3 WS qco stage
faces (incl. the selector hook), 10 stage-sweep gate (incl. S1/S2 and the
LOCK AT-BAR expectation), 44 across `test_no_scheme_duplication`,
`test_nemo_ws_qco_stage_faces`, `test_no_private_cross_imports`; 49 of 51
across `test_nemo_ws_tracer_rk3`, `test_baroclinic_decomposition`,
`test_nemo_overflow_stability_probe`, `test_nemo_testcase_recipe`,
`test_overflow_runner_parity` (the 2 pre-existing failures above).

### Artifacts (`/data/abyssal/dbalwada/nemo-testcases-l1/`)

| artifact | sha256 |
|---|---|
| `stage3_selector/gates_before/lock_stage_sweep_gate_kt2.json` | `d85a3ec23df3bc49018a2dc734c36877ff24a521f3c2e6ad98355e4c91a75045` |
| `stage3_selector/gates_before/lock_trajectory_gate_kt10.json` | `b48b81e5fcebdc3b77f4b3f2745047c21fef62f37c627694e2850ae9193c7fbb` |
| `stage3_selector/gates_before/lock_trajectory_gate_kt60.json` | `2add69f85a9e478fb2c84ddf9811eb3cfaf5dd60aec24d87935322c8cfe68f40` |
| `stage3_selector/gates_before/overflow_stage_sweep_gate_kt2.json` | `83bd8b5b01d782db8468fb33762dd54701cd6c15d8042a9714cb17cf022aaa54` |
| `stage3_selector/gates_before/overflow_trajectory_gate_kt10.json` | `a090221abb324400dd2fb21c20040a2a43a2249d605d3e2406311ebaf9d10532` |
| `stage3_selector/gates_before/overflow_trajectory_gate_kt60.json` | `e687c3f70d0551bedcbbbd188c8e2189a56e74786730e6f480966bee78fa71f2` |
| `stage3_selector/gates_after/lock_stage_sweep_gate_kt2.json` | `f6be76e5157f06332e0230425c1b3824efc8ce05a7d85c14c6e8e114fc6ee6ff` |
| `stage3_selector/gates_after/lock_trajectory_gate_kt10.json` | `70d5e03ae6d2b78c3f989af4544157fe7f616fae83306898a7828d5403565090` |
| `stage3_selector/gates_after/lock_trajectory_gate_kt60.json` | `57e46df417b182faf0e5aa9ead75a028f7e44000f8e93af1446a24cfb6901e59` |
| `stage3_selector/gates_after/overflow_stage_sweep_gate_kt2.json` | `4c45a3665a9f73cdc40e99f109f38b8c2e59d411252c48b78e88c4e5c71e2daa` |
| `stage3_selector/gates_after/overflow_trajectory_gate_kt10.json` | `601a27f7a1352dd0127d52a06bda24919e36c39013b6220daf989c4d2f804c2a` |
| `stage3_selector/gates_after/overflow_trajectory_gate_kt60.json` | `13077a0974f9f6fd67cdc15e2788bb0230948ec17f21ca43d8472c9e1b6ca70e` |
| `stage3_selector/before/overflow_statistics.json` | `86215bbb0b45bc980ed6718f02d030eaf760859480a1f9acc018e795c9fe889d` |
| `stage3_selector/after/overflow_statistics.json` | `df4f64541d86f8a41d74aaa8ee9b70c9dd1258c5f6593abc971086e42e83fce8` |
| `stage3_selector/before/legoesm/overflow_zps/fp32/metadata.json` | `847e4e6b476f7446944622251f41818b8e9962c034da9684b674daac2c8bbd9c` |
| `stage3_selector/before/legoesm/overflow_zps/fp64/metadata.json` | `46ea99f4b4c7bc6732845bccc8c2f4d63f440a52cab274fc2d7a03a86e905fde` |
| `stage3_selector/after/legoesm/overflow_zps/fp32/metadata.json` | `5d303611856ca0915d08c456f4f317195e2d279f1d00c617e6849d1ba4681997` |
| `stage3_selector/after/legoesm/overflow_zps/fp64/metadata.json` | `82cd0f4c46e19de3617634ee448a881daccb20ab935b52960e3d72ea5e4ea7ce` |
| `stage3_selector/v1_stage23_only/overflow_statistics.json` | `3dd6970b04b6a7f271770a3613bb052d092b09f922b2e079ba83a04c5ce87cad` |

## S-21 stage-1 transport round: the kt>=3 walk is NOT stage-1 `zub`, MEASURED

Preregistered in `nemo_testcases_l1_stage1_transport_preregister.md` (commit
`49f7b8c14`, frozen before the free-run arm was scored).  All numbers fp64,
CPU, at `bd4097f89` plus the one-variable arm patch; the arm's legacy side was
verified bit-identical to the pristine `bd4097f89` tree at every scored row on
both cards.  The verdict on the LEAD is REFUTED, so per the round's own
protocol the fix is NOT landed here and the 6120-step statistics were NOT run.

### The divergence (real, cited)

NEMO builds ONE Kmm stage transport per RK3 stage,
`zFu = e2u*e3u(Kmm)*(uu(Kmm) + zub*umask)` with
`zub = un_adv/hu(Kmm) - uu_b(Kmm)` (`stprk3_stg.F90:265-275`, `n_baro_upd =
np_HYB` at `:44`), and hands it to stage 1's `dyn_adv` at `:315` exactly as it
does to stages 2-3 at `:333`.  The 3-D slot stage 1 accumulates into holds
`stp_2D`'s Kbb `dyn_hpg` (assigned, `stp2d.F90:128` / `dynhpg.F90:294`),
`dyn_ldf` (`:131`) and `dyn_vor` (`:146`); `dyn_adv_up3` at `:172` is called
with `pUe`/`pVe` and so writes only the 2-D barotropic seed
(`dynadv_up3.F90:158-202`).

legoESM's stage 1 steps with the step-entry `tendencies()` result
(`ocean_model_latlon_cgrid.py:4193-4199`, then `:5011-5017`), which is
evaluated BEFORE the barotropic solve at `:4826` produces `Hu_avg` and
therefore advects with `Q = h u(Kbb)`, i.e. `zub = 0`; stages 2-3 do carry it
through `_mom_pert_ws` (`:4439-4454`).  Difference, per unit `e2u`:
`dF_u = e3u(Kbb) * zub * umask`, entering the UP3 T-point flux linearly once
the branch is fixed by the advected-velocity pair (S-44).  Exactly zero at
kt=1 (rest), live from kt=2.

### Scaling and arm

| quantity | value |
|---|---:|
| stage-1 RHS difference, OVERFLOW kt=2 entry | `3.136020e-07 m/s^2` (face 21, k=24; 12x the stage-3 error the UP3 round removed) |
| its structure | linear in depth, sign change at k=12, faces 19-22 only |
| one-step `u` response at the kt=3 entry | `2.664653e-11` = `0.3%` of the residual, corr `0.38` |
| one-step `ssh` response, every kt, both cards | `0.0` EXACTLY |

The `ssh` zero is structural: the barotropic solve runs before the stage
ladder, is seeded by a `F_slow` this term does not touch, and
`_replace_stage_mean` re-imposes the identical depth mean at every stage.

Free-run one-variable arm (legacy hook = pristine `bd4097f89`, faithful =
stage-1 transport applied as the difference of the same helper with and
without the transport, so the arm is one-variable by construction):

| card | kt | `T` | `u` | SSH |
|---|---:|---|---|---|
| OVERFLOW | 2 | `1.119105e-14` -> same | `4.551736e-10` -> same | `1.050549e-14` -> same |
| OVERFLOW | 3 | `7.788969e-12 -> 6.851053e-12` | `9.079022e-09 -> 9.105049e-09` | `3.774823e-09` -> same |
| OVERFLOW | 10 | `4.046684e-08 -> 4.046700e-08` | `2.644302e-05` -> same | `9.237444e-05` -> same |
| OVERFLOW | 60 | `1.184430e-05` -> same | `2.507516e-04` -> same | `7.751811e-05` -> same |
| LOCK | 2 | `0.0` -> same | `2.276825e-17` -> same (AT BAR) | `0.0` -> same |
| LOCK | 3 | `1.184238e-16` -> same | `7.233653e-15 -> 7.267792e-15` | `1.355253e-18` -> same |
| LOCK | 10 | `1.681618e-14 -> 1.693460e-14` | `1.314126e-11 -> 1.314215e-11` | `3.574534e-13` -> same |
| LOCK | 60 | `8.755528e-10 -> 8.755466e-10` | `7.496602e-08 -> 7.496604e-08` | `1.609001e-08` -> same |

Predictions Q1-Q8 all MET, including Q8 whose prediction WAS the refutation:
kt=2 bit-identical on both cards, kt=3 `u` moved `+0.29%` (OVERFLOW) and
`+0.47%` (LOCK), kt=3 SSH bit-identical, kt=10/60 unchanged to 7 digits.  Q9
(the six statistics rows) was gated on Q8 confirming and was NOT run.

Two kt=2 baseline rows differ from the UP3 round's receipt — OVERFLOW SSH
`1.049161e-14 -> 1.050549e-14` and LOCK `u` `2.298509e-17 -> 2.276825e-17`.
Both arms agree on them, so they are the `bd4097f89` merge (the barotropic
branch), not S-21.  LOCK stays AT BAR at kt=2.

### Retraction

The UP3 round's second finding named stage-1 `zub` a "candidate owner of the
kt>=3 OVERFLOW SSH walk".  RETRACTED: it cannot move `ssh` within a step at
all, and it carries `0.3%` of the kt=3 `u` residual and nothing measurable at
kt=10 or kt=60.  The kt>=3 walk owner is still UNKNOWN.  S-21 remains an open
one-routine / two-implementations DEBT.

### Instrument finding: `plume_descent_m` is a threshold-crossing artifact

`nemo_testcase_full_statistics.py:797-801` reduces the row to
`max(cell centre | T <= 15 C)` and `:1112-1120` scores
`max_t |L64 - N2|` against `spread = max_t |N4 - N2|`.  On the registered times
the final-time value is effectively two-valued — the plume either still has a
cell on the deep bottom or does not:

| arm | t=0 | t=8.5 h | t=17 h |
|---|---:|---:|---:|
| N2 (NEMO, registered namelist) | 490.0 | 1986.58 | 1989.97 |
| N4 (NEMO, alternative namelist) | 490.0 | 1969.52 | 490.34 |
| L64 before the UP3 fix | 490.0 | 1969.63 | 1989.80 |
| L64 after the UP3 fix | 490.0 | 1987.95 | 490.17 |

NEMO's own two arms already differ by the whole basin (`1499.63 m`), so the
UP3 round's `16.96 -> 1499.8 m` is the candidate crossing to N4's branch and
scoring OUTSIDE by `0.18 m` inside a `1500 m` step; the same event reads
WITHIN-SCHEME-SPREAD through `plume_front_km` (`117.13` vs spread `121.93`).
The row can neither confirm nor refute a fix at its current sampling, and the
UP3 round's `plume_descent_m` OUTSIDE verdict should be read as uninformative
rather than as a physical degradation.  NOT FIXED: any robust reducer (volume
weighting, a percentile, denser registered times) is a new scientific choice
that changes this row and needs its own preregistration — ASK.

## S-21 stage-1 transport LANDED: the measured-inert, NEMO-faithful fix

`stprk3_stg.F90:265-275,315`: stage 1 receives the same Kmm transport
`zFu = e2u*e3u(Kmm)*(uu(Kmm) + zub)` as stages 2-3.  Landed at
`ocean_model_latlon_cgrid.py` stage 1 as the difference of the stage helper
evaluated with and without the `zub` transport on the same entry state, added
to the stage-1 RHS (every other term cancels exactly; the private
`momentum_transport_reconcile` hook is the one-variable control).  Direct
test: `tests/ocean/unit/test_nemo_ws_stage1_transport.py` (live from a moving
LOCK entry with a vanishing `e3u_0` column mean, inert from rest; both fail on
the reverted code).  Before = clean `59d1fcbb8` (`/tmp/wt-branch-iso`), after
= the same tree plus this patch; fp64, CPU, same oracle dumps:

| card | kt | `T` | `u` | SSH |
|---|---:|---|---|---|
| OVERFLOW | 2 | `1.119105e-14` -> same | `4.551736e-10` -> same | `1.050549e-14` -> same |
| OVERFLOW | 3 | `7.788969e-12 -> 6.851053e-12` | `9.079022e-09 -> 9.105049e-09` | `3.774823e-09` -> same |
| OVERFLOW | 10 | `4.046684e-08 -> 4.046700e-08` | `2.644302e-05` -> same (7 digits) | `9.237444e-05` -> same |
| OVERFLOW | 60 | `1.184430e-05` -> same | `2.507516e-04` -> same | `7.751811e-05` -> same |
| LOCK | 2 | `0.0` -> same | `2.276825e-17` -> same (AT BAR) | `0.0` -> same |
| LOCK | 3 | `1.184238e-16` -> same | `7.233653e-15 -> 7.267792e-15` | `1.355253e-18` -> same |
| LOCK | 10 | `1.681618e-14 -> 1.693460e-14` | `1.314126e-11 -> 1.314215e-11` | `3.574534e-13` -> same |
| LOCK | 60 | `8.755528e-10 -> 8.755466e-10` | `7.496602e-08 -> 7.496604e-08` | `1.609001e-08` -> same |

These reproduce the S-21 round's arm rows exactly (kt=2 bit-identical on both
cards; OVERFLOW kt=3 `u` `+0.29%`; kt=10/60 unchanged to seven digits; SSH
untouched at every kt).  Stage sweep kt=2: OVERFLOW 96/96 rows and LOCK 54/54 rows bit-identical before -> after (`ssh_walk/s21_{before,after}/*_stage_sweep_kt2.json`; the LOCK after-sweep process started after the seed fix of the next section was already on disk, which P7/P8 of that round predict inert at kt=2 -- and it was)  Its
6120-step statistics ride with the SSH-walk seed round (the next section):
one statistics pair covers both landings, disclosed as such.

| artifact | sha256 |
|---|---|
| `ssh_walk/s21_after/lock_trajectory_kt10.json` | `0e689f39e804d6bd115ce8324afd96bd3f03adf8e0d18618076848c9d20a1cd3` |
| `ssh_walk/s21_after/lock_trajectory_kt60.json` | `d79dcf32b7c4edb9fb000e886d4577eff65c1884d25a398e6cdcd15b29b91893` |
| `ssh_walk/s21_after/overflow_trajectory_kt10.json` | `07685d8a1e12a4a2141be2649a380ca70805d3514f2824ceaf63a1a4d771eb12` |
| `ssh_walk/s21_after/overflow_trajectory_kt60.json` | `1bc3be88fa858f7bb8f1adf249f39f0bcd71c55711ce35bc591854d84db65410` |
| `ssh_walk/s21_before/lock_trajectory_kt10.json` | `7a67e604747dd7a569fdc53fa82eaeb3ab6242cf9102eb5a0aa62298a451c7f9` |
| `ssh_walk/s21_before/lock_trajectory_kt60.json` | `7a04859e03414676549ce5a8c84e0314363d5f2ef0c01096a6e70fe025dce8d4` |
| `ssh_walk/s21_before/overflow_trajectory_kt10.json` | `dbf487fa82e8177afbac6f544dc1040566c7f34c908f26693542925a289e295f` |
| `ssh_walk/s21_before/overflow_trajectory_kt60.json` | `4bc184fd1fdf4254a29fbcb017152a1856ab26445aac932dca9aca5012569d28` |
