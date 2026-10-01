# NEMO testcase Lane 4 — ORCA2 Phase-2l review fix and handoff

Date: 2026-09-06

Parent: `3e425e24ded84e9bd1cbba7af492aaec43e99170`

Status: **STOP AT `GYRE_OWNER_SHARED_TRACER_ADVECTION`.**  The alleged
factor-three production WZV clock error is retracted under Rule 11.  A compiled
production step proves that legoESM pairs the full external SSH endpoint with
the full 10,800 s denominator; the old arm instead paired NEMO's materialized
stage-1 HYB SSH with that denominator, a state/clock combination executed by
neither model.  A card-owned one-column error in the tripolar U-face metric
layout was corrected and pinned without changing V metrics or `f_v`.  The
first remaining production debt after the oracle-supplied external endpoint
is shared stage-transport source association.  With the exact NEMO
`zFu/zFv/zFw` substituted, the next boundary is shared stage-1 centered tracer
advection, also over bar.  Both debts are handed to GYRE ownership without a
Lane-4 arithmetic change.  BBL and TKE/EVD/IWM were not entered.

SI3 exchange fields remain `ORACLE_SUPPLIED`; SI3 dynamics and thermodynamics
remain `UNMEASURED_PENDING_ICE_MERGE`.

## 1. Crash recovery and evidence integrity

The host filled `/tmp` and killed the preceding process at approximately
02:42 UTC on 2026-09-05.  Recovery was fail-closed:

- the ordinary worktree remained at `3e425e24ded`; the separate localgit branch
  was clean at `78811d028` when recovery began;
- `git fsck --full --no-dangling` was clean;
- all four repository paths touched before the crash matched their committed
  localgit blobs, parsed/compiled, and passed `git diff --check`;
- the pre-crash production-W JSON parsed, but was not trusted: a fresh run was
  byte-identical, both SHA-256
  `0dfcf1aab12e2066a6db5b7c3fc95f35968061c97bedd84fe7137d7fb4a1af4b`;
- no truncated repository artifact and no lost completed measurement was
  found; the only incomplete cross-card state contained no citable output;
  and
- no NEMO build, MPI run, or handed-off `run.sh` was in flight.

No campaign tree or evidence root was deleted during recovery.

## 2. Resolved card identity

The shipped ORCA2 selector is `ln_dynadv_vec=.true., nn_dynkeg=0` at
`cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg:346-347`.  `dynadv.F90:78-90` and
`:106-146` resolve this to vector-invariant momentum with C2 kinetic-energy
gradient.  The VARIANT oracle's `ocean.output:1315-1316` reports the same
selection.  The card now fails validation unless it selects exactly
`momentum_advection="vector_invariant"` and `ke_gradient_scheme="c2"`.

The binding selector plant changes only C2 to Hollingsworth and is rejected by
`validate_nemo_testcase_card`.  The compiled production gate records:

| selector | NEMO | legoESM |
|---|---|---|
| momentum form | `ln_dynadv_vec=T` | `vector_invariant` |
| KE gradient | `nn_dynkeg=0` | `c2` |

This is a card-selection fix, not a new numerical arm.  Under vector form,
`stprk3_stg.F90:286-294` does not compute or consume momentum `ww` at stage 1;
the stage-1 W measured here is the separately executed tracer transport path
at `traadv.F90:220-227`.

## 3. Compiled-production W adjudication

The gate is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2l_production_w_gate.py`.
It executes the complete full-global ORCA2 card through the production JIT on
CPU with fp64 and scalar-libm.  A private post-step hook exposes the actual
stage-1 W that production handed to tracer advection; it does not alter any
public card or returned production state.

The external-mode endpoint is already a registered shared debt, so the exact
NEMO final SSH and time-mean transports are substituted.  Three other
upstream, discarded tendencies are made constructible only inside this
diagnostic: EOS80 `bn2` uses the TEOS10 helper, tripolar EEN `nemo_avg4` uses
the constructible `min` arm, and tripolar `nemo_div_curl` viscosity is zero.
None can reach the substituted external endpoint or the exposed stage-1
transport.  These substitutions certify neither those operators nor the
external mode.

The comparison is on 228,641 live rank-zero T cells over 30 levels, excluding
the east MPI support column, west support column, and north-fold support row
whose remote operands rank zero did not dump.

| compiled boundary | unequal / n | maximum absolute difference | result |
|---|---:|---:|---|
| card `e2u` after U-face layout repair | 0 / 8,568 | 0 | AT BAR |
| card `e1v` control | 0 / 8,589 | 0 | AT BAR |
| production `zFu` | 102,431 / 228,641 | 4.656612873077393e-10 | DEBT |
| production `zFv` | 108,307 / 228,641 | 9.313225746154785e-10 | DEBT |
| production tracer `ww` | 220,589 / 228,641 | 4.6872685718684845e-19 m/s | DEBT |
| valid full-endpoint/full-step clock replay | 189,969 / 228,641 | 3.48342299558331e-20 m/s | DEBT |

The first production over-bar boundary is therefore
**`GYRE_OWNER_SHARED_STAGE_TRANSPORT_SOURCE_ASSOCIATION`**, not a WZV clock.  NEMO
forms `zub/zvb` at `stprk3_stg.F90:259-270`, then materializes
`metric*e3(Kmm)*(velocity+barotropic correction)` at `:272-275`.  legoESM's
single shared helper is `_nemo_ws_stage_transport` at
`ocean_model_latlon_cgrid.py:1338-1435`, with the corresponding reciprocal,
subtract, corrected-velocity, and metric-product statements at `:1364-1387`.
The V2 record is
`oracle_stage1_wzv_operands_kt00000001.bin`, SHA-256
`245be2ea348002b93358e5dd723f0b84198af47c3d38f0bc0bb1acb0fc3100fe`;
its `pFu/pFv` frames are the already-materialized stage-1 transport operands,
while `ww/pFw` are the post-`wzv` products at `(Kbb,Kmm,Kaa)=(1,1,3)`.

The remaining transport error is at most four ULP after exact card metrics;
it is shared and is registered for the GYRE lane.  Lane 4 does not change
`_nemo_ws_stage_transport`.

Before that shared boundary, the probe found `grid.dy_u[:,1:]` differed from
NEMO `e2u` in 1,961 / 8,568 wet faces, while the unshifted native array was
0 / 8,568.  `create_tripole_grid` had appended a duplicate after NEMO's native
east-face array even though legoESM U index 0 denotes the west periodic face.
The `ORCA2_OWNER_TRIPOLAR_U_FACE_LAYOUT` repair now prepends the native last U
face and places native `e1u/e2u` and U rotations at indices `1:`.  A synthetic
test pins that map, both redundant endpoints, unchanged V metrics, unchanged
generic `f_v`, and U rotations.  The ORCA2 entry gate remains exact for T, S,
u, v, and SSH; its JSON SHA-256 is
`893f38dfa12c0ee7054e4ebfb2a23f23b16fbfe1f96c42f17137112210228d12`.

## 4. Rule-11 clock retraction

NEMO first materializes the HYB stage SSH
`2/3*ssh(Kbb)+1/3*ssha` at `stprk3_stg.F90:141-145`, sets
`rDt=rn_Dt/3` at `:118-124`, and consumes the pair in
`sshwzv.F90:330-336`.  legoESM instead builds a full endpoint delta and passes
the full step through `_stage_transport_kw` at
`ocean_model_latlon_cgrid.py:5852-5867`.  These are algebraically the same
real quotient, with different source association.

Phase 2k's 233,341 / 233,341, 7.329623319094706e-05 m/s arm held the
materialized NEMO HYB delta but changed only its denominator to 10,800 s.  It
therefore measured an impossible mixed state/clock pair, not production.  The
old gate no longer prints a production clock owner or lets that arm set its
verdict.  It retains the observation only as
`retracted_invalid_mixed_state_full_dt_ablation` with disposition
`RETRACTED_RULE_11_NOT_A_PRODUCTION_CONFIGURATION`.  The Phase-2k receipt is
marked superseded in place.  There is no
`GYRE_OWNER_SHARED_WZV_STAGE_CLOCK` handoff.

The valid clock-only replay is over bar only at source-association scale
(maximum 3.48e-20 m/s).  It is registered as
`GYRE_OWNER_SHARED_WZV_STAGE_SSH_ASSOCIATION`, downstream of the earlier
stage-transport source-association debt.  GYRE is **UNMEASURED on W**, not
insensitive: its prior gate scores no vertical-velocity row, and vector form
explicitly skips stage-1 momentum `ww` at `stprk3_stg.F90:286-294`.  Its
bit-exact stage-1 T/S can coexist with a tiny W association difference because
the resulting tracer update can round to the same stored binary64 values.

Both production-W plants are binding: the C2 selector plant is rejected, and
a one-ULP W plant traverses `score`, exits 1, and reports exactly
1 / 228,641.

## 5. Rule-12 runoff cross-card register

The ORCA2-owned runoff extension adds an optional
`runoff_mass_flux` input to the one shared WZV program.  GYRE, LOCK_EXCHANGE,
and OVERFLOW supply `None`, which selects the pre-existing branch.  The
pre-change reference is localgit commit `e4f62e664` (immediately before runoff
commit `020a5043b`); the current side contains the same production path plus
the optional runoff branch.

| card / kt=1 gate | measured result | disposition |
|---|---|---|
| LOCK_EXCHANGE-zco stage sweep | 0 moved cells, 0 ULP in all 9 certified rows | VERIFIED |
| GYRE round-13 transport candidate against round-21 scalar oracle | pre/current JSON byte-identical; 0 movement in every reported operand and candidate row | VERIFIED |
| OVERFLOW-zps faithful stage sweep | no completed artifact after two fresh compile attempts; the final attempt was stopped after prolonged no-output JIT compilation | UNMEASURED_COMPILER_RESOURCE |

The LOCK oracle-relative comparison itself reports `PASS`, zero worsened,
improved, or moved cells, and zero maximum worsening ULP in all nine rows.  The
GYRE pre/current JSON share SHA-256
`c56aea3ba36fc0faa8a7471cae1fe1a1704765082605f429800c4e3eac5f4f51`.
No result is inferred for OVERFLOW; its `None`-branch source proof is not
promoted to measurement.

The focused test mentioned in Phase 2k is exactly
`tests/ocean/unit/test_nemo_qco_generic_mesh_operands.py::test_arm_is_constructible_on_the_certified_l1_cards`.
It fails before WZV because its partial fixture supplies some but not all raw
NEMO mesh operands (`nemo_hu_0`, `nemo_hv_0`, `nemo_e1e2t`, `nemo_e1e2u`,
`nemo_e1e2v`, `nemo_e2u`, `nemo_e1v`).  The runoff delta touches only the
downstream optional WZV input and does not touch
`vertical.py::nemo_qco_resolved_mesh_operands` or fixture construction.

## 6. Oracle-supplied transport continuation

The ladder continues past both shared production transport debts by
substituting NEMO's exact `zFu/zFv/zFw` from
`oracle_rktracer_operands_kt00000001_s1.bin`, SHA-256
`213d2fd9a4ea422be2bf2f1d7e96f70e1fb87aa6823fd7a0a427855e702701ce`.
The record has magic `NEMO_L2_RKTRA_1`, binary64, kt/stage 1,
`(Kbb,Kmm,Krhs,Kaa)=(1,1,3,3)`, dimensions `(94,152,31)`, and a derived
6,686,784-value payload.  The hook embeds rank zero in the full global card;
scoring uses the same 228,641-cell support-safe wet interior as the production
W gate.

Resolved `ln_traadv_fct=T`, `nn_fct_h=2`, `nn_fct_v=2`, and `nn_fct_imp=1`
are printed at `ocean.output:1273-1276`.  Under RK3,
`traadv.F90:280-283` disables the limiter at stages 1 and 2 and
`:355-365` executes the centered FCT2 precursor; the two-step FCT operator in
`traadv_fct.F90:153-189` is executed only at stage 3.  The shared legoESM
precursor is `ocean_model_latlon_cgrid.py:1582-1621`.

| stage-1 centered precursor | unequal / n | maximum absolute difference | result |
|---|---:|---:|---|
| T accumulator | 180,882 / 228,641 | 9.952637130238029e-21 | DEBT |
| S accumulator | 190,802 / 228,641 | 1.4558378780933287e-20 | DEBT |

This is the next first over-bar boundary after operand substitution:
**`GYRE_OWNER_SHARED_TRACER_ADVECTION`**.  The handoff must compare NEMO's
already-rounded horizontal and vertical centered flux products and its
`r1_e1e2t/e3t(Kmm)` accumulator statements against legoESM `:1579-1599`, one
statement at a time.  The exact V2 record path/hash and its `after_advection_T`
and `after_advection_S` frames above are the target.  This is not assigned to
the ORCA2 north fold: all cells whose stencil reaches the absent fold or MPI
neighbor are excluded.

The binding tracer plant starts from an exact copy of both oracle frames,
changes one T value by one ULP, traverses the same scorer, exits 1, and reports
exactly 1 / 228,641 while S remains exact.

## 7. Coverage at this stop

| boundary | disposition |
|---|---|
| ORCA2 vector/C2 card selector | VERIFIED + binding alternative-selector plant |
| O1 CORE input mapping | previously VERIFIED 0 / 13,320 per field |
| NCAR bulk | `ORACLE_SUPPLIED`; `LANE3B_OWNER` handoff retained |
| SI3/ice exchanges | `ORACLE_SUPPLIED`; producer `UNMEASURED_PENDING_ICE_MERGE` |
| RGB, EOS80, SCO HPG | previously VERIFIED exact against V2 |
| frozen EEN external coefficients | CONFIRMED DEBT; `GYRE_OWNER_SHARED_EXTERNAL_MODE` |
| external endpoint | `ORACLE_SUPPLIED_EXTERNAL_MODE` |
| tripolar U-face native layout | VERIFIED; `ORCA2_OWNER_TRIPOLAR_U_FACE_LAYOUT` |
| production stage `zFu/zFv` | CONFIRMED DEBT; `GYRE_OWNER_SHARED_STAGE_TRANSPORT_SOURCE_ASSOCIATION` |
| divhor + runoff + literal WZV source program on NEMO operands | VERIFIED exact |
| factor-three stage clock | RETRACTED under Rule 11 |
| WZV full-endpoint association | CONFIRMED DEBT; `GYRE_OWNER_SHARED_WZV_STAGE_SSH_ASSOCIATION` |
| stage-1 centered tracer precursor | CONFIRMED DEBT; `GYRE_OWNER_SHARED_TRACER_ADVECTION` |
| stage-3 two-step FCT internals | NOT ENTERED; stopped at earlier shared precursor |
| tripolar FCT north fold | NOT ENTERED; support cells excluded |
| census-round BBL | NOT ENTERED; stopped earlier |
| TKE/EVD/shared ZDF | UNMEASURED; public EOS80 `bn2` helper is not constructible |
| IWM/geothermal | NOT ENTERED; stopped earlier |

The diagnostic also registers, without repairing: EOS80 `bn2`, tripolar EEN
`nemo_avg4`, and tripolar `nemo_div_curl` viscosity as upstream shared/card
constructibility gaps.  No claim crosses those proxy calculations.

## 8. ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| prove production W through compiled card | ASKED | measured; old clock owner retracted |
| preserve and relabel the invalid ablation | ASKED by Rule 11 | gate and Phase-2k receipt corrected in place |
| name partial-mesh test and scope delta | ASKED | exact pytest node and missing fields recorded |
| GYRE/LOCK/OVERFLOW runoff non-regression | ASKED | GYRE + LOCK verified; OVERFLOW explicitly unmeasured |
| correct GYRE W wording | ASKED | `UNMEASURED`, not insensitive |
| confirm vector/C2 card | ASKED | exact selector + binding plant |
| continue with oracle-supplied stage transports | ASKED | stopped at shared tracer precursor debt |
| change shared stage transport, W association, or tracer arithmetic | UNASKED and forbidden in Lane 4 | handed to GYRE |
| infer stage-3 FCT, BBL, TKE/EVD, or IWM across the debt | UNASKED | no inherited claim |
| execute another NEMO MPI run | UNASKED and unnecessary | V2 records suffice |
| enter SI3 operators | UNASKED pending merge | exchanges remain oracle-supplied |
| delete crash or superseded evidence | UNASKED and forbidden | preserved and flagged |

## 9. Next action

The GYRE lane owns three ordered shared repairs: stage-transport source
association, WZV full-endpoint/source association, then the stage-1 centered
tracer accumulator.  Each must use the V2 records named above, production JIT,
one-variable arms, and binding plants, followed by Rule-12 cross-card gates.
Only after the tracer precursor is 0 / 228,641 may Lane 4 enter the stage-3
two-step FCT internals and its tripolar fold, then census-round BBL and
TKE/EVD/IWM entry.  The pre-existing ORCA2-specific runoff implementation is
unchanged by this review round.

This is a Phase-2 stopping receipt, not a Phase-2 closure, not an SI3
certification, and not authorization for legoESM numerics outside the shared
NEMO identity.
