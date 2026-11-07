# NEMO testcase Lane 4 — ORCA2 Phase-2o receipt

Date: 2026-09-06

Starting parent: `b6a6189c9357`

Status: **BBL CERTIFIED; ZDF ENTRY FAIL-CLOSED AS UNMEASURED.**  The
icebergs-off BBL twins are admitted as a reproducible `VARIANT_ORACLE_V2`
extension.  The selected ORCA2 diffusive bottom-boundary-layer arm is exact in
the one shared implementation.  The ordered walk then reaches `zdf_sh2`, where
the retained post-closure record cannot identify the first internal statement;
the exact WRITE-only frames needed for a future acquisition are registered.
No MPI/NEMO command was run in the sandbox, no shipped NEMO file was modified,
no artifact was deleted, and nothing was pushed.

All numerical scores use production JIT on CPU, binary64, the explicit
scalar-libm policy, oracle-relative cellwise comparison, and rank-zero owned
cells.  Phase-2o was preregistered in
`nemo_testcases_l4_orca2_phase2o_preregister.md` before payload scoring or a
numerical change.

## 1. BBL twin admission and V2 pin

The user shell ran the unchanged launchers in:

- `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2n_bbl_a_10step_np2`
- `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2n_bbl_b_10step_np2`

Both runs report `MPIRUN_RC=0`, `RUN DONE`, and `time.step=10`.  Their external
launcher logs are retained as `/tmp/orca2_p2n_run_a.log` and
`/tmp/orca2_p2n_run_b.log`.

The admission gate derives every count from the inventories and schemas:

| comparison | result | disposition |
|---|---:|---|
| twin A versus twin B oracle streams | **98 / 98 raw-byte exact** | PASS |
| inherited streams versus Phase-2m twin A | **97 / 97 raw-byte exact** | PASS |
| ocean restart shards | **2 / 2 exact** | PASS |
| SI3 restart shards | **2 / 2 exact** | PASS |
| history data-variable payloads | **8 / 8 exact** | PASS; only the registered global `TimeStamp` metadata differs |

Unlike the preceding realization, even the pre-consumer `zFw` slot in
`oracle_transport_kt00000001_s1.bin` happened to be raw identical.  This does
not revoke its `UNINFORMATIVE_PRE_CONSUMER_WORKSPACE` classification; no claim
is made about source-undefined bytes beyond this observed 97/97 identity.

The new stream is
`oracle_bbl_diffusive_kt00000001.bin`, SHA-256
`ec221a1947be20ab8c5025e842ed5193d04b5d06da16f52e302dd415bb61362e`.
Its magic is `NEMO_L4_BBLDF_1`; its 13 integers are
`[1,1,3,1,2,3,1,0,94,152,31,64,2686144]`; and its exact size is 21,489,220
bytes.  The payload count is derived as
`6*(jpi*jpj*jpk) + 2*(jpi*jpj)`, covering T/S at Kbb, T/S Krhs before and
after the call, and `ahu_bbl`/`ahv_bbl`.  All source-owned values are finite,
all canonical halo/land/inactive slots are zero, and parsing reaches exact EOF.

Twin A is the pinned `VARIANT_ORACLE_V2` extension and twin B is its independent
witness.  Each external pin manifest has 103 dynamically enumerated entries:
98 oracle streams, four restart shards, and `ocean.output`.  The two manifests
are byte-identical, SHA-256
`8efaa8c42fd6ca2fc44ba85ebd6ba697a0a24f175c378c8f2a2928efad55ed32`.
No multi-megabyte record is committed to Git.

The admission plants all traverse their real validators and exit nonzero:
derived-header count, source-owned twin payload, canonical zero, restart
identity, and history raw payload.  The gate result is PASS, SHA-256
`6d6330b4b111318e393f6f0fc4e05de4b7dd290052922a59ec7f58d0dbb070c9`.

## 2. Executed diffusive-BBL arm

The resolved ORCA2 deck selects `ln_trabbl=.true.`, `nn_bbl_ldf=1`,
`nn_bbl_adv=0`, and `rn_ahtbbl=1000 m2/s`
(`cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg:268-271`).  NEMO dispatches the
diffusive call at `trabbl.F90:118-127`; the advective call at `:129-138` is
inactive.  At RK stage 3, `stprk3_stg.F90:583-598` orders `tra_qsr`, `tra_ldf`,
bottom heat, `tra_bbl`, and finally implicit `tra_zdf`.

The repository search found the existing shared
`packages/ocean/legoesm/ocean/physics/bbl_adv.py`; the diffusive arm extends
that module and is selected by the NEMO option name.  There is no ORCA2 fork.
The implementation follows the executed source in order:

| row | NEMO statement | result |
|---|---|---:|
| static slope geometry and face thickness | `trabbl.F90:507-537` | source-literal shared preparation |
| bottom T/S and Kmm depth gather | `trabbl.F90:342-353` | exact operands |
| EOS-80 alpha/beta criterion and gated coefficients | `trabbl.F90:356-378` | `ahu_bbl`: **0 / 8,489**; `ahv_bbl`: **0 / 8,554** |
| bottom Kbb tracer gather | `trabbl.F90:187-190` | exact |
| four face terms, two grouped divergences, area/thickness scaling, Krhs addition | `trabbl.F90:192-200`; `domhgr.F90:144` for `r1_e1e2t` | T: **0 / 231,519**; S: **0 / 231,519** |

The rank-zero decomposition makes three exclusions explicit rather than
silently shrinking coverage: `ahu` rank-zero `i=89` requires a Kmm neighbor
owned by rank 1; `ahv` rank-zero `j=147` requires a fold-neighbor Kmm value not
in this record; and RHS rank-zero `i=0` requires a west `ahu` face owned by rank
1.  The already-scored oracle `ahu_bbl`/`ahv_bbl` are then supplied to isolate
the full covered RHS statement.

### Rule 11 association census

This boundary exposed scalar division association rather than different
physics.  The frozen census is retained in the gate:

| arm | T unequal / n | S unequal / n | disposition |
|---|---:|---:|---|
| flat/unbarred XLA divide | 48 / 231,519 | 1 / 231,519 | DEBT |
| factored `zbtr` | 75 / 231,519 | 0 / 231,519 | DEBT |
| single-statement barrier | 67 / 231,519 | 3 / 231,519 | DEBT |
| nearest-residual scalar-gfortran divide | **0 / 231,519** | **0 / 231,519** | AT-BAR |

The production helper reconstructs the scalar division residual, considers
the XLA quotient and its adjacent binary64 values, and selects the nearest
source-scalar result; its custom JVP retains the analytic quotient tangent.
This is the campaign's census-round compiler identity, not a numerically
different BBL scheme.  Every NEMO source-level consumer remains separated by
`nemo_source_round`.

Coefficient, pre-tracer, and post-RHS one-variable plants each traverse the
production scorer and exit 1.  The complete numerical result is PASS,
SHA-256
`d76901e82ae14a8f6d7d66ab647d81a9819335f0d67354ba2b6810c9efaf9626`.

## 3. Rule 12 cross-card controls

These are direct production-branch BBL boundary controls, not claims that an
entire cross-card time step was rerun.  Each card is built through its normal
recipe; the gate JITs the selected shared dispatcher and compares every T/S
cell before and after the boundary.

| card | `nn_bbl_adv` | `nn_bbl_ldf` | T movement | S movement |
|---|---:|---:|---:|---:|
| GYRE-zco | 0 | 0 | **0 / 21,120** | **0 / 21,120** |
| LOCK_EXCHANGE-zco | 0 | 0 | **0 / 7,800** | **0 / 7,800** |
| OVERFLOW-zps | 2 | 0 | **0 / 60,600** | **0 / 60,600** |

OVERFLOW is not “no BBL”: its distinct certified advective option 2 remains
selected and unchanged.  Only the new diffusive branch is skipped.  Thus the
diffusive change has 3 / 3 direct cross-card support without presenting these
operator controls as whole-step integrations.

## 4. TKE/EVD/DDM/IWM entry audit

The vertical closure is computed once at whole-step entry, before the RK
stages: `stprk3.F90:150-165` calls
`zdf_phy(kstp,Nbb,Nbb,Nrhs)`.  This must not be conflated with the later stage-3
implicit applications: `stprk3_stg.F90:428-430` calls momentum `dyn_zdf`, and
`:596-598` calls tracer `tra_zdf`.

The resolved execution order in `zdfphy.F90:264-344` is:
`zdf_sh2`, bottom/top drag, mixed-layer depth, TKE closure, copy `avm_k/avt_k`,
river-mouth `avt` increment, EVD, DDM, IWM, turbocline depth, and the `avm`
lateral boundary condition.  Runtime `ocean.output` re-verifies 15 selectors:
TKE, EVD with `nn_evdm=0` and `rn_evd=100`, DDM, IWM, `nn_pdl=1`, `nn_mxl=3`,
`ln_mxl0`, Langmuir circulation, `nn_etau=1`, `nn_htau=1`, `nn_eice=1`, fixed
IWM efficiency, and differential T/S IWM.

The existing `oracle_zdf_entry_kt00000001.bin`, SHA-256
`da27af4d951fa8b5ad9416d8956be1ca9b4f26afa60036201d484e09a210eb8c`,
is schema-valid and finite.  Its 13,453,548 bytes contain full-domain `avm`
and rank-zero-interior `avt`, `avs`, and `en` after the *entire* `zdf_phy`
call.  It contains neither `sh2` nor the pre-closure `avm_k` operand.
Consequently the first numerical boundary is:

| boundary | owner | disposition |
|---|---|---|
| `zdf_sh2`, `zdfphy.F90:264-270`; face-native now-times-before arithmetic at `zdfsh2.F90:80-100` | `GYRE_OWNER_SHARED_TKE` | **UNMEASURED_NEEDS_WRITE_ONLY_FRAMES** |

No downstream post-closure score is inferred.  The current ORCA2 card audit
also registers, without repairing shared code: squared-centered rather than
face-native shear production; no TKE bottom boundary; no `nn_etau=1` path; no
`nn_eice=1` attenuation; EVD momentum enabled despite NEMO `nn_evdm=0`; and
DDM disabled.  On the 1,779 record cells with ice, the card's absent ice
attenuation differs at 1,779 / 1,779 from NEMO's `tanh(10*fr_i)` statement.
Those are `GYRE_OWNER_SHARED_TKE/ZDF` debts.

Lane-4-owned IWM remains unmeasured: NEMO enables IWM and differential T/S
mixing, while the card does not.  NEMO's IWM initialization rebases the
viscosity/diffusivity floors to `1.4e-6` and `1e-10 m2/s`, while the current
card retains `1.2e-4` and `1.2e-5`.  These are registered card-selection/input
debts, not silently omitted physics.

### Required canonical WRITE-only frames

The audit emits the complete machine-readable inventory.  The minimum sites
are:

1. After `zdf_sh2`, before `zdf_tke`: `sh2`, pre-closure `avm_k/avt_k/en`,
   `rn2/rn2b`, U/V at Kbb and Kmm, `e3uw/e3vw` at both levels, face masks,
   `taum/fr_i`, bottom drag/depth/scale fields, and `mbkt`.
2. After `zdf_tke`, before EVD: post-TKE `en`, `avm_k`, `avt_k`, and `p_pdlr`.
3. Before/after EVD and after DDM: `rn2/rn2b`, `avm`, `avt`, and `avs`.
4. At `zdf_iwm`: the six input maps, `ht/gdept/e3w/rn2`, `zfact1..4`, `zReb`,
   `zemx_iwm`, `zav_wave`, `zav_ratio`, and pre/post `avm/avt/avs`.

All must use the Lane-4 header conventions, `lwp` rank-zero guard, derived
counts, zero-first canonical temporaries, and twin reproducibility admission.
No new instrument or run directory was built this phase, as preregistered.

## 5. Coverage and time-level register

| state/boundary | time level | disposition |
|---|---|---|
| BBL source T/S | `Kbb=1` | VERIFIED input |
| BBL coefficient depth/velocity | `Kmm=2` | VERIFIED input on covered cells |
| pre/post BBL T/S tendency | `Krhs=3`, stage 3 | VERIFIED, post fields AT-BAR |
| `ahu_bbl`, `ahv_bbl` | stage-3 coefficient call | VERIFIED AT-BAR on complete rank-zero stencils |
| GYRE/LOCK/OVERFLOW BBL dispatcher | card-native selectors | VERIFIED 0-ULP branch controls |
| ZDF closure input | `kt=1`, `Kbb=1`, `Kmm=1`, `Krhs=3` | selector order VERIFIED; arithmetic UNMEASURED at `zdf_sh2` |
| post-complete-ZDF `avm/avt/avs/en` | `kt=1`, after `zdf_phy` | schema VERIFIED; insufficient for attribution |
| SI3 exchange | inherited oracle levels | ORACLE_SUPPLIED; `UNMEASURED_PENDING_ICE_MERGE` |

## 6. Verification commands and controls

- Phase-2o admission gate: PASS; all five binding plants nonzero.
- Diffusive-BBL numerical gate: PASS; three one-variable numerical plants exit
  1.
- Focused unit tests for BBL, config footguns, and testcase recipes: **69
  passed**.
- Focused Python syntax/static-name check (`ruff --select=E9,F63,F7,F82`):
  PASS.  No claim is made that the repository-wide style backlog is clean.
- ZDF entry audit: exits 0 with status
  `UNMEASURED_NEEDS_WRITE_ONLY_FRAMES`; this is a successful fail-closed audit,
  not a numerical certification.

## 7. ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| `bbl_diffusive_option=0` / `bbl_aht_m2_s=0.0` | ASKED | current default `0` / `0.0` -> proposed `0` / `0.0` (NEMO reference namelist `ln_trabbl=.false.`); per-card values ORCA2 `1` / `1000`, OVERFLOW `0` / `1000` (`nn_bbl_adv=2`), GYRE and LOCK `0` / `0`; decision pending with the user |
| admit and pin BBL twins | ASKED | PASS; A pinned, B witness |
| state raw inherited identity | ASKED | 97 / 97 observed and recorded |
| implement selected diffusive BBL | ASKED, ORCA2 owner | one shared selectable implementation; exact |
| execute ORCA2 advective BBL | UNASKED / inactive | `nn_bbl_adv=0`; not walked |
| preserve OVERFLOW option 2 | ASKED by Rule 12 | exact no-movement diffusive-branch control |
| audit TKE/EVD/IWM entry | ASKED | fail-closed at missing `zdf_sh2` operands |
| repair shared TKE/ZDF | forbidden by ownership | registered for GYRE; not changed |
| repair ORCA2 IWM without operands | UNASKED | registered, not guessed |
| build a speculative NEMO instrument | excluded by preregistration | exact future frames listed; none built |
| sandbox MPI/NEMO | forbidden | none |
| shipped edit, deletion, push | forbidden | none |

### Explicit claim-disposition register (Phase-2q review repair)

| receipt claim | label |
|---|---|
| BBL twin admission, hashes, 97/97 inherited identity, schema, and plants | **CONFIRMED** |
| resolved ORCA2/GYRE/LOCK/OVERFLOW BBL selectors and source ordering | **CONFIRMED** |
| diffusive coefficient and RHS scores, source-statement localization, and Rule-12 no-movement rows | **CONFIRMED** |
| current card coverage/time-level rows and first unscored `zdf_sh2` boundary | **CONFIRMED** |
| a future WRITE-only ZDF frame can discriminate SH2/TKE/EVD/IWM | **PLAUSIBLE** until admitted records exist |
| SI3 merge will provide the certified operator | **PLAUSIBLE**; remains `UNMEASURED_PENDING_ICE_MERGE` |

## Stop receipt

Phase 2o closes the ORCA2-owned diffusive BBL boundary at **0 / n** and stops
before `zdf_sh2`.  The next meaningful action is a separately preregistered,
canonical WRITE-only ZDF discriminator build and twin user-shell run.  No run
directory is handed off in this receipt, and no further legoESM ZDF/IWM
numerics were attempted.
