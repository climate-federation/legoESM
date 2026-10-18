# Preregistration — NEMO-testcases grid ladder G9, round 1 (gridladder_r46)

**Frozen:** 2026-10-17, before any G9 card change and before any G9 record
exists.  Nothing is scored against NEMO in this round.  Every statement is
READ (off the compiled source), MEASURED (with its source named), or
PREDICTED.

Builds cited: `G5` = `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo` (the
record binary G9 reuses; writer-free twin `ORCA2G5_OMIP_L4_R31REF_FIX1`);
Decision-122 pointer `ORCA2` = `ORCA2_OMIP_L4/BLD/ppsrc/nemo`.
`traadv_fct.f90`, `traadv_cen.f90` and `dommsk.f90` are byte-identical in the
record build and ORCA2 (SHA-256 prefixes `30cdc33a`, `0f16d6af`, `4ee445ed`);
`traadv.f90` is identical in the record build and ORCA2 (`028411d4`; the
writer-free twin differs only by writer removal), and the record build's
`stprk3_stg.f90` call lines equal ORCA2's.  So every line below holds in both.

## 1. Survey (search before build)

Searched `namtra_adv`, `traadv`, `tra_adv`, `fct2`, `nonosc`,
`tracer_advection`, `vmask`, `_resolved_stage_vmask_arm`,
`nemo_qco_resolved_mesh_operands`, `rk3_vector_raw_vmask` across
`packages/`, `scripts/`, `tests/` and the receipts.

| item | value | where |
|---|---|---|
| ORCA2 `&namtra_adv` | three lines: `ln_traadv_fct = .true.`, `nn_fct_h = 2`, `nn_fct_v = 2` | `ORCA2_OMIP_L4/EXP00/namelist_cfg:293-295` |
| resolved from `namelist_ref` | `ln_traadv_OFF/cen/mus/ubs/qck = F`; `nn_fct_imp = 1` (inert: `ln_zad_Aimp = F`); `nn_cen_h/v`, `ln_mus_ups`, `nn_ubs_v` unused | `ORCA2_OMIP_L4/EXP00/namelist_ref:898-913` |
| namelist read | `NAMELIST/namtra_adv/` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:586-596` (SUBROUTINE `tra_adv_init` :577) |
| `ln_traadv_fct` -> `nadv = np_FCT`, exactly one option | | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:626-633` |
| RK3: FCT at stage 3 only | `IF (kstg/=3) ll_dofct = .FALSE.` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:491-494` (SUBROUTINE `tra_adv` :463) |
| dispatch: stage 3 `tra_adv_fct(..., nn_fct_h, nn_fct_v, nn_fct_imp)`; stages 1-2 `tra_adv_cen(..., nn_fct_h, nn_fct_v)` | | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:529-535` |
| centred 2nd order (stages 1-2) | horizontal :149-158, vertical :214-226 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_cen.f90:149-158` (SUBROUTINE `tra_adv_cen` :67) |
| FCT (stage 3): upstream two-step | `CALL fct_up1_2stp` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:171` (SUBROUTINE `tra_adv_fct` :77; `fct_up1_2stp` :472) |
| FCT antidiffusive fluxes, `nn_fct_h = 2` / `nn_fct_v = 2` | | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:195-199`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:262-266` |
| antidiffusive-flux fold exchange (`'V'`, -1) | | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:309-311` |
| the limiter | `CALL nonosc(Kaa, pt(Kbb), ...)` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:316` (SUBROUTINE `nonosc` :743-938) |
| final trend, row j reads `ztFv(j) - ztFv(j-1)` | | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:322-329` |
| the advective transport handed to `tra_adv` | `zFv = e1v*e3v*(1+r3v*vmask)*(vv + zvb*vmask)` every stage | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:283` (SUBROUTINE `stp_RK3_stg` :78); call `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:637` |
| `vmask` itself: compact product, then the V-point fold exchange | | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:211`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:232` (SUBROUTINE `dom_msk` :116) |
| G5 deck and its record | `ln_traadv_OFF = .true.` — **no tracer advection** | `nemo_testcase_g5_orca2ice/namelist_cfg:218` |
| G5 card | `tracer_advection = "none"` | `build_orca2g1_flat_card` |
| existing transcription | the ORCA2 identity's `tracer_advection = "fct2"`: CEN2 at stages 1-2, two-step upstream + `nonosc` at stage 3, `nn_fct_imp = 1` (`advection.py`, `ocean_model_latlon_cgrid.py`); the resolved raw-mesh V mask (D114) `nemo_qco_resolved_mesh_operands(...).vmask3`, already routed to G6's vector stage update by `rk3_vector_raw_vmask` | shared package |
| round-258 / 303 transport-mask arm | a monkeypatch probe in the ORCA2 lane's clone only (`_resolved_stage_vmask_arm`); NOT on this branch, no card selector | ORCA2 lane |

So G5 does NOT run ORCA2's tracer advection; G9 is real work.  The tracer
operator itself is reused unchanged.  Next D128 item after G9: isoneutral
tracer diffusion.

**ORCA2 card (report only).**  Its tracer advection is the ORCA2 identity's
`fct2` = `nn_fct_h = 2`, `nn_fct_v = 2`, `nn_fct_imp = 1` (resolved), CEN2 at
RK3 stages 1-2 and FCT + `nonosc` at stage 3: it matches `&namtra_adv`.  Its
stage transport uses the compact V mask (the gap below).  Its unmeasured list
names no tracer-advection item.

## 2. The fold gap (ORCA2 round 303) is exercised by G9

- READ: NEMO's tracer update on T row j reads the north-face transport
  `zFv(j)` (`traadv_fct.f90:322-329`; CEN2 `traadv_cen.f90:149-158`).  On the
  T-pivot row 147 that face is a fold V face, and `zFv` there is multiplied by
  `vmask` (`stprk3_stg.f90:283`), the FOLD-ASSOCIATED mask (`dommsk.f90:232`).
- MEASURED (NEMO's own G8 record,
  `round45/oracle_g8_r45/twin_a/oracle_rkstage1_transport_operands_kt00000001.bin`,
  kt 1 stage 1, 30 wet levels): NEMO's `vmask` on row 147 has 1,319 wet V
  faces, and `zFv` is nonzero on all 1,319 (max |zFv| = 4.0158e5 m3/s, equal
  to row 146's).  Against it, the card's compact face mask
  (`compute_face_masks_3d`) differs on exactly 1,319 cells, ALL on row 147;
  the resolved D114 mask differs on 0 cells.
- With advection OFF (G5-G8) nothing consumed row 147's tracer flux, which is
  why those rungs certified with the compact mask.  G9 is the first rung
  where it is consumed.  **So G9's statements include NEMO's fold-associated
  V mask in the stage transport (G9-selected).**  This is the ORCA2 lane's
  stage-1 3-D V transport statement of round 303 (`stprk3_stg.f90:283`;
  R299W7 :291), on the ladder, at every RK3 stage (the line runs at all three).
- Round 303's partner statement, the end-of-stage top-V fold exchange
  (`stprk3_stg.f90:795-796`), is ALREADY selected on every G card
  (`rk3_fold_exchange_top_v_row = True`).

## 3. Claim and exact target

G9 = the certified G5 deck plus ORCA2's `&namtra_adv` lines and nothing else.
Deck `scripts/validate/ocean_fidelity/testcases/nemo_testcase_g9_orca2tadv/namelist_cfg`;
`namelist_ice_cfg` is G5's, byte-identical.

| key | G5 | G9 (resolved = ORCA2's) |
|---|---|---|
| `namtra_adv.ln_traadv_OFF` | `.true.` | `.false.` (ORCA2: unstated -> ref `.false.`) |
| `namtra_adv.ln_traadv_fct` | `.false.` | `.true.` (ORCA2 `:293`) |
| `namtra_adv.nn_fct_h` | unstated (inert) | `2` (ORCA2 `:294`) |
| `namtra_adv.nn_fct_v` | unstated (inert) | `2` (ORCA2 `:295`) |

## 4. Card change (built after this commit)

- New case `ORCA2G9-tadv` = G5's rung tables; `tracer_advection = "fct2"`
  (the ORCA2 identity's own value) instead of `"none"`.
- One new config field, `rk3_stage_transport_raw_vmask` (default `False`):
  the stage transport's V mask is the resolved D114 mask (`stprk3_stg.f90:283`
  with `dommsk.f90:232`).  Selected by `ORCA2G9-tadv` only.
- The deck reader picks the `&namtra_adv` switch set by `ln_traadv_fct` and
  refuses any stated `nn_fct_imp`, `nn_cen_h`, `nn_cen_v`, `ln_mus_ups`,
  `nn_ubs_v` (ORCA2 states none).
- Fail-closed: `ORCA2G9_UNMEASURED = (nemo_post_si3_surface_records,
  g9_record_not_admitted, tracer_advection_fold_step)`.
- Every other card unchanged (pickled configs, coordinates, grids and initial
  states byte-identical before/after).

## 5. Predictions (frozen; scored when the record exists)

- **P1 (READ+PREDICTED).** NEMO's log prints `ln_traadv_fct = T`,
  `ln_traadv_OFF = F` and `FCT scheme is used. Horizontal order: 2`
  (`traadv.f90:609-613`, `:666-668`).  Falsified if absent.
- **P2 (PREDICTED).** The record is admitted, passivity PASS (writer set
  unchanged from G5).
- **P3 (PREDICTED).** With the G9 card (resolved mask), NEMO's kt-1 stage-1
  `zFv` and tracer-advection trend (`oracle_rktracer_operands_kt00000001_s1.bin`,
  after-`tra_adv` minus before) are bit-identical on row 147.  The one-variable
  control (`rk3_stage_transport_raw_vmask = False`) differs on row 147 T and S
  on the 1,319-cell class and nowhere else.  Falsified if the control does
  not differ on row 147, or if the selected card differs there.
- **P4 (PREDICTED).** G9 is NOT bit-identical over kt 1..10.  Candidates for
  the first non-bit statement, in order: (a) the stage-3 antidiffusive-flux
  fold exchange `lbc_lnk(ztFv, 'V', -1)` (`traadv_fct.f90:309-311`) on row
  147 before `nonosc`; (b) the `nonosc` bounds on the fold rows; (c) the
  stage-transport V mask also reaching the stage V face thickness and the
  continuity W in ways NEMO does not (falsifier for the selector's scope).
  Falsified if every row is bit-identical kt 1..10 under both labels.
- **P5 (PREDICTED).** NEMO's writer-free reference stops at kt 11 like G5
  (`stp_ctl`, 10 completed steps): tracer advection does not damp the
  momentum that stops G5.  Falsified if it completes beyond kt 12.

## 6. Acquisition (operator runs it)

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_g9_orca2tadv/run.sh`,
G8's pattern.  No build: tracer advection is a namelist switch, and
`traadv`, `traadv_cen`, `traadv_fct` are compiled into G5's two round-31
binaries (pinned by SHA-256).  No extra input file.  Record = G5's set
(per-step surface input + per-step restarts from three twins) plus what the
record binary already writes at kt 1: stage-1 transport operands (NEMO's
`vmask`, `zFv`), per-stage transports, and stage 1/2/3 tracer RHS before and
after `tra_adv` — the advection trend on row 147.  Length: the reference runs
up to 100 steps to find NEMO's boundary; records cover `min(completed, 10)`.
A longer record would need NEMO to run past kt 11, which P5 predicts it will
not.
