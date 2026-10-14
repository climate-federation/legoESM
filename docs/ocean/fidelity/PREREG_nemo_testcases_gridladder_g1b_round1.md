# PREREGISTRATION — grid ladder rung G1b, round 1 (gridladder_r22)

Frozen before any G1b record exists. Rung G1b (user decision 2026-10-14) =
G1's certified deck plus ONE switch: ORCA2's vector-form momentum advection.
Purpose: certify the np_CRV path (EEN with relative vorticity, the
kinetic-energy gradient, the vertical momentum advection) on ORCA2's grid.

Builds cited: `R4REF` = `ORCA2G1_OMIP_L1_R4REF` (the reference binary this
acquisition runs); `ORCA2` = `ORCA2_OMIP_L4` (D122 pointer).

## 1. The deck switch, as ORCA2's deck resolves it

| line | G1 | G1b | source |
|---|---|---|---|
| ln_dynadv_OFF | .true. | .false. | ORCA2 does not state it; namelist_ref:1054 .false. |
| ln_dynadv_vec | .false. | .true. | ORCA2_OMIP_L4/EXP00/namelist_cfg:346 |
| nn_dynkeg | not stated: namelist_ref:1056 = 0, read and range-checked (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynadv.f90:191`), inert under linear dynamics | 0 | ORCA2_OMIP_L4/EXP00/namelist_cfg:347 |

No other deck line changes (the acquisition script refuses any other
difference). The neighbouring choices NEMO reads on this path are already
ORCA2's resolved values in G1's deck: ln_dynvor_een = T (namelist_cfg:352),
ln_dynvor_msk = F and nn_e3f_typ = 0 (namelist_ref:1069,1072, ORCA2 leaves
them), ln_zad_Aimp = F (namelist_ref:1177, ORCA2 leaves it). Nothing is left
for the deck to settle: **no DECISION_NEEDED**.

## 2. What NEMO derives from the switch (not deck lines)

- `n_dynadv = np_VEC_c2` (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynadv.f90:186`).
- Total vorticity = Coriolis + relative: `ntot = np_CRV`
  (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynvor.f90:864`); the EEN statement
  (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynvor.f90:744-748`), unmasked
  (ln_dynvor_msk = F).
- KE gradient, C2 (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynkeg.f90:117-125`);
  vertical advection (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynadv.f90:137-138`).
  With one wet level dyn_zad's loop is empty
  (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynzad.f90:105`, jk = 1..jpk-2) and
  its bottom statement adds a zero
  (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynzad.f90:132-137`).
- Stage-0 right-hand side in stp_2D adds KEG + ZAD after a velocity wzv call
  (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/stp2d.f90:144-163`); stages 2-3 call
  wzv on the velocities (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/stprk3_stg.f90:287-291`)
  and dyn_adv without transports
  (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/stprk3_stg.f90:331-332`).
- **Velocity-form time stepping replaces G1's thickness-weighted form**:
  stage 1-2 update (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/stprk3_stg.f90:366-370`),
  stage-3 update in dyn_zdf (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynzdf.f90:138-141`),
  the external-mode update
  (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynspg_ts.f90:533-545`), its
  velocity (not transport) sum
  (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynspg_ts.f90:627-629`), and no
  end-of-loop depth recomputation
  (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynspg_ts.f90:674`).

## 3. The G1b card (`build_orca2g1b_vec_card`, case `ORCA2G1b-vec`)

G1's builder with one rung table, `_ORCA2G_DYNADV_VEC`. Measured on G1's
admitted domain file: the G1b and G1 model configurations differ in exactly
seven fields (test pins it):

- the four momentum selectors keep the ORCA2 identity's own values:
  vector_invariant, een_total, nemo_advective, and the inert upwind flux
  selector (ke_gradient c2 is the same in both);
- **the two flux-form solver statements G1 certified are not selected on
  G1b** (barotropic_flux_update_nemo_literal, barotropic_entry_depth_nemo_literal):
  NEMO's vector deck takes the velocity-form branch at
  `ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynspg_ts.f90:533` and never runs
  them. The model refuses the combination
  ("needs NEMO's flux-form external update");
- the flux-form stage-1 right-hand-side option (rk3_stage1_rhs_nemo_literal)
  is not selected on G1b: under vector form the model already hands stage 1
  the whole right-hand side and ignores the option (review finding, added
  before any record; no behaviour change).

Searched before building: the G1 builder, its per-rung tables and validator
pair, the ORCA2 identity's vector-form selectors (`_model_config(...,
"orca2_vector_een_c2")`, read off: vector_invariant, c2, nemo_advective,
een_total, zad nemo_literal, stage wzv split), the VORTEX_VEC card's same arm.
New code: one table, one switch-set reader, one wrapper. Nothing else.

Card smoke (dirty tree on 0650720cf, 4 steps, both cards finite): the
vector terms move U by 1.1e-6 m/s at step 1 and ssh by up to 5.9e-5 m by
step 3 relative to G1 (card_smoke.log). The switch is far above the 1e-15
bar, so the record will discriminate.

Model docstring finding (one line): `nemo_flux_form_update_active` says NEMO's
RK3 external mode "exposes no velocity-form arm"; that holds for flux-form
decks only; NEMO's vector branch is at
`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynspg_ts.f90:533`.

## 4. Acquisition (`nemo_testcase_g1b_orca2vec/run.sh`)

No build: G1's pinned round-4 binaries (R4REF reference, P5 record) and G1's
domain file. Smoke 2 steps, then the 100-step reference, then ten record
twins with stride-10 restart lists, so a restart exists at every step 1..N
(NEMO's list holds 10 entries, `ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/in_out_manager.f90:51`).
N = the reference's length: 100, or the largest even step before an stp_ctl
stop.

## 5. Predictions (frozen)

| id | prediction | falsifier |
|---|---|---|
| S-P1 | NEMO completes all 100 steps; no stp_ctl stop | stp_ctl's predicate (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/stpctl.f90:243-250`) stops it (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/stpctl.f90:310`) at any kt <= 100 |
| S-P2 | max abs U and V stay below 0.05 m/s at every step (G1's NEMO record: at most 0.018) | any step above 0.05 m/s |
| V-P1 | G1b's NEMO record differs from G1's at kt = 1 by more than 1e-8 m/s on U (card: 1.1e-6) | bit-identical, or below 1e-8: the switch would be inert at kt 1 |
| F-P1 | the INDEPENDENT label's first non-bit row is at kt = 1 | kt 1 bit-identical on every field |
| F-P2 | the first non-bit statement is the velocity-form external update `ua_e`/`va_e` (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynspg_ts.f90:533-545`), seen in the kt = 1 substep record (ua_new) — PLAUSIBLE: the card runs the model's legacy velocity update, never compared statement by statement with NEMO on any ladder | the kt = 1 substep loop is bit-identical |
| F-P3 | if F-P2 fails: the velocity sum (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynspg_ts.f90:627-629`), then the stage update (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/stprk3_stg.f90:366-370`), then the np_CRV vorticity (`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynvor.f90:744-748`) | the first unequal statement is none of these four |
| F-P4 | the relative vorticity and KE gradient contribute nothing before kt = 1 stage 2 (stp_2D reads the rest state at kt = 1) | a nonzero KEG or relative-vorticity term in kt = 1's stage-0 right-hand side |

Discriminating measurement, in order: kt = 1 substep record (F-P2/F-P3 first
two), then kt = 1 stage frames 1-3 (stage update, then vorticity/KEG at
stage 2), each by offline replay of the literal statement from NEMO's own
operands (rule 5).

## 6. Choices

ASKED (round brief): survey, deck, acquisition with binary reuse, smoke,
frame twins, restarts at every step, card, preregistration.

UNASKED, each offered for revert:
1. The two flux-form solver statements are deselected on G1b (NEMO's branch;
   the model refuses them with vector form).
2. Restarts at every step = ten record twins (13 GB), the reference binary
   only for smoke and reference.
3. The record length rule for an early stop: the largest even step before
   the stop; a stop before kt = 10 is refused.
4. The G1 ladder is not yet pointed at G1b (next round, with the record).
