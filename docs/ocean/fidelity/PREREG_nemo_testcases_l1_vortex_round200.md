# Pre-registration — round 200 / VORTEX round 16 (flux card's own kt=2 velocity owner)

Frozen BEFORE any measurement of this round.  Lane tip d7de69d51.

## The target
VORTEX-zco (the FLUX-form card, NEMO build `tests/VORTEX_OMIP_L1_P3`) carries, at the
certified round-199 registry, kt=2 velocity
`u = 1.1355340046037554e-07`, `v = 1.1354649764871994e-07` (bar 1e-15); kt=2 ssh is
`2.66e-15` (at bar since round 198) and kt=2 T/S are at the bar.  Rounds 198 and 199
left both rows unmoved (flux card inert, 0/50).

## What the deck resolves (printed, not assumed)
`tests/VORTEX_OMIP_L1_P3/EXP00/namelist_cfg`: `ln_dynadv_vec=.false.`,
`ln_dynadv_up3=.true.` (flux-form UBS-3), `nn_dynkeg=0`, `ln_dynvor_een=.true.`,
`ln_dynldf_OFF=.true.` (no lateral momentum diffusion), `rn_shlat=0.` (free slip),
`ln_vvl_zstar=.true.`, `ln_bt_fw=.true.`, `nn_bt_flt=3`, `ln_bt_auto=.false.`,
`ln_traadv_fct=.true.`.  Anything this round needs that the deck does not pin is
reported as DECISION_NEEDED, not chosen.

## Predictions (each with its falsifier)
P1. The kt=2 velocity residual is made INSIDE step 2's RK3 stages, not before it:
    substituting NEMO's recorded kt=2 step entry (arm 1) leaves the velocity row at
    ~1.1355e-07.  FALSIFIER: arm 1 drops below 1e-9 — then the owner is the kt=1 step
    and this round walks step 1 instead.
P2. The residual is localised to ONE stage by the existing flux-card record
    (phase3/vortex/round1, same build): one of arms 2/3/4 drops the velocity row by
    >10x while the previous arm does not.  FALSIFIER: all arms stay within 2x of arm 1
    — then no stage boundary owns it and the owner is a per-stage term that re-makes
    the same error every stage, which the stage-local rows (stage1_out/stage2_out/
    stage3_out) must then show.
P3. Given the flux card has NO lateral diffusion and its vorticity scheme is EEN (the
    scheme round 197/198 already fixed at `nn_e3f_typ=0`), the surviving per-term
    candidates inside the owning stage are, in NEMO's own order:
    the flux-form advection trend (`dyn_adv` → `dyn_adv_ubs`), the EEN vorticity trend
    (`dyn_vor_een`), the hydrostatic pressure gradient (`dyn_hpg`), the vertical
    diffusion (`dyn_zdf`), and the surface pressure gradient / barotropic handoff
    (`dyn_spg_ts`).  PREDICTED first non-bit producer: the flux-form advection trend
    (it is the one operator the vector card does not run, and rounds 198/199 moved the
    vector card to the bar while leaving this card exactly unmoved).
    FALSIFIER: the per-term record shows `dyn_adv`'s trend bit-identical to NEMO's at
    both stages while another term is not.
P4. The vector card (VORTEX_VEC-zco) stays inert under anything landed here: 0/50 rows
    move.  FALSIFIER: any vector row moves — then the statement is not flux-form-local
    and the landing is HELD.

## Acquisition, if needed
The flux build has NO per-stage per-term record (only `phase3/vortex/round1`:
step entry kt=1..10, stage s1/s2/s3 at kt=1, pre-stage RHS, barotropic frames).  If the
walk needs per-term operands, the round extends the round-192 instrument
(`stprk3_stage_terms_record.patch` + `vortex_r8_stage_terms.F90`) to the flux build,
runs ONE mpirun, proves it additions-only by a byte-identical kt=10 restart against
`phase3/vortex/round1/VORTEX_OMIP_L1_ZCO_00000010_restart.nc`, and admits it through
`check_records.py` with the plants firing.  No record size or header tuple is predicted
by hand (note BD).

## Landing gates (unchanged)
Decisions 43/45/55/59; both cards' certified 50-row registries (vector expected 0/50);
the two-ULP cellwise ratchet (red ⇒ HOLD, report, do not land); GYRE byte-identical
(ladder, and the year if production code changes); generic NEMO-GYRE recipe gate; DINO
month gate in land.sh (reference 2.053801168e-03 K, bar 2.244317642e-03); one fresh
code-reviewer subagent on the diff before landing.
