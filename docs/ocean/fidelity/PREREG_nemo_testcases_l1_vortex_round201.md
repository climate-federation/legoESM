# Pre-registration — round 201 / VORTEX round 17 (the flux card's stage-1 `dyn_adv` and update statements)

Frozen BEFORE any measurement of this round.  Lane tip `9183cc4c4`.
This is round 200's prediction **P3**, now runnable: the oracle side was acquired
and committed in round 200, and no NEMO run is needed.

## The target
VORTEX-zco (the FLUX-form card, NEMO build `tests/VORTEX_OMIP_L1_P3`) carries at the
certified round-199/200 registry kt=2 velocity
`u = 1.1355340046037554e-07`, `v = 1.1354649764871994e-07` (bar 1e-15).
Round 200 established, one variable at a time and in the production-jitted step:

* every stage re-makes the error (stage-local 4.358845e-08 / 6.181787e-08 /
  1.203856e-07), so no stage boundary owns it;
* the first non-bit boundary in stage 1 is the advective transports
  `zFu/zFv` (`stprk3_stg.f90:276-277`) at ONE unit in the last place — **not** the
  owner, because the stage output is wrong by `5.4e-07` of the stage's own
  increment while those inputs are wrong by `1.4e-16` of theirs;
* the stage-1 output error has no depth-uniform part, so the barotropic
  replacement (`stprk3_stg.f90:409-421`) is not the producer either.

Two statements remain between `:302` and `:409`, and the round-200 record already
carries NEMO's value on both sides of each:

| statement | compiled citation | record group |
|---|---|---|
| flux-form momentum advection added to the stage RHS | `stprk3_stg.f90:316` (`IF( .NOT.ln_dynadv_vec ) CALL dyn_adv( ..., zFu, zFv, zFw )`) | `adv_u` / `adv_v` |
| thickness-weighted explicit velocity update | `stprk3_stg.f90:372-379` (`uu(Kaa) = ((1+r3u(Kbb))*uu(Kbb) + rDt*(1+r3u(Kmm))*uu(Krhs)) / (1+r3u(Kaa)) * umask`) | `update_u` / `update_v` |

(Round 200's receipt cited these as `:315` and `:371-378`; the exact compiled
lines of `VORTEX_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90` are `:316` for the call
and `:372-379` for the update block.  This round uses the exact lines.)

## What the deck resolves (printed, not assumed)
`tests/VORTEX_OMIP_L1_P3/EXP00/namelist_cfg`: `ln_dynadv_vec=.false.`,
`ln_dynadv_up3=.true.`, `nn_dynkeg=0`, `ln_dynvor_een=.true.`,
`ln_dynldf_OFF=.true.`, `rn_shlat=0.`, `ln_vvl_zstar=.true.` (so `lk_linssh` is
false and the update takes the thickness-weighted branch at `:371`),
`ln_bt_fw=.true.`, `nn_bt_flt=3`, `ln_bt_auto=.false.`, `ln_traadv_fct=.true.`.
Anything this round needs that the deck does not pin is reported as
DECISION_NEEDED, not chosen.

## The seam this round builds
legoESM refuses stage 1 for the momentum-operator exposure by construction
(`expose_momentum_operator_stage must be 2 or 3`), and no hook publishes the
stage-1 Krhs or the stage-1 raw Kaa.  This round adds the stage-1 companions of
the two hooks that already exist for stage 2
(`expose_stage2_momentum_rhs`, `expose_stage2_raw_momentum`):

* `expose_stage1_momentum_rhs` — publishes the completed stage-1 Krhs;
* `expose_stage1_raw_momentum` — publishes stage-1 Kaa after the update and
  before the barotropic replacement.

Both default `False`, both are private test hooks, no card constructs either, and
both substitute the returned `u`/`v` slots only AFTER the ordinary step has run.

## Predictions (each with its falsifier)
P1. **Zero behaviour change.** With both hooks present and unset, both VORTEX
    cards' certified 50-row registries are `0/50`, GYRE's certified kt=1..10
    ladder is byte-identical, and the DINO month gate is unchanged at
    `2.053801168e-03` K.  FALSIFIER: any row moves — then the seam is not
    write-only and the round is HELD with the diff reverted.

P2. **The seam is live.** Each new row's exposed slot differs from the plain
    step output in at least one cell (the walk's existing `require_live`
    control), and each new row's plant is VISIBLE as a difference against the
    unplanted run.  FALSIFIER: a `require_live` refusal or a NOT VISIBLE plant —
    then the row is withdrawn, not reported.

P3. **The stage-1 Krhs after `dyn_adv` is NOT bit-identical** to NEMO's `adv_u`.
    PREDICTED first non-bit producer: `stprk3_stg.f90:316`, the flux-form
    advection trend, with a discrepancy of order `1e-11..1e-9` m/s^2 — the size
    that, multiplied by the stage clock `rDt = rn_Dt/3 = 48 s`, makes the
    observed `4.36e-08` m/s stage-1 output error.
    FALSIFIER: `adv_u`/`adv_v` are bit-identical while `update_u`/`update_v` are
    not — then the owner is the thickness-weighted update at `:372-379` and the
    operand to walk next is `r3u(Kbb)/r3u(Kmm)/r3u(Kaa)`.

P4. **The budget closes.** Whichever of the two rows is first non-bit, its
    discrepancy times the stage's own coefficient reproduces the stage-1 output
    error to within a factor of 2: for the RHS row, `rDt * max|d(adv_u)|`
    against `4.358845e-08`; for the update row, `max|d(update_u)|` against the
    same number up to the barotropic replacement's column mean (measured at
    `7.7e-17`, i.e. negligible).
    FALSIFIER: the product is more than 2x away from `4.358845e-08` — then the
    named statement is not the magnitude owner either and the receipt says so
    (round 200's lesson: a first non-bit boundary is not automatically an owner).

P5. **The vector card is inert.** Nothing landed here moves VORTEX_VEC-zco:
    `0/50` rows.  FALSIFIER: any vector row moves.

## Landing gates (unchanged)
Decisions 43/45/55/59; both cards' certified 50-row registries (register every
moved row; the vector card must be `0/50`); the two-ULP cellwise ratchet (red =>
HOLD and report, no landing without a user decision); GYRE byte-identical
(certified kt=1..10 ladder, and the 360-day year if production code changes:
day 30 `2.3432510206121264e-06`, day 240 `6.581707093530567e-05`, day 360
`5.407735418221895e-05` K); the generic NEMO-GYRE recipe gate; the DINO month
gate in `land.sh` (reference `2.053801168e-03` K, bar `2.244317642e-03`); one
fresh code-reviewer subagent on the diff before landing, verdict recorded.

## Plants
The round-200 walk's plant discipline is kept and extended to the new rows: a
plant is a DIFFERENCE in that row's `max_abs` against the unplanted run of the
same walk, never a status read (round 200's vacuous-plant lesson).
