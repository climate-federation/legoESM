# VORTEX round 17 (lane round 201) — the flux card's stage-1 producer, named by substitution

**ROUND_STATUS: HELD** — the instrument lands, no physics and no configuration
changes, and no NEMO run was needed.  Round 200's prediction P3 ran from the
record already in hand; it is **REFUTED as stated** and replaced by a measured
result that is sharper than either of its two branches.

## The one-sentence result

The flux card's stage-1 velocity error is made in the **completed stage-1
momentum right-hand side** — the array NEMO holds immediately after
`stprk3_stg.f90:316`.  Substituting NEMO's recorded value of that one array,
and changing nothing else about the arm, takes legoESM's stage-1 output from
`4.358845e-08` to `1.110223e-16` m/s, which is at the bar.

**Said precisely, because the first draft of this sentence was refuted in
review.**  Every arm of this walk — plain, production and carrier alike — is
driven with NEMO's recorded barotropic quintuple
(`stage_barotropic_output_override`), which is round 200's design: the
barotropic solve is a known separate owner and is held fixed so it cannot
contaminate a stage walk.  The carrier arm's ONE extra variable against the
production arm is the stage-1 right-hand side.  So what the arm exonerates is
the **depth-varying channel of the update at `:372-379`**; the barotropic
replacement at `:409-421` is **not** exonerated by it, because its target
operand is NEMO's on both sides of the comparison.

## What this round built

legoESM had no stage-1 momentum seam at all: the momentum-operator exposure
refuses stage 1 by construction, and nothing published either the stage-1
right-hand side or the stage-1 velocity before the barotropic replacement.
Three private test hooks, the stage-1 companions of the stage-2 pair that
already existed, were added to the one shared implementation:

| hook | what it publishes / substitutes | NEMO boundary |
|---|---|---|
| `expose_stage1_momentum_rhs` | the completed stage-1 `Krhs` | after `stprk3_stg.f90:316` |
| `expose_stage1_raw_momentum` | `Kaa` after the update, before the replacement | after `stprk3_stg.f90:372-379` |
| `stage1_momentum_rhs_override` | one-variable substitution of that `Krhs` | the same boundary, as an input |

All three are at defaults no card constructs (`False`, `False`, `None`), all
three substitute the returned velocity slots only AFTER the ordinary step has
completed, and a construction guard refuses any two of them together, or
either with another momentum exposure, because they share those slots.

## The compiled statements, read from the build under test

`/data/abyssal/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/VORTEX_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90`:

* `:316` — `IF( .NOT.ln_dynadv_vec )   CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )`.
  In the flux-form program this is the ONLY momentum statement stage 1 runs;
  the `CASE ( 1 )` arm at `:311-317` contains nothing else.
* `:372-379` — the `ELSE` branch at `:371`, taken because the deck pins
  `ln_vvl_zstar=.true.` so `lk_linssh` is false:
  `uu(Kaa) = ( (1+r3u(Kbb))*uu(Kbb) + rDt*(1+r3u(Kmm))*uu(Krhs) ) / (1+r3u(Kaa)) * umask`.
* `:409-421` — the barotropic replacement, one number per column.

Round 200's receipt cited these as `:315` and `:371-378`; the exact compiled
lines are `:316` and `:372-379`, and this round uses the exact lines.
`stp2d.f90:147-170` is why the stage-entry `Krhs` is NOT a like-for-like row:
under `np_FLX_up3` NEMO sends the advection to the two-dimensional
`Ue_rhs/Ve_rhs` (`pUe=Ue_rhs`, "2D RHS only") and leaves the three-dimensional
array with HPG + LDF + COR/MET only, while legoESM's own pre-stage array is its
completed right-hand side.

## The walk, with the two new statements scored

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_round200_flux_stage1.py`,
production-jitted step, NEMO's recorded kt=1 stage entry, evidence
`phase3/round201/flux_stage1_walk.{json,log}`.  "depth-uniform" is the largest
column mean of the difference; "depth-varying" the largest departure from it.
Only the depth-varying part can reach the stage output, because the
replacement at `:409-421` sets the column mean from the external solve.

| boundary | citation | cells | max abs | depth-uniform | depth-varying |
|---|---|---:|---:|---:|---:|
| pre-stage RHS `u` | `stp2d.f90:126-171` | 18997 | `1.370473e-05` | `2.790517e-06` | `1.091421e-05` |
| `zFu` | `stprk3_stg.f90:276` | 118 | `1.862645e-09` | `8.819825e-10` | `1.190210e-09` |
| `zFv` | `stprk3_stg.f90:277` | 128 | `1.862645e-09` | `8.822099e-10` | `1.166927e-09` |
| `zFw` | `stprk3_stg.f90:302` | 36443 | `2.447726e-08` | `1.309205e-08` | `1.184920e-08` |
| `ww` | `stprk3_stg.f90:298` | 36444 | `2.719696e-17` | `1.454672e-17` | `1.316543e-17` |
| **`adv.u`** | **`stprk3_stg.f90:316`** | 36600 | `8.142171e-05` | `8.142170e-05` | **`4.540459e-11`** |
| **`adv.v`** | **`stprk3_stg.f90:316`** | 36600 | `8.171077e-05` | `8.171076e-05` | **`4.536336e-11`** |
| `update.u` | `stprk3_stg.f90:372-379` | 36600 | `7.816487e-02` | `7.816487e-02` | `4.358845e-08` |
| `update.v` | `stprk3_stg.f90:372-379` | 36600 | `7.844237e-02` | `7.844236e-02` | `4.354887e-08` |
| `out.u` | `stprk3_stg.f90:409-421` | 6253 | `4.358845e-08` | `7.715183e-17` | `4.358845e-08` |
| `out.v` | `stprk3_stg.f90:409-421` | 6338 | `4.354887e-08` | `8.772280e-17` | `4.354887e-08` |

The pre-stage row is reported and carries no attribution, for the reason the
compiled source gives above.

## The budget, and why the two consuming statements are exonerated

**Link 1, the update statement.**  Its coefficient on the right-hand side is
`rDt`, which at stage 1 is `rn_Dt/3 = 2880/3 = 960` s exactly
(`VORTEX_OMIP_L1_P3/EXP00/namelist_cfg:41`).  Measured ratio of the two
depth-varying numbers:

* `u`: `4.3588447087850035e-08 / 4.540458838715646e-11 = 960.00093`
* `v`: `4.3548868552956144e-08 / 4.536336142336485e-11 = 960.00092`

i.e. the update multiplies the right-hand-side error by `rDt`, to `9.7e-07`
relative.  **Two caveats this ratio needs**: it divides two independently
taken maxima and no argmax is kept, so it assumes the same cell on both
sides; and the `9.7e-07` excess is not noise — it is the statement's own
`(1 + r3u(Kmm)) / (1 + r3u(Kaa))` factor, which "multiplies by exactly `rDt`"
drops.  The claim rests on the carrier arm below, not on this ratio.

**Link 2, the barotropic replacement.**  The update row's depth-varying error
`4.3588447087850035e-08` and the output row's `4.358844709092917e-08` agree to
`7.06e-11` relative, and the output's depth-uniform part is `7.7e-17`, at the
bar.  The replacement carries the depth-varying error unchanged and removes
the depth-uniform one, exactly as the statement says it should — on an arm
whose barotropic target is NEMO's, which is why this is a consistency check on
the replacement's arithmetic and not an exoneration of its operand.

**The direct test, one variable.**  Both links above are arithmetic on rows.
The carrier arm is a measurement: substitute NEMO's recorded completed `Krhs`
(`adv_u`/`adv_v`) as the ONLY NEMO-sourced operand and leave every other stage
input legoESM's own.

| arm | stage-1 output `u` | `v` |
|---|---:|---:|
| production | `4.358845e-08` | `4.354887e-08` |
| NEMO's `Krhs` substituted | `1.110223e-16` | `1.110223e-16` |

A factor `3.9e+08`.  **This is AT-BAR, not bit-exact**, and the walk's own
rule is bit equality, so it is stated as what it is: 1436 `u` cells and 1379
`v` cells still differ, every one of them by `1.1102230246251565e-16`, which
is exactly `numpy.spacing(0.8755)` — one unit in the last place of the field's
own peak, the floor of adding and then removing a per-column constant of
`0.078` m/s.  **The producer is upstream of `:372`.**

The override array is put back on the model's internal layout by `_unowned`,
which restores with a zero the one ghost line `_owned` drops.  That ghost
cannot reach a scored cell — the stage update and the barotropic correction
are both column-local, and the scored slice excludes it — and the measurement
is its own witness: a contaminated ghost would show as `~0.078` m/s, not
`1.1e-16`.  A unit control pins the helper's axis and its round trip.

## What P3 predicted, and what is REFUTED

P3 predicted that `adv_u/adv_v` would be non-bit and named the flux-form
advection trend as the first non-bit producer, with the falsifier "the
advection accumulator is bit-identical while the update is not".  Neither
branch happened: BOTH rows are non-bit, and the attribution is settled by the
depth structure instead, which the prediction did not anticipate.  **P3 as
written is REFUTED**; what survives is its direction — the producer is at or
before `:316`, not after it.

**P4 as written is also REFUTED, and the receipt says so rather than quietly
rescoring it.**  P4 named `rDt * max|d(adv_u)|` against `4.358845e-08`; that
product is `960 x 8.142171e-05 = 7.8e-02`, a factor `1.8e+06` out, because
`max|d(adv_u)|` is dominated by the depth-uniform part the replacement
removes.  The quantity that does close is the DEPTH-VARYING part, which the
pre-registration did not name.  P4's intent — the budget closes — holds; its
written falsifier fires.

A second pre-registration error is retracted here: P3 wrote
"`rDt = rn_Dt/3 = 48 s`".  The deck pins `rn_Dt = 2880.`
(`VORTEX_OMIP_L1_P3/EXP00/namelist_cfg:41`) and `stprk3_stg.f90:128` sets
`rDt = r1_3 * rn_Dt`, so the stage clock is **960 s**.  Every number in this
receipt uses 960.

P1 (zero behaviour change), P2 (the seam is live and its plants visible) and
P5 (the vector card inert) are CONFIRMED; the gate lines are in the next
section.

## What this round does NOT name, said plainly

**The statement inside the completed right-hand side is not yet named.**  NEMO
builds that array from two places: `stp_2D`'s HPG + Coriolis/metric at `Kbb`,
which the record carries as `base_u`, and the stage-1 `dyn_adv` call at
`:316`.  legoESM builds it through a different channel — its own pre-stage
array already contains an advection term — so the `base` row cannot separate
the two, and this round does not pretend it can.

What the rows DO bound: legoESM's depth-varying pre-stage difference is
`1.091421e-05`, and after NEMO's advection is added the two completed arrays
agree to `4.540459e-11`.  So the two advection channels agree to `4.2e-06`
relative, and whatever makes the residual is a last-few-digits disagreement
between large, nearly cancelling quantities, not a missing term.  That
sentence is **PLAUSIBLE**, inferred from a convention-sensitive row; it is not
used to name anything.

Also measured and NOT attributed: the two completed right-hand sides differ by
a per-column constant of `8.142170e-05` (`u`) and `8.171076e-05` (`v`).  NEMO
leaves the depth mean in the three-dimensional array (`stp2d.f90:174-186`
copies it, it does not subtract it) and legoESM's stage array does not carry
it.  The carrier arm proves this is immaterial to the trajectory — NEMO's
array, mean included, lands at the bar — but it is a real difference in what
the two codes hold under one name, and it is recorded here rather than left
for a later round to rediscover.

## Gates

| gate | result |
|---|---|
| `VORTEX-zco` certified registry vs round 199 | **PASS**, 50 rows, `max_worsening_ulps 0`, largest field move `0.0` row-scale ULPs — **0 of 50 rows move**, `first_over_bar` unchanged `{T,u,v,ssh} kt=2` |
| `VORTEX_VEC-zco` certified registry vs round 199 | **PASS**, 50 rows, `max_worsening_ulps 0`, largest field move `0.0` — **inert, 0 of 50**, `first_over_bar` unchanged `{u,v,ssh} kt=2` |
| cellwise two-ULP ratchet plants (flux card) | `worsen-3ulp` exit 1 (`max_worsening_ulps=3`), `at-bar-to-debt` exit 1 (`kt1.before.S` crossed) — both red on the same inert pair the unplanted run passes |
| GYRE certified kt=1..10 ladder vs round 199 | **PASS**, **954 rows, 0 move, 0 ULP**, `first_over_bar` unchanged `{T,S,u,v,ssh} kt=3` |
| GYRE 360-day from-rest year | **byte-identical**, 360 of 360 snapshot files equal to the certified carried arm, 0 differing, 0 missing |
| citation gate, `DEFAULT_RECEIPT` | **PASS**, 274 citations, `unmapped_citations []` |
| generic NEMO-GYRE recipe gate + focused battery | see below |
| DINO month gate (`land.sh`, reference `2.053801168e-03` K, bar `2.244317642e-03`) | see below |

```
certified arm files: 360 | compared: 360 | differing: 0 | missing: 0
day030.npz certified 95f336ef7ef9e1e0 round201 95f336ef7ef9e1e0
day240.npz certified 018b75e12bc52968 round201 018b75e12bc52968
day360.npz certified 446a9041887a2c85 round201 446a9041887a2c85
```

The three certified day numbers therefore stand to every printed digit:
**day 30 `2.3432510206121264e-06`**, **day 240 `6.581707093530567e-05`**,
**day 360 `5.407735418221895e-05` K**.  They are carried by byte-identity of
the snapshots they are computed from; the day-gap scorer was not re-run, and
that is said here rather than implied.

Evidence: `phase3/round201/traj_VORTEX{,_VEC}-zco_after.{json,log}`,
`ulp_VORTEX{,_VEC}-zco.json`, `ratchet_plants.txt`,
`ulp_plant_{worsen-3ulp,at-bar-to-debt}.{json,log}`,
`gyre_ladder_after.{json,log}`, `gyre_ladder_compare.json`,
`gyre_year_r201.log`, `gyre_year_byte_identity.txt`,
`citations_default_receipt.json`, `flux_stage1_walk{,_recheck}.{json,log}`,
`flux_stage1_plants.log`.

## Non-vacuity

* **Seam control, inside the walk.**  Every row is read out of a state slot the
  ordinary step also fills, so an inert hook would hand the walk the plain
  output and the row would still score.  The walk runs the plain step once and
  refuses unless every exposed slot differs from it; all four new rows pass it.
* **Plants, as a DIFFERENCE against the unplanted run** (round 200's lesson —
  a row that is already non-bit reports "visible" from a status read whatever
  the plant did).  `adv.u`, `adv.v`, `update.u`, `update.v`: all four
  **VISIBLE** (`phase3/round201/flux_stage1_plants.log`).
* **Known-answer control for the override**, in the committed unit tests:
  feeding the model its own exposed stage-1 right-hand side reproduces the
  production step bit for bit, and bending one operand by `1e-7` relative
  moves it — so the control cannot pass vacuously.
* **Boundary control**: the raw frame and the stage-1 output frame must differ,
  and must differ by one number per column; a seam wired at the old boundary
  fails both.
* **Ratchet plants**: `worsen-3ulp` and `at-bar-to-debt` both exit 1 on the
  same inert pair the unplanted run passes at 0 ULP.
* **The carrier arm carries no plant of its own, and does not need one**: its
  control is the DIFFERENCE between two arms of the same walk.  A substitution
  that bound nothing would return the production number `4.358845e-08`; it
  returns `1.110223e-16`.  That is said here rather than left implied, because
  a plant flag that skips an arm is exactly the vacuity round 200 shipped.
* **The walk was re-run after the layout helper was hoisted to module scope**
  so a unit control could reach it: every row and both carrier numbers are
  identical (`flux_stage1_walk_recheck.json`).

## Choices made this round

| choice | ASKED / UNASKED |
|---|---|
| Build the stage-1 seam as the stage-2 pair's companions rather than relaxing the `expose_momentum_operator_stage` guard | mechanical; the guard exists so a stage cannot be scored under another's name |
| Add the one-variable override as well as the two exposures | required to turn the budget arithmetic into a measurement; note BT's "given NEMO's recorded operands" |
| Report the pre-stage row and exclude it from attribution | forced by `stp2d.f90:147-170` |
| Cite `:316` and `:372-379` instead of round 200's `:315` and `:371-378` | correction; the compiled lines were read |
| Re-anchor 226 citation spans by the difflib map | the standing brief's CITATION RE-ANCHOR RULE |
| Widen the construction guard after review to cover the transport exposures and the override | fail-closed; no card or gate selects any of those combinations |
| Run the 360-day GYRE year although the diff is a private hook | the round brief requires it when production code changes; it was run, not argued away |

## Independent review

One fresh `code-reviewer` subagent on the diff, told to refute the claim.
**First verdict: DO NOT SHIP**, three blockers and five minors, every one of
them in the write-up rather than the measurement.  All accepted:

1. "handed NEMO's recorded right-hand side and nothing else of NEMO's" was
   FALSE — every arm, production included, is driven with NEMO's recorded
   barotropic quintuple.  The headline is rewritten and the barotropic
   replacement's exoneration is withdrawn.
2. The update-to-output agreement was quoted as `7.1e-10`; it is `7.06e-11`.
   Corrected.
3. P4 was reported CONFIRMED against a quantity the pre-registration does not
   name.  P4 as written is now recorded REFUTED, and the pre-registration's
   `rDt = 48 s` is retracted in favour of the compiled `960 s`.
4. The construction guard did not refuse everything the receipt said it did.
   The guard was widened (transport exposures, and the override together with
   an exposure) and the parametrised refusal test extended.
5. The budget ratio is max-over-max and its `9.7e-07` excess is the
   statement's own thickness ratio.  Both now stated.
6. "the gate lines are below" with no gate lines.  They are now in their own
   section.
7. The carrier arm has no plant and `_unowned`'s ghost pad was untested.  A
   unit control for the helper was added, and the arm's own
   difference-against-production control is stated.
8. The exoneration is AT-BAR, not bit-exact.  Now said, with the ULP.

The reviewer independently re-ran both VORTEX ULP gates, the GYRE ladder, the
citation gate and both ratchet plants and reproduced them; it confirmed the
compiled citations, that `lk_asminc` is a `.FALSE.` PARAMETER so the stage-1
arm really does run one momentum statement, that `adv_u` IS like-for-like,
and that eight spot-checked re-anchored spans land on their pinned symbols.

UNASKED list: **empty**.  No default, scheme, bound, threshold, deck value or
carried state was changed; nothing the NEMO deck does not pin was selected.

## OPEN, in order

1. **Name the statement inside the completed stage-1 right-hand side.**  Its
   two sources are `stp_2D`'s HPG + Coriolis/metric at `Kbb` (recorded as
   `base_u`) and the stage-1 `dyn_adv` call at `:316`.  Separating them needs
   ONE of: (a) a legoESM stage-1 seam that publishes the right-hand side
   WITHOUT its advection contribution, which would make `base_u` like-for-like
   and needs no NEMO run; or (b) a per-term `stp_2D` record on the FLUX build
   (the `oracle_rhsterm` family exists for other builds; this build has only
   the total `oracle_rhs`), which does.  (a) is the cheaper first move.
2. The transports' own one-unit-in-the-last-place difference
   (`stprk3_stg.f90:276-277`) as a separate smaller debt: split `zub` (`:270`)
   from the `e3t_1d*(1+r3u(Kmm)*umask)` face thickness; neither is recorded.
3. The depth-uniform `8.14e-05` difference between the two codes' stage-1
   right-hand sides: immaterial to the trajectory (measured), but it is a real
   difference in what the two hold under one name and deserves its own line in
   whatever round touches the barotropic/baroclinic split.
4. The `fmask`-versus-`fe3mask` `r3f` factor, inert only at `rn_shlat=0`.
5. Decision 74's 30/15/10-km ladder, once the flux card's kt=2 rows are at the
   bar or proven harness floor.
