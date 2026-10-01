# RECEIPT — VORTEX round 6: NEMO's after-SSH slot is carried state, and legoESM now carries it

Date 2026-10-01. Lane tip at the start `ea12107cb`. Preregistration
`PREREG_nemo_testcases_l1_vortex_round6.md`, frozen before any measurement.
Evidence `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round6`.

Status: **BUILT AND MEASURED, NOT SWITCHED.** The carried form exists, every
RK3 card has been scored under both forms, and no card states it. The switch is
the operator's decision and the numbers for it are in section 6.

---

## 1. The statement

Round 5 landed NEMO's first-`wzv` after-SSH slot and said plainly that it was
exact only at the first step, because legoESM's RK3 lane carried nothing from
the previous one. This round reads what NEMO actually carries.

**NEMO writes the next step's after-SSH guess at the END of every step, after
the time-level rotation.**

> `Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs    ! Swap: Nnn unchanged, Nbb <==> Naa`
> `! linear extrapolation of ssh to compute ww at the beginning of the next time-step`
> `ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)`
> — `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:221-225` and
> `GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3.f90:222-226`

The rotation is what makes the two operands concrete, and it is read off the
code rather than assumed. `Nbb` is untouched through all three stages (the two
stage swaps exchange `Nnn` and `Naa` only), so after the final swap `Nbb` holds
the height the step PRODUCED and `Naa` the height it ENTERED with. The
assignment is therefore

`after-SSH slot  <-  2 * (end-of-step height)  -  (step-entry height)`.

**The next step consumes it unchanged:**

> `r3t(ji,jj,Kaa) =  ssh(ji,jj,Kaa) * r1_ht_0(ji,jj)`
> — `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:149`,
> `GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stp2d.f90:152`
>
> `CALL wzv    ( kt, Kbb, Kbb, Kaa , uu(:,:,:,Kbb), vv(:,:,:,Kbb), ww, np_velocity )`
> — `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:153`,
> `GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stp2d.f90:156`
>
> `pww(ji,jj,jk) = pww(ji,jj,jk+1) - ( ... + r1_Dt * e3t * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) ) ) * tmask(ji,jj,jk)`
> — `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90:295-298` and
> `GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/sshwzv.f90:295-298` (the two builds differ
> only in the thickness array, `e3t_1d(jk)` against `e3t_3d(ji,jj,jk)`)

**At `nit000` there is no previous step, and NEMO says what the slot holds —
the step-entry height, not a continuity prediction and not zero:**

> `ssh(:,:,Kaa) = ssh(:,:,Kbb)               ! no ssh variation in ww computation`
> — `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:362-370` (`rst_read_ssh`)

**And the slot is part of NEMO's OWN restart layout** — this is the citation
that makes carrying it NEMO's statement rather than legoESM's invention:

> `IF( PRESENT(Kaa) )   CALL iom_rstput( kt, nitrst, numrow, 'ssha', ssh(:,:,Kaa) )   ! after  fields`
> — `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:184`

## 2. What was built, and what it costs the state a run carries

A new carried slot, `eta_rk3_after`, holding exactly NEMO's `ssha`, written at
the end of every step and read by the next step's first `wzv` call. It is
selected by a NEW value of the explicit card field round 5 added:

| `nemo_first_wzv_after_ssh` | what the first `wzv` reads | carried state |
|---|---|---|
| `rk3_extrapolated` — today, every RK3 card | the step-entry height: exact at the first step only | none |
| `rk3_extrapolated_carried` — NEW, no card states it | the slot the previous step left | `eta_rk3_after` |
| `leapfrog_continuity` — both DINO NEMO recipes | `ssh_nxt`'s continuity prediction | none |

Unset still RAISES. **No card is switched** — section 6 is the decision.

The writer and the reader share ONE predicate, so a card cannot write the slot
without reading it, and a census of which cards execute the change is computed
from that predicate rather than re-derived beside it (operator note AR,
finding 2).

**Restart.** `eta_rk3_after` is a PROGNOSTIC slot and the archive format goes
4 -> 5. A format-4 archive is refused by a message that names the slot.
NEMO's own missing-`ssha` fallback is deliberately NOT copied into the loader:
it is right for NEMO reading an older NEMO's restart, and wrong here, where it
would silently make a resumed run a different trajectory from a continuous one.

**Scan.** A slot that appears on the first iteration changes a `lax.scan`
carry's tree structure, so the scan-carry preparation seeds it first — with
NEMO's own pre-first-step value, which is why seeding moves no number
(measured, section 5).

## 3. Non-vacuity, and one measured limit

| control | result |
|---|---|
| feed the carried arm the step-entry height instead of the slot | reproduces the uncarried arm **BIT-FOR-BIT** in every prognostic field — so the slot is read, and its VALUE is the only difference between the arms |
| a 1e-9 m change to the slot | moves the next step; the same change is EXACTLY inert on the arm that does not read it |
| a ONE-ULP change to the slot | **absorbed, measured, and not a defect**: the slot enters as `ssh/h_0` against a ~4 km column and is then scaled by `e3t/dt`, so its last bit lands far below the vertical velocity's own. Stated because the plant had to be sized above it |
| the two arms at `kt=1` and `kt=2` | identical to the last bit — the first step's slot holds the initial height either way |
| seeding the scan carry | every prognostic field identical to the unseeded eager step |

## 4. The VORTEX ladders, both arms, every step

Bar `1.0e-15`. "card" is the arm every card states today; "carried" is the new
form. The ladder is the card's OWN chained trajectory, so the row at `kt=n` is
the state after `n-1` steps and the first step that can READ a carried slot is
the one from `kt=2` to `kt=3`.

| kt | u card | u carried | u factor | ssh card | ssh carried | ssh factor | T card | T carried | T factor |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `2.2204e-16` | `2.2204e-16` | same | `1.3553e-20` | `1.3553e-20` | same | `0.0000e+00` | `0.0000e+00` | same |
| 2 | `3.3693e-06` | `3.3693e-06` | same | `3.7090e-08` | `3.7090e-08` | same | `3.6255e-09` | `3.6255e-09` | same |
| 3 | `5.9832e-06` | `4.0101e-06` | 1.49x | `6.5057e-06` | `6.5057e-06` | **1x worse** | `3.8455e-08` | `3.2611e-08` | 1.18x |
| 4 | `4.2479e-06` | `2.3727e-06` | 1.79x | `8.2200e-06` | `8.2306e-06` | **1x worse** | `1.1421e-07` | `1.1403e-07` | 1x |
| 5 | `6.2107e-06` | `3.7122e-06` | 1.67x | `8.0021e-06` | `8.0267e-06` | **1x worse** | `1.5950e-07` | `1.6918e-07` | **1.06x worse** |
| 6 | `8.9156e-06` | `4.7431e-06` | 1.88x | `8.0199e-06` | `8.0265e-06` | **1x worse** | `1.5089e-07` | `1.7248e-07` | **1.14x worse** |
| 7 | `1.1073e-05` | `5.4589e-06` | 2.03x | `5.2308e-06` | `5.2328e-06` | **1x worse** | `1.3071e-07` | `1.5821e-07` | **1.21x worse** |
| 8 | `1.2745e-05` | `5.9393e-06` | 2.15x | `6.0089e-06` | `6.0158e-06` | **1x worse** | `1.1384e-07` | `1.4752e-07` | **1.3x worse** |
| 9 | `1.2944e-05` | `5.5930e-06` | 2.31x | `4.5933e-06` | `4.5933e-06` | **1x worse** | `1.4740e-07` | `1.4552e-07` | 1.01x |
| 10 | `1.2543e-05` | `4.8655e-06` | 2.58x | `5.2990e-06` | `5.3566e-06` | **1.01x worse** | `1.9256e-07` | `1.4953e-07` | 1.29x |

Status `DEBT` on both arms, first over bar `kt=2` on both
(`['T', 'u', 'v', 'ssh']`) — **the first-over-bar does not move
earlier, and `kt=1`/`kt=2` are identical to the last bit**, which is the
preregistered P1 and it holds.

What the carry buys on this card is the FLAT `kt=3..10` velocity error round 3
named: `kt=10` `u` goes `1.2543e-05 -> 4.8655e-06`, a factor `2.58`, and every
velocity row from `kt=3` on improves. Salinity is identical at every row.
Temperature and height are MIXED and the rows that worsen are registered in the
table above rather than summarised away: `T` at `kt=5` and `kt=6`, `ssh` at
`kt=4`, `kt=5`, `kt=6` and `kt=10`, each by less than 15 parts in a thousand.

**Flux-ENS card** (`VORTEX-zco`): it selects flux-form advection, never reaches
NEMO's first `wzv` call, and the measurement arm REFUSES it rather than
reporting a silent no-op. Its ladder is unchanged:

| kt | T | S | u | v | ssh |
|---|---|---|---|---|---|
| 1 | `0.0000e+00` | `0.0000e+00` | `2.2204e-16` | `2.2204e-16` | `1.3553e-20` |
| 2 | `1.6428e-09` | `4.0602e-16` | `1.1355e-07` | `1.1355e-07` | `3.7088e-08` |
| 3 | `5.5097e-09` | `6.0904e-16` | `2.5818e-07` | `2.1427e-07` | `6.5028e-06` |
| 4 | `1.1279e-08` | `6.0904e-16` | `1.0726e-06` | `9.8923e-07` | `8.2182e-06` |
| 5 | `1.6984e-08` | `8.1205e-16` | `1.0612e-06` | `1.0414e-06` | `7.9777e-06` |
| 6 | `2.0751e-08` | `8.1205e-16` | `1.2877e-06` | `1.1994e-06` | `8.0238e-06` |
| 7 | `2.1536e-08` | `1.0151e-15` | `1.8336e-06` | `1.6287e-06` | `5.1600e-06` |
| 8 | `2.0278e-08` | `1.0151e-15` | `2.1668e-06` | `1.8979e-06` | `5.9498e-06` |
| 9 | `2.0751e-08` | `1.0151e-15` | `2.3598e-06` | `2.0963e-06` | `4.5366e-06` |
| 10 | `2.1033e-08` | `1.2181e-15` | `2.4640e-06` | `2.1917e-06` | `5.3364e-06` |

Every row is byte-equal to round 5's published ladder (verified row by row, not
by eye: 0 of 50 rows differ).

## 5. GYRE — the certified ladder and the from-rest year

GYRE states `rk3_extrapolated` and is NOT switched, so its own certified
numbers are unchanged; the carried arm below is what the decision in section 6
would buy.

| kt | T card | T carried | T factor | u card | u carried | u factor | ssh card | ssh carried | ssh factor |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `0.0000e+00` | `0.0000e+00` | same | `0.0000e+00` | `0.0000e+00` | same | `0.0000e+00` | `0.0000e+00` | same |
| 2 | `6.0544e-16` | `6.0544e-16` | same | `8.3267e-17` | `8.3267e-17` | same | `0.0000e+00` | `0.0000e+00` | same |
| 3 | `4.6060e-09` | `5.8619e-10` | 7.86x | `1.0481e-06` | `4.4348e-10` | 2.36e+03x | `1.9549e-07` | `6.6957e-11` | 2.92e+03x |
| 4 | `9.2919e-09` | `8.1965e-10` | 11.3x | `9.0072e-07` | `9.3838e-10` | 960x | `2.6969e-07` | `2.0591e-10` | 1.31e+03x |
| 5 | `1.5360e-08` | `9.4335e-10` | 16.3x | `2.3077e-06` | `3.7491e-10` | 6.16e+03x | `1.9714e-07` | `2.1071e-10` | 936x |
| 6 | `7.6849e-09` | `1.5917e-09` | 4.83x | `1.3546e-06` | `9.3730e-10` | 1.45e+03x | `2.3872e-07` | `4.3892e-10` | 544x |
| 7 | `2.5728e-08` | `2.4272e-09` | 10.6x | `1.9264e-06` | `1.5479e-09` | 1.24e+03x | `2.3305e-07` | `5.2883e-10` | 441x |
| 8 | `1.1554e-08` | `3.4213e-09` | 3.38x | `1.0007e-06` | `1.1014e-09` | 909x | `3.7126e-07` | `7.8101e-10` | 475x |
| 9 | `8.0350e-09` | `4.5636e-09` | 1.76x | `1.8917e-06` | `1.7265e-09` | 1.1e+03x | `3.0030e-07` | `9.2183e-10` | 326x |
| 10 | `8.2365e-09` | `5.3208e-09` | 1.55x | `1.2375e-06` | `1.7900e-09` | 691x | `2.8813e-07` | `1.2750e-09` | 226x |

Status `DEBT` on both arms, first over bar `kt=3` on both — **not earlier**, and
`kt=1`/`kt=2` stay at the bar (`kt=2` velocities `8.326673e-17` /
`9.714451e-17`, the digits the certification receipt publishes). **All 50
scored rows move and all 50 move TOWARD NEMO; none worsens.** The velocity rows
improve by about three orders of magnitude — `kt=3` `u`
`1.0481e-06 -> 4.4348e-10` is a factor `2363` — and sea surface height by
`2920` at `kt=3` and `226` at `kt=10`.

### The from-rest year

`nemo_testcase_l2_gyre_year_fromrest.py --member 0 --days 360 --snap-steps 6`,
scored by `nemo_testcase_l2_gyre_year_owners.py --day-gap --days 30,240,360`.
The harness's own run-to-run floor is `~2e-10 K`.

| day | certified (round 5, the card) | the carried arm | direction |
|---|---|---|---|
| 30 | `2.343968708874433e-06` | `2.343251020612126e-06` | toward, `-0.031%` |
| 240 | `6.582552471436041e-05` | `6.581707093530567e-05` | toward, `-0.013%` |
| 360 | `5.408463499669056e-05` | `5.407735418221895e-05` | toward, `-0.013%` |

Sea surface height at day 30 improves by `1.7x`
(`2.26797e-07 -> 1.33064e-07`); day-360 `ssh` moves the other way
(`3.66981e-06 -> 3.70213e-06`) and is registered.

**PREDICTION P4 IS REFUTED, and this is the round's main negative result.**
The preregistration predicted, following operator note BM item 2, that GYRE's
day-30 temperature would come back down because the `+0.7%` round 5 registered
was the missing extrapolation. It does not. Day 30 improves by
`7.2e-10 K` — about three and a half units of the harness's own `~2e-10 K`
run-to-run floor — against the `1.6e-08 K` that the `+0.7%` is worth. Measured
against the PRE-round-5 certified day 30 (`2.3276772e-06 K`), the carried arm is
still `+0.67%` away. **The day-30 regression is NOT owned by the missing
after-SSH extrapolation and the walk for it has to look elsewhere.**

The second, larger finding is the contrast between the two scales. The same
change makes GYRE's first ten steps up to `2 900x` more exact and moves its
year by `0.03%`. That is a quantitative statement of something this campaign
has met before: a step-level exactness gain of three orders of magnitude does
not propagate to day 240. It is reported as a measurement, not explained — no
mechanism is claimed here.

## 6. DECISION NEEDED — switch the RK3 cards to the carried form?

**One line: switch `GYRE-zco`, `ORCA2-zps` and `VORTEX_VEC-zco` from
`rk3_extrapolated` (X, today) to `rk3_extrapolated_carried` (Y)?**

**My pick: Y.** The statement is NEMO's own, cited from both cards' compiled
builds, and NEMO itself treats the slot as carried state (it is in NEMO's
restart file). What it buys, measured:

| what | before | after |
|---|---|---|
| GYRE certified ladder, 50 scored rows | — | **50 move, 50 toward NEMO, 0 worse** |
| GYRE `kt=3` `u` | `1.0481e-06` | `4.4348e-10` (`2363x`) |
| GYRE `kt=10` `u` | `1.2375e-06` | `1.7900e-09` (`691x`) |
| GYRE `kt=3` `ssh` | `1.9549e-07` | `6.6957e-11` (`2920x`) |
| GYRE first over bar | `kt=3` | `kt=3` — not earlier |
| VORTEX-vector `kt=10` `u` | `1.2543e-05` | `4.8655e-06` (`2.58x`) |
| VORTEX-vector `kt=2` | `u 3.3693e-06`, `ssh 3.7090e-08` | identical |
| GYRE day 30 / 240 / 360 | `2.3440e-06` / `6.5826e-05` / `5.4085e-05` | `2.3433e-06` / `6.5817e-05` / `5.4077e-05` — all three toward, all three by `~0.03%` or less |

**What it costs**, stated plainly because it is the half that needs the
decision and not the numbers:

1. **A new field in the state a run carries.** That is on this lane's
   stop-and-ask list by itself.
2. **Every existing restart archive stops loading.** The format goes 4 -> 5
   and a format-4 archive is refused by name. Any spun-up leg that has to be
   resumed must be regenerated, so the decision has a cost outside this
   receipt.
3. **The year barely moves.** Three orders of magnitude at the step level buys
   `0.03%` at day 30 and `0.013%` at days 240 and 360. If the year is the
   target, this is not the lever; if NEMO-exactness is the target, it is a
   large step toward it.
4. **ORCA2 is unmeasured.** It is on the RK3 lane and would execute the change
   at every step. Switching it without re-running its ladder would spend
   numbers nobody has measured.

If the answer is Y, the right shape is: switch GYRE and VORTEX-vector now with
the numbers above registered, and switch ORCA2 only after its ladder is
re-measured on this lane (which is already an open item from round 5).

## 7. Fingerprints: nothing that is not switched moved

Every card keeps its stated form, so nothing in this list should move, and
each row below is a MEASUREMENT of that rather than an argument from the
census.

| gate | result |
|---|---|
| `VORTEX_VEC-zco` ladder, card arm | **0 of 50 rows differ** from round 5's published head ladder |
| `VORTEX-zco` (flux) ladder | **0 of 50 rows differ** from round 5's published head ladder |
| `LOCK_EXCHANGE-zco` kt=1..3 | `AT-BAR`, no row over the bar — unchanged (`kt=3` `u 8.6736e-19`) |
| `OVERFLOW-zps` kt=1..3 | `DEBT`, first over bar `{T,u}` at `kt=2`, `T 7.8160e-15` / `u 7.0692e-12` — its own published digits |
| GYRE certified ladder, card arm | first over bar `kt=3`, `kt=2` velocities `8.326673e-17` / `9.714451e-17` — the certified values |
| the three certified card digests | **unchanged** (`GYRE da52bd90a40f71fd`, `LOCK_EXCHANGE d794c4c5cb3dd880`, `OVERFLOW 2bb9d9be75fd924d`) — this round adds a VALUE to an existing field, not a field, which is the preregistered P8 |
| DINO from-rest month gate | `2.040288957e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS`, exit 0, private work directory. **Identical to rounds 3, 4 and 5** — `1.9e-10 K` from the certified value, which is that harness's own run-to-run floor, so this round is INERT on DINO's from-rest month, as its card census predicts |
| focused battery | <!--BATTERYLINE--> |

**Provenance of the year arm.** It ran at `24cc0674b`, the implementation
commit; three commits landed after it. They are the citation map plus the
receipts' prose, one comment correction, and the scan-carry seeding — and the
year harness steps eagerly through `model.step` and never calls the seeder, so
the third cannot reach it either. The ladders and the tanks above all ran at
the final head on a clean tree.

## 8. Which cards execute the change

The census is computed from the model's OWN predicate
(`nemo_rk3_after_ssh_is_carried`), imported rather than re-derived.

| card / recipe | reaches NEMO's first `wzv` | states | carries the slot |
|---|---|---|---|
| `GYRE-zco` | yes | `rk3_extrapolated` | no |
| `ORCA2-zps` | yes | `rk3_extrapolated` | no |
| `VORTEX_VEC-zco` | yes | `rk3_extrapolated` | no |
| DINO `nemo_dino_kamm`, `nemo_dino_kamm_mlf` | yes | `leapfrog_continuity` | no |
| `VORTEX-zco`, `LOCK_EXCHANGE-zco`, `OVERFLOW-zps`, DINO `legoesm_default`, `nemo_paper` | no | nothing | no |

**Nothing carries it today.** That is the point of the round and it is pinned
by a test that goes red the moment a card is switched — which is when these
numbers have to be re-measured.

**ORCA2 is UNMEASURED this round, as it was in round 5, and its ladder claims
stay blocked.** It is on the RK3 lane and would execute the change at every
step if switched.

## 9. Choices made this round

| choice | ASKED or UNASKED | note |
|---|---|---|
| carry NEMO's `ssha` slot itself rather than the previous step's entry height | not a physics choice | the two are the same number (`2*a` is exact in binary, so the subtraction is the only rounding either way), but `ssha` is the variable NEMO's own restart carries (`restart.f90:184`) and it puts the arithmetic where `stprk3.f90:225` puts it |
| no card is switched to the carried form | ASKED — it IS the decision in section 6 | |
| a pre-format-5 restart archive is REFUSED rather than taking NEMO's own missing-`ssha` fallback | UNASKED, and stated | NEMO's fallback is right for NEMO reading an older NEMO's file; here it would silently make a resume a different trajectory from a continuous run |
| the scan carry is seeded with the step-entry height | not a choice | it is NEMO's own value before any step has run, and seeding is measured to move no number |
| the measurement arm REFUSES a card that never reaches the branch | UNASKED, and stated | an arm that silently does nothing would report "no change" as if it were a measurement |

## 10. One correction to round 5

Round 5's code comment for this branch cited NEMO's SOURCE files
(`stprk3.F90`, `stp2d.F90`) with the line numbers of the PREPROCESSED build,
which are two to eight lines away. The comment now names the source file's own
lines and this receipt cites the compiled builds explicitly. No number moves;
it is a citation defect, and it is the class the campaign has been bitten by
before.

## 11. OPEN

**OPEN ITEM 1 — the decision in section 6.** Until it is answered, every RK3
card still evaluates NEMO's after-SSH slot as the step-entry height at every
step past the first.

**OPEN ITEM 2 — ORCA2 is still unmeasured on this lane** (round 5's open item,
not closed here).

**OPEN ITEM 3 — the vector card's remaining `kt=2` residual.** Round 5 put it
in stages 2-3, not in the completed pre-stage right-hand side, and this round
does not touch it: `kt=2` is identical in both arms by construction.

**OPEN ITEM 4 — which after-SSH form DINO's non-MLF NEMO card should state**,
read from its own build (round 5's open item 3, unchanged).

**OPEN ITEM 5 — the resolution ladder** (decision 74, note BK), still owed.

## 12. Review

<!--REVIEW-->
