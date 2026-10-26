# PREREGISTRATION — VORTEX round 6: carry NEMO's after-SSH slot across RK3 steps

Frozen before any measurement. Lane tip at the start `ea12107cb`.
Evidence `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round6`.

Round 5 made NEMO's first `wzv` call read the after-SSH slot the RK3 program
leaves behind, and that is exact **only at the first step**: legoESM's RK3 lane
carries nothing from the previous step, so the slot evaluates to the step-entry
height at every step instead of only the first. This round builds the carry,
behind a NEW value of the card field, with no card switched to it, and measures
both arms. Operator note BM items 1 and 2.

---

## 1. NEMO's statement, cited from the compiled source of the cards' own builds

The VORTEX vector-EEN card's build is
`tests/VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo`; GYRE's is
`cfgs/GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo`. The statement is the same in both and
both are cited.

**(a) The slot is written at the END of every step, after the time-level
rotation.**

> `Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs    ! Swap: Nnn unchanged, Nbb <==> Naa`
> `! linear extrapolation of ssh to compute ww at the beginning of the next time-step`
> `! ssh(n+1) = 2*ssh(n) - ssh(n-1)`
> `ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)`
> — `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:221-225`,
> `GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3.f90:222-226`

The rotation is what makes the two operands concrete. Follow the three slots
through one step: entry `Nbb=1, Nnn=2, Naa=3`; `stp_2D` and stage 1 write slot
3; the stage swap (`stprk3.f90:203`, `:209`) exchanges `Nnn` and `Naa` twice and
leaves stage 3 writing slot 3 again, with slot 1 — `Nbb` — untouched for the
whole step. The final swap therefore puts the END-OF-STEP height in `Nbb` and
the STEP-ENTRY height in `Naa`, so the assignment is

`ssh(Naa) <- 2 * ssh_end_of_step(n) - ssh_entry(n)`,

which the NEXT step reads as its after-SSH guess. In legoESM's array
convention that is `2*eta_after_this_step - eta_entering_this_step`, and the
value it leaves behind is a CARRIED field — the state the next step reads.

**(b) The next step consumes it unchanged.**

> `r3t(ji,jj,Kaa) =  ssh(ji,jj,Kaa) * r1_ht_0(ji,jj)` ! "after" ssh/h_0 ratio guess at t-column at Kaa (n+1)
> — `VORTEX_VEC_OMIP_L1_P3/.../stp2d.f90:149`, `GYRE_OMIP_L2_P3_SM/.../stp2d.f90:152`
>
> `CALL wzv    ( kt, Kbb, Kbb, Kaa , uu(:,:,:,Kbb), vv(:,:,:,Kbb), ww, np_velocity )`
> — `VORTEX_VEC_OMIP_L1_P3/.../stp2d.f90:153`, `GYRE_OMIP_L2_P3_SM/.../stp2d.f90:156`
>
> `pww(ji,jj,jk) = pww(ji,jj,jk+1) - ( ze3div(ji,jj,jk) + r1_Dt * e3t_1d(jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) ) ) * tmask(ji,jj,jk)`
> — `sshwzv.f90:295-298`

**(c) At `nit000` there is no previous step, and NEMO says what the slot holds.**
It is NOT a continuity prediction and it is NOT zero: it is the step-entry
height, so the scale-factor term vanishes.

> `id1 = iom_varid( numror, 'ssha', ldstop = .FALSE. )`
> ... `CALL iom_get( numror, jpdom_auto, 'ssha'   , ssh(:,:,Kaa) )`
> ... `ssh(:,:,Kaa) = ssh(:,:,Kbb)               ! no ssh variation in ww computation`
> — `restart.f90:362-370` (`rst_read_ssh`)

and on a from-rest run `ssh` comes from the user-defined initial state through
the same routine, so all three slots hold it.

**(d) The slot is part of NEMO's OWN restart layout.** This is the citation
that makes the carry NEMO's statement rather than legoESM's invention:

> `IF( PRESENT(Kaa) )   CALL iom_rstput( kt, nitrst, numrow, 'ssha', ssh(:,:,Kaa) )   ! after  fields`
> — `restart.f90:184`

NEMO writes the extrapolated after-SSH into its restart file under the name
`ssha`, and reads it back at (c). legoESM's new carried field is the same
quantity under the same contract.

## 2. What is built

A NEW carried state slot, `eta_rk3_after`, holding exactly NEMO's `ssha`, and a
NEW value of the existing explicit card field:

| `nemo_first_wzv_after_ssh` | after-SSH the first `wzv` reads | carried state |
|---|---|---|
| `rk3_extrapolated` (today, every RK3 card) | the step-entry height — exact at the first step only | none |
| `rk3_extrapolated_carried` (NEW, no card states it) | the slot `2*eta_end − eta_entry` the previous step left | `eta_rk3_after` |
| `leapfrog_continuity` (DINO) | `ssh_nxt`'s continuity prediction | none |

Unset still RAISES. **No card is switched this round** — the switch is the
DECISION_NEEDED the operator answers with the numbers below.

Restart: `eta_rk3_after` is a PROGNOSTIC slot, the archive format goes 4 -> 5,
and a format-4 archive loaded by this build is REFUSED with a message that
names the missing slot (the bt_hist precedent). NEMO's own fallback at (c) is
deliberately NOT copied into the loader: silently substituting the step-entry
height would make a resumed run differ from a continuous one with nothing said.

## 3. Predictions, frozen

The ladder gate scores the card's own chained trajectory: the row at `kt=n` is
the state after `n-1` steps. So the step that first READS a carried slot is the
step from `kt=2` to `kt=3`.

| # | prediction | falsifier |
|---|---|---|
| P1 | VORTEX-vector `kt=1` and `kt=2` rows are BIT-IDENTICAL between the two arms (`kt=2` `u 3.3693e-06`, `ssh 3.7090e-08`) — the first step's slot holds the initial height in both | any movement at `kt=1` or `kt=2` |
| P2 | VORTEX-vector `kt=3..10` MOVE, and move TOWARD NEMO (today `u` 5.98e-06 … 1.25e-05, flat) | rows move AWAY from NEMO, or do not move at all |
| P3 | VORTEX-flux card: every row unchanged (it never reaches the branch) | any row moves |
| P4 | GYRE day 30 DECREASES below the certified `2.3440e-06` K — note BM item 2's prediction that the +0.7% round 5 registered is this missing extrapolation | day 30 `>= 2.3440e-06` K ⇒ the +0.7% is NOT owned by the missing extrapolation, and the round says so |
| P5 | GYRE day 360 stays below `1.0e-04` K (it does not regress toward the pre-round-5 `2.671e-03`) | day 360 `> 1.0e-04` K |
| P6 | GYRE day 240 moves by less than a factor of 2 either way from `6.5826e-05` K | a larger move, which is then registered and NOT claimed |
| P7 | With no card switched: GYRE's certified ladder, both tanks, both VORTEX ladders and the DINO month gate are byte-identical to round 5 | any difference ⇒ the new branch is not inert |
| P8 | The three certified card digests do NOT move: this round adds a VALUE to an existing field, not a field | any digest moves |
| P9 | ORCA2-zps is on this lane by configuration but is NOT measured this round; its claims stay blocked (round 5's open item) | — |

The 2e-10 K harness floor is quoted beside every year number.

## 4. Non-vacuity

* a one-ULP perturbation of the carried slot must move the SECOND step's output
  and must not move the first;
* the carried arm and today's arm must differ at `kt=3` (if they do not, the
  carry is not wired);
* the restart round-trip must restore the slot, and a format-4 archive must be
  refused by the NAMED message, not by the generic layout check.

## 5. Choices this round makes

| choice | ASKED or UNASKED |
|---|---|
| carry NEMO's `ssha` slot itself rather than the previous step's entry height | not a choice between physics — the two are the same number (`2a-b` with `2a` exact in binary, so no rounding difference), but `ssha` is the variable NEMO's own restart carries (`restart.f90:184`) and it puts the extrapolation arithmetic where `stprk3.f90:225` puts it |
| no card is switched to the carried form | ASKED — it is the DECISION_NEEDED |
| a format-4 archive is REFUSED rather than taking NEMO's own missing-`ssha` fallback | UNASKED, and stated: NEMO's fallback is for a restart written by an older NEMO; here it would silently turn a resume into a different trajectory |
