# VORTEX round 19 (lane round 205) — `dyn_adv_up3` split, and the divisor that owns it

> **AMENDED by lane round 206 (Decision 86).**  The decision at the end of
> this receipt was taken: the divisor MOVED to NEMO's `hu_0*(1+r3u(Kmm))`
> and landed on the shared RK3 path.  Two names in this receipt are
> therefore stale.  The private arm `momentum_transport_stage1_operand=
> "qco_depth"` no longer exists — the production path now IS that divisor
> and the arm is inverted to `"legacy_min_rule_depth"`; the retired string
> raises.  The held patch
> `manifests/nemo_testcase_l1_vortex_round205_transport_qco_depth_held.patch`
> is retired in place as `..._LANDED.patch`.  Read the round-206 receipt
> (`nemo_testcases_l1_vortex_round206_transport_depth_landing_receipt.md`)
> for what landed, for the GYRE/vector/tank measurements this round did not
> make, and for the review finding that the landed form is the ALGEBRAIC
> and not the LITERAL transcription of `stprk3_stg.f90:270`.

**ROUND_STATUS: HELD** — the instrument and the statement land; the FIX does
not, because it trips the cellwise two-ULP ratchet and that is a user
decision.  No NEMO run was needed.  No production default, scheme, deck
value, bound or carried state changed in what is landed.

## The one-sentence result

**The flux card's whole kt=2 error is made by ONE operand of ONE compiled
statement: `stprk3_stg.f90:270` divides the barotropic transport by
`hu_0*(1+r3u(Kmm))`, the surface-height-ratio column depth, and legoESM
divides it by the sum of its MIN-RULE face thicknesses instead.**  Feeding
NEMO's own depth into that one place — nothing else — drops the stage-1
flux-form advection trend's disagreement from `6.285649e-11` to
`2.032879e-20` in `u` (`6.279942e-11` to `2.668154e-20` in `v`), the
arithmetic floor, and moves **39 of the 50 certified `VORTEX-zco` rows
toward NEMO**, kt=2 `u`/`v` by 9.3x and kt=10 `u`/`v` by 49x.

It is NOT landed.  On the same measurement it worsens individual cells by up
to `2.5133237e+07` row-scale oracle ulp against a 2-ulp bar, so the standing
rule makes it a HOLD.  **The decision is at the end of this receipt.**

## What the deck selects, read from the resolved namelist and the source

`/data/abyssal/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/VORTEX_OMIP_L1_P3/`:

* `EXP00/namelist_cfg:182` `ln_dynadv_vec = .false.` — flux form.
* `EXP00/namelist_cfg:184-185` `ln_dynadv_cen2 = .false.`,
  `ln_dynadv_up3 = .true.` — the third-order upwind-biased scheme, so
  `dyn_adv` is `dyn_adv_up3`.
* `BLD/ppsrc/nemo/dynadv_up3.f90:44-45` `gamma1 = 1/3`, `gamma2 = 1/32`
  are compiled PARAMETERs, not namelist values.
* `EXP00/namelist_cfg:177` `ln_vvl_zstar = .true.`; `ln_zad_Aimp` is not
  overridden, so `cfgs/SHARED/namelist_ref:1177`'s `.false.` stands and the
  vertical advection is explicit.
* `EXP00/namelist_cfg:99` `rn_shlat = 0.` — free slip.
* `BLD/ppsrc/nemo/stprk3_stg.f90:48` `n_baro_upd = np_HYB` is a MODULE
  DEFAULT and is assigned nowhere else in the compiled source (`grep
  n_baro_upd` returns only that line and the four `SELECT CASE`s at
  `:135,165,185,262`), so the barotropic correction NEMO builds is the
  `np_LIN, np_HYB` branch at `:270`:
  `zub = un_adv*(r1_hu_0/(1+r3u(Kmm))) - uu_b(Kmm)`.
  **Nothing here is an unpinned option**: no DECISION_NEEDED on the deck.
* `BLD/ppsrc/nemo/stprk3_stg.f90:316`
  `CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )` — BOTH
  velocity time levels are `Kmm`, so no `Kbb`/`Kmm` split can explain
  anything here.

## The split, measured in the production-jitted step

Three new values of the private WRITE-only hook
`expose_stage1_momentum_rhs_split` (default `""`, closed set, unknown string
raises), each publishing one part of the content round 204 removed:

| arm | what it publishes | NEMO's statement |
|---|---|---|
| `advection_horizontal` | the flux-form `-div(transport (x) velocity)` trend plus the `zub` transport increment | `dynadv_up3.f90:174-215` |
| `advection_vertical` | the vertical UP3 term plus the stage ZAD increment | `dynadv_up3.f90:245-360` |
| `advection_zub_increment` | the barotropic cross-term alone | `stprk3_stg.f90:264-277` fed into both |

The two halves add back to round 204's trend to `1.355e-20` (`u`) /
`1.365e-20` (`v`) — the floor, not zero, because adding and subtracting the
same array is not an identity in floating point.

**Per level, with the stage-entry velocity the record itself carries**
(largest magnitude on the active mask; `H` the horizontal half, `V` the
vertical half, `d` the difference against NEMO's recorded total):

| level | `|u|` entry | `|H|` | `|V|` | cells differing | max `|d|` |
|---:|---:|---:|---:|---:|---:|
| 1 | `8.723e-01` | `1.371e-05` | `3.526e-08` | 956 | `6.2856e-11` |
| 2 | `6.783e-01` | `8.291e-06` | `9.479e-09` | 998 | `4.8880e-11` |
| 3 | `4.844e-01` | `4.228e-06` | `8.156e-09` | 1038 | `3.4904e-11` |
| 4 | `2.904e-01` | `1.520e-06` | `1.923e-08` | 1171 | `2.0927e-11` |
| 5 | `9.648e-02` | `1.681e-07` | `1.615e-08` | 1724 | `6.9511e-12` |
| 6 | `0.000e+00` | `0.000e+00` | `3.923e-09` | 3167 | `6.7763e-21` |
| 7 | `0.000e+00` | `0.000e+00` | `1.124e-09` | 1560 | `5.5057e-21` |
| 8-10 | `0.000e+00` | `0.000e+00` | `0.000e+00` | 0 | `0.0` |

**The vertical block and the shared metric divisor are EXONERATED, by
measurement and not by a bound.**  The recorded stage-entry velocity is
EXACTLY ZERO at levels 6 to 10, and every horizontal flux `dyn_adv_up3`
builds is the transport times the ADVECTED velocity `zui`/`zl`, both of
which are the routine's `puu(:,:,:,Kbb)`/`puu(:,:,:,Kmm)` arguments
(`dynadv_up3.f90:150-157,174-196`) — and `stprk3_stg.f90:316` passes `Kmm`
as BOTH dummies, so both are `uu(:,:,:,Kmm)` here.  The transport `zFu` does
NOT vanish at those levels (it carries the depth-uniform `zub`), but every
one of `zFu_t`, `zFv_t`, `zFu_f`, `zFv_f` multiplies it by a velocity
combination that does.  So
NEMO's horizontal half vanishes there too.  Levels 6 and 7 are therefore the
vertical block RUNNING ALONE, through the SAME
`r1_e1e2u / (e3t_1d(jk)*(1+r3u(Kmm)*umask))` divisor
(`dynadv_up3.f90:335-336`) the horizontal half uses
(`dynadv_up3.f90:206-207`) — and there the two codes agree to `6.8e-21` and
`5.5e-21` against a vertical half of `3.923e-09` and `1.124e-09`, i.e. to
`1.7e-12` and `4.9e-12` RELATIVE.  Scaled onto the vertical half's largest
value anywhere (`3.526e-08`, level 1), that relative agreement bounds the
vertical block's possible contribution at `~6e-20` — nine orders below the
measured `6.286e-11`.

**The owner is linear in the velocity, and only one thing in the horizontal
half is.**  The difference's per-level maxima fall by `0.777, 0.714, 0.600,
0.332`; the entry velocity's fall by `0.778, 0.714, 0.600, 0.332`; the
horizontal half's fall by the SQUARES of those (`0.605, 0.510, 0.360,
0.111`).  So the difference goes as the FIRST power of the velocity while
the horizontal half goes as the second.  The only piece of the horizontal
flux that is first order in the advected velocity is the DEPTH-UNIFORM
barotropic transport correction `zub` multiplying it — the
`advection_zub_increment` row, whose own per-level maxima are `2.020e-08,
1.571e-08, 1.122e-08, 6.726e-09, 2.235e-09`, falling by the same `0.778,
0.714, 0.599, 0.332`.

## Walking that to the statement: two operands, two causal arms

`stprk3_stg.f90:270` has exactly two operands legoESM could build
differently.  Each was swapped for NEMO's, ONE AT A TIME, through the
private `momentum_transport_stage1_operand` hook (default `""`, unknown
value raises), with every other input left as legoESM's:

| arm | what it replaces | trend vs NEMO | moved vs the unswapped run |
|---|---|---:|---:|
| none (production) | — | `6.285649e-11` u / `6.279942e-11` v | — |
| `prognostic_mean` | subtract the external mode's prognostic `uu_b(:,:,Kmm)` instead of a depth mean re-reduced from the three-dimensional velocity | `6.285649e-11` / `6.279942e-11` | `1.355e-20` / `6.776e-21` |
| `qco_depth` | divide `un_adv` by `hu_0*(1+r3u(Kmm))` instead of by the sum of the min-rule face thicknesses | **`2.032879e-20` / `2.668154e-20`** | `6.286e-11` / `6.280e-11` |

**`prognostic_mean` is REFUTED, and its control is what makes that a
result.**  The arm is LIVE — it moves the trend by `1.355e-20`, not by zero
— and the two quantities were measured apart as well: on this card the
re-reduced depth mean and the prognostic `uu_b` differ by `5.551115e-17`
(`2.292088e-16` relative, 1312 of 3660 columns), one unit in the last place.
A hypothesis that reads well in the source and dies on the measurement.

**`qco_depth` closes it.**  The two depths differ by up to `0.129 m` out of
`5000.86 m` — `2.581937e-05` relative, 698 of 3660 `u` columns — because the
min of two stretched column depths is first order wrong in the sea-surface
height difference ACROSS the face, and the sea surface spans `0.000` to
`0.916 m` here.  That is a depth-uniform error in the advective transport
velocity, which multiplies the advected velocity and therefore makes a trend
error LINEAR in it: the measured signature, by construction.

## What the fix would do to the certified registry — measured, not landed

Candidate = the production path divides by NEMO's depth
(`manifests/nemo_testcase_l1_vortex_round205_transport_qco_depth_held.patch`,
97 lines).  Evidence `phase3/round205/candidate_flip_{traj,ulp}_VORTEX-zco.json`.

**39 of 50 rows move TOWARD NEMO, 1 away, 10 unchanged; one row leaves
DEBT.**

| row | before | after | |
|---|---:|---:|---|
| `kt2.before.u` | `1.135534e-07` | `1.216831e-08` | TOWARD, 9.3x |
| `kt2.before.v` | `1.135465e-07` | `1.239432e-08` | TOWARD, 9.2x |
| `kt2.before.T` | `1.642800e-09` | `2.823263e-10` | TOWARD |
| `kt3.before.u` | `2.144706e-07` | `1.042543e-08` | TOWARD |
| `kt6.before.S` | `1.015061e-15` | `8.120488e-16` | TOWARD, **DEBT -> AT-BAR** |
| `kt10.before.u` | `4.888403e-07` | `9.953976e-09` | TOWARD, 49x |
| `kt10.before.v` | `4.859896e-07` | `8.874215e-09` | TOWARD, 55x |
| `kt10.before.ssh` | `1.899224e-07` | `1.226045e-08` | TOWARD |
| `kt5.before.S` | `6.090366e-16` | `8.120488e-16` | **AWAY**, AT-BAR both sides |

`first_over_bar` does not move earlier (`{T,u,v,ssh} kt=2` both sides) and no
AT-BAR row becomes DEBT.  **But the cellwise oracle-relative ratchet is RED**:
`largest_oracle_residual_worsening_ulps 2.5133237e+07` against a 2-ulp bar,
with the worst named cells `kt10.before.u` cell 10 (`1.77e-12`),
`kt10.before.ssh` cell 586 (`9.17e-13`), `kt10.before.v` cell 5 — all on rows
whose own maximum residual improved by 15x to 55x.  The field as a whole
moved `2.2507631e+09` row-scale ulp, i.e. the worsening is 1 per cent of the
move; it is still a worsening, and the rule is mechanical.

## Pre-registration: what held and what did not

`phase3/round205/prereg.md`
(`sha256 e6248c324e69c923a89eddf5a671a94aaf9d48ce4b075aeef717922ad8997c79`),
frozen before any measurement.

* **P1 zero behaviour change of what is LANDED — CONFIRMED.**  Gate table below.
* **P2 the new seams are live — CONFIRMED**, all six plants VISIBLE as a
  DIFFERENCE against the unplanted run (`phase3/round205/split_plants.log`).
* **P3 the split is the same partition round 204 scored — CONFIRMED**, and
  it is an algebraic identity of the construction, so it says nothing about
  how NEMO partitions its own trend:
  `1.355e-20` / `1.365e-20`, under the predicted `1e-20`-scale floor, and the
  baseline rows reproduce round 204's `6.285649e-11` / `6.279942e-11` to
  every printed digit.
* **P4's EXONERATION TEST, AS WRITTEN, IS VACUOUS AND ITS FALSIFIER FIRED.**
  P4 predicted the horizontal half would be NON-ZERO (`> 1e-9`) at levels
  8-10 so that the zero difference there would exonerate it.  It is EXACTLY
  ZERO there, which P4 itself names as the falsifier ("then those levels
  prove nothing and the horizontal is NOT exonerated").  Recorded REFUTED
  rather than rescored.  What the measurement gave instead is the OPPOSITE
  and stronger separation — levels 6-7, where the VERTICAL half runs alone
  and agrees to the floor — and that argument is stated above on its own
  evidence, not smuggled in under P4.  P4's prediction of WHICH part owns the
  error (the vertical block) is therefore also REFUTED: it is the horizontal
  half's transport operand.
* **P5 the metric divisor is not the owner — CONFIRMED, and by a better
  instrument than the one preregistered.**  P5 proposed a bound from round
  200's transport agreement and a ratio test; the ratio test's own criterion
  (interquartile spread below 1 per cent of the median) does not fire —
  measured `1.149` — and the divisor is in any case measured EXACT at levels
  6-7, where it carries the vertical half alone.
* **P6 — the first non-bit PART is named**, but it is `dynadv_up3.f90:174-215`
  (the horizontal half), not the vertical block P4 predicted, and the round
  went one step further than P6 required and named the statement.

## Gates

| gate | result |
|---|---|
| `VORTEX-zco` certified registry vs round 204 | **PASS**, 50 rows, `max_worsening_ulps 0`, **0 of 50 rows move**, no row-status change, `first_over_bar` unchanged `{T,u,v,ssh} kt=2` |
| `VORTEX_VEC-zco` certified registry vs round 204 | **PASS**, 50 rows, `max_worsening_ulps 0`, **inert, 0 of 50**, `first_over_bar` unchanged `{u,v,ssh} kt=2` |
| cellwise two-ULP ratchet plants (flux card) | `worsen-3ulp` exit 1 (`max_worsening_ulps=3`), `at-bar-to-debt` exit 1 — both red on the same 50-row pair the unplanted run passes at 0 ulp |
| GYRE certified kt=1..10 ladder vs round 204 | **PASS**, **954 rows, 0 move, 0 ulp**, `first_over_bar` unchanged `{T,S,u,v,ssh} kt=3` |
| GYRE 360-day from-rest year (note BW) | **byte-identical**, 360 of 360 snapshots equal to the round-203 certified arm, 0 differing, 0 missing |
| citation gate, `DEFAULT_RECEIPT` | **PASS** exit 0, 274 citations, `unmapped_citations []`, `map_entries_failing_audit []`, all self-test plants fired |
| focused battery | **53 passed in 437.04s** (`..._round205_adv_split.py`, `..._round204_stage1_split.py`, `..._round201_stage1_seam.py`, `test_nemo_testcase_receipt_citation_gate.py`) |
| generic NEMO-GYRE recipe gate | run by `land.sh` with the push gate |
| DINO month gate (`land.sh`, reference `2.053801168e-03` K, bar `2.244317642e-03`) | run by `land.sh`; line quoted in the ledger |

```
certified arm files: 360 | compared: 360 | differing: 0 | missing: 0
day030.npz certified 4e36c106403b495e round205 4e36c106403b495e
day240.npz certified a63befc30bf03b44 round205 a63befc30bf03b44
day360.npz certified dcb7bc46c8bc75bd round205 dcb7bc46c8bc75bd
```

The three certified day numbers therefore stand to every printed digit
(note BW): **day 30 `2.3432465132112266e-06`**, **day 240
`6.58170624837412e-05`**, **day 360 `5.407736527246344e-05` K**, carried by
byte-identity of the snapshots they are computed from; the day-gap scorer
was not re-run, and that is said here rather than implied.  The ladder and
the year were measured at `66b42841a`, after the review fixes; the only
later commit touches `tests/` alone.

## Non-vacuity

* **Seam control, inside the walk** (round 200's): every row is read out of a
  state slot the ordinary step also fills, so an inert hook would hand the
  walk the plain output and the row would still score.  The walk runs the
  plain step once and REFUSES unless every exposed slot differs from it.
* **Plants as a DIFFERENCE against the unplanted run**, all six VISIBLE
  (`phase3/round205/split_plants.log`).  SCOPE, because the first draft of
  this receipt overstated it: a plant perturbs the scored ARRAY after it is
  read, so it proves the row's REDUCTION reacts, not that the exposure is
  live.  Seam liveness is `require_live`, which refuses any exposure that
  handed back the ordinary step output, and it runs on every row.
* **The refuted arm's own liveness is measured, not assumed**: `1.355e-20`
  and `6.776e-21` of movement, plus the independent `5.551115e-17` operand
  difference.  A "no effect" from an arm that never fired would have been the
  easy mistake here, and it is the one the control is for.
* **The exoneration rests on a measured zero in the OPERAND, not on a
  guess**: the stage-entry velocity's per-level maxima are published with the
  walk, and levels 6-10 are exactly `0.0`.
* **Citation re-anchor was mechanical**: a difflib old->new line map over
  `ocean_model_latlon_cgrid.py` (15147 -> 15278 lines) applied once to the 18
  spans in the gate's DEFAULT_RECEIPT and once to the 40 spans in its
  CITATION_MAP; never an eyeballed delta.

## Independent review

One fresh `code-reviewer` subagent on the diff, told to refute the claims.
**First verdict: DO NOT SHIP**, six blockers and six minors.  All accepted;
the fixes are their own commit and the receipt is corrected in place:

1. **The receipt and the test module were untracked.**  Both are committed.
2. **Every gate row said PLACEHOLDER** while P1 cited that table.  Filled.
3. **Two of the six plants could not pass**: the walk's `planted` predicate
   omitted the `zub` rows, so `--plant zub.u` exited before any visibility
   test.  Fixed and re-run; all six VISIBLE.
4. **The plants do not test the seams** — see the scope note above.
5. **The transport arm was validated inside the stage branch**, so on a card
   with the transport reconcile off, or a non-WS-RK3 integrator, an unknown
   string was accepted in silence; and `prognostic_mean` fell through to a
   no-op when the state carries no `uu_b` pair, which is exactly what makes
   a refutation vacuous.  Validation moved to model CONSTRUCTION against a
   named closed tuple, and the missing pair now RAISES.
6. **The measured arm and the held patch are different code**: the arm is
   `transport_u_mean * H_u_pre / q` at STAGE 1 ONLY, the patch is
   `Hu_avg / q` at EVERY stage.  Algebraically the same at stage 1
   (`transport_u_mean` is `Hu_avg / H_u_pre * mask`), not bitwise, and not
   the same SCOPE.  So `2.032879e-20` was produced by the arm and the
   39-of-50 registry flip by the patch, and whoever takes the decision
   below should know that.

Minors, all taken: the curvature `zlu_uu`/`zlv_vv`/`zlu_uv`/`zlv_vu` read
`puu(...,Kbb)`, not `Kmm` — the conclusion is unchanged because
`stprk3_stg.f90:316` passes `Kmm` as BOTH dummies, so `Kbb` IS `Kmm` inside
this call, and the receipt now says that rather than citing `:150-157` as
"Kmm"; the exoneration's last step (scaling a level-6/7 relative agreement
onto level 1) is an EXTRAPOLATION and is labelled PLAUSIBLE below; P3 is an
algebraic identity of the construction and says nothing about NEMO's
partition; the citation re-anchor's first pass missed comma-joined ranges
and left an overlapping span, so both files were restored and re-anchored
once with a pattern that maps every range in a comma list; the held patch's
path is written in full; and the qco column depth is now built inside the
arm instead of on every step.

**What the reviewer could not refute:** the partition (legoESM's stage W
comes from the zub-corrected stage transport, matching
`wzv(..., zFu, zFv, ww, np_transport)` at `stprk3_stg.f90:299`, so the ZAD
increment carries zub's vertical share and the transport increment its
horizontal share); the `qco_depth` arm's algebra; the `prognostic_mean`
arm's liveness; and that production outputs do not change.

## Choices made this round

| choice | ASKED / UNASKED |
|---|---|
| Publish the two halves through the existing stage-1 split hook rather than a new one | mechanical; they are the same removed content, split |
| Assign the `zub` transport increment to the horizontal half and the stage ZAD increment to the vertical half | forced; round 204's review established that each reaches exactly one of them |
| Gate the transport-operand probe to stage 1 | forced; `Kmm = Kbb` only at stage 1, so `state.uu_b` is the right time level only there |
| Keep dry columns on the production ratio in the `qco_depth` arm | forced; their qco depth is zero and they carry no transport |
| NOT landing the fix | forced by the two-ULP ratchet; the decision is the user's |

**UNASKED list: empty.**

## THE DECISION THIS ROUND NEEDS

One line, as the rule requires: **the flux card's transport divisor stays
legoESM's min-rule column depth (today, `kt=2 u` `1.135534e-07`), or moves to
NEMO's `hu_0*(1+r3u(Kmm))` (`1.216831e-08`, 39/50 rows toward NEMO, one row
DEBT->AT-BAR, and a cellwise ratchet that goes red at `2.51e+07` ulp on rows
that improve 15x to 55x)?**  My pick: **move it**, registering the ratchet
red as an exception the way Decision 76 registered the day-30 move — the
divisor legoESM uses is not the one the compiled statement uses, so the
cells that worsen are worsening against a residual that was cancelling an
error, not getting further from NEMO's arithmetic.  Landing it also needs the
GYRE ladder and year re-measured with the patch applied, which this round did
not do (the patch was reverted before those gates ran, and that is said here
rather than implied).

## OPEN, in order

1. **The decision above.**  Until it is taken, the patch sits in
   `manifests/nemo_testcase_l1_vortex_round205_transport_qco_depth_held.patch`
   and every flux-card row keeps its current value.
2. If the decision is to move: measure the GYRE certified ladder and the
   360-day year WITH the patch applied (it changes a shared path; GYRE's own
   `momentum_transport_reconcile` arm was not measured this round), and the
   vector card, before landing.
3. The `v` card's 162 cells per level at levels 8, 9 and 10, where BOTH
   legoESM halves are exactly zero and NEMO's trend is not: a separate, much
   smaller thing (`1.3553e-20`, the partition floor) that the split makes
   visible for the first time.
4. The transports' own one-unit-in-the-last-place difference
   (`stprk3_stg.f90:276-277`): split `zub` (`:270`) from the
   `e3t_1d*(1+r3u(Kmm)*umask)` face thickness; neither is recorded.
5. The depth-uniform `8.14e-05` difference between the two codes' stage-1
   right-hand sides (round 204's OPEN 3).
6. The `fmask`-versus-`fe3mask` `r3f` factor, inert only at `rn_shlat=0`.
7. Decision 74's 30/15/10-km ladder.
