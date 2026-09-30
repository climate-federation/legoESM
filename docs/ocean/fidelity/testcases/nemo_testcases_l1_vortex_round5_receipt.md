# RECEIPT — VORTEX round 5: the vector card's second step is owned by the vertical velocity, not by either surviving term

Date 2026-09-30. Lane tip at the start `5add31a068ad`. Preregistration
`PREREG_nemo_testcases_l1_vortex_round5.md`, frozen before any measurement.
Evidence `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round5`.

Status: **HELD for the operator's landing decision on ONE row.** Everything
is measured and committed on the lane; every gate has run. The fix takes the
vector card's second step from `1.4325e-05` to `3.3693e-06` in velocity and
from `2.2685e-05` to `3.7090e-08` in height, improves 40 of GYRE's 50
certified ladder rows with none worsened, and improves GYRE's day-360
temperature by a factor of **49**. It also moves GYRE's **day 30 up by 0.7%**
(`2.327677e-06` to `2.344e-06` K), which is outside every automatic landing
rule this lane has, so the round does not land itself.

---

## 1. Retraction first

**Round 4's exclusion of the vertical advection of momentum was WRONG, and
it is retracted.** It read: "the whole of the term is smaller than the
difference the measured error requires, so even a completely wrong vertical
advection cannot produce it". The bound behind that sentence is one-sided.
The difference being bounded is legoESM's version of a term MINUS NEMO's,
so it is bounded by the SUM of the two sides' peaks, not by NEMO's alone: a
term whose two versions carry opposite signs at a cell produces a difference
larger than either. Here NEMO's term peaks at `1.6928e-08` and legoESM's at
`1.8621e-08`, and their difference is `2.8783e-08` — larger than both, and
the whole of the error.

The same correction is made in the instrument, not only in prose: the
exclusion now uses the two-sided bound. Its own non-vacuity is that it
un-excludes the term round 4 excluded.

Round 4's two named survivors — the relative-vorticity half of the
energy-and-enstrophy triad, and the kinetic-energy gradient — are both
**innocent**, and were already innocent when they were named.

## 2. The per-term table

legoESM's own decomposition at the SAME boundary, from the step's own
tendency call, against NEMO's recorded per-term increments. The map was
fixed in the preregistration before it was run; `keg+hpg` is a group because
legoESM bundles the kinetic-energy gradient with the pressure gradient in
one field, exactly as NEMO's flux-form `adv` bundles the kinetic-energy
gradient with the vertical advection in the other direction.

| row | legoESM peak | NEMO peak | legoESM − NEMO |
|---|---|---|---|
| vorticity (the whole EEN triad) | `6.626650e-05` | `6.626650e-05` | `2.03e-20` |
| KE gradient + pressure gradient | `8.445078e-05` | `8.445078e-05` | `6.78e-21` |
| **vertical advection of momentum** | `1.862080e-08` | `1.692844e-08` | **`2.878294e-08`** |
| lateral viscosity | `0` | `0` | `0` |

The **comparison floor is `2.03e-20`** — the next-largest row — and the
owning row is `1.4e+12` times it. The whole completed right-hand-side
difference is `2.878294e-08`, i.e. the owning row accounts for **100%** of
it. Every diagnostic component the map does not use is identically zero, or
the probe refuses.

**Owner: the vertical advection of momentum (`dyn_zad`).**

## 3. Then one more variable, because a term is not a statement

`dyn_zad`'s only non-geometric operand is the vertical velocity the
preceding `wzv` call produces, and the per-term record carries NEMO's own
copy of it. Substituting that ALONE, through the seam the GYRE rounds
already use:

| arm | completed right-hand side, legoESM − NEMO |
|---|---|
| the card | `2.878294e-08` |
| the card + NEMO's own vertical velocity | **`1.668689e-13`** |

So `dyn_zad`'s arithmetic — the four-cell averaging, the quarter factor, the
thickness, the areas, the surface and bottom ends — is right to `1.7e-13`,
and **the vertical velocity it is handed is the statement.**

## 4. The statement, on both sides

Reading NEMO's own vertical velocity out of the record and against
legoESM's, on the same faces:

| interface | NEMO `ww` | legoESM `w` |
|---|---|---|
| 0 (surface) | `-1.734121e-04` | `0` |
| 5..9 | `0` | `8.67e-05 … 1.73e-05` |
| 10 (bottom) | `0` | `0` |

legoESM's field is NEMO's MINUS a sigma-weighted copy of NEMO's own surface
value: `w_lego = ww − sigma·ww(surface)` reproduces legoESM's array to
`2.3e-07` out of `9.7e-05`, i.e. **99.76% of the difference is that one
redistribution**. That is legoESM's generic z-star diagnosis
(`vertical.py`, `diagnose_w_from_flux_div`): it spreads each column's own
surface tendency through the column so the vertical velocity vanishes at
BOTH ends. NEMO's does not vanish at the surface.

NEMO's statements, cited from the compiled source of this card's own build
(`VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo`):

> `pww(ji,jj,jk) = pww(ji,jj,jk+1) - ( ze3div(ji,jj,jk) + r1_Dt * e3t_1d(jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) ) ) * tmask(ji,jj,jk)`
> — `sshwzv.f90:295-298`

> `r3t(ji,jj,Kaa) = ssh(ji,jj,Kaa) * r1_ht_0(ji,jj)`   ! "after" ssh/h_0 ratio guess at t-column at Kaa (n+1)
> — `stp2d.f90:149`, immediately before `CALL wzv( kt, Kbb, Kbb, Kaa, uu(:,:,:,Kbb), vv(:,:,:,Kbb), ww, np_velocity )` at `stp2d.f90:153`

> `ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)`   ! "linear extrapolation of ssh to compute ww at the beginning of the next time-step"
> — `stprk3.f90:225`

So the scale-factor term is built from the after-SSH the PREVIOUS step left
in the after slot by linear extrapolation — not from a continuity
prediction. At the first step the extrapolation has never run, the slot
still holds the initial height, and the term is exactly zero. That is why
NEMO's recorded vertical velocity at `kt=1` is the plain bottom-up integral,
and it is what the 99.76% figure above measures.

**The two time-stepping programs differ here and a shared default cannot
serve both** (operator note BI). NEMO's modified leapfrog fills the same slot
from the barotropic continuity in `ssh_nxt` before `wzv_MLF` reads it, which
is the form legoESM already had and which DINO's round 39 measured. The form
is therefore selected from the time integrator the card already states.

## 5. The fix

Two statements, in one commit, each cited above.

1. `nemo_qco_wzv_operands` builds the first call's after-SSH from NEMO's own
   program: the RK3 linear extrapolation `2*eta_now − eta_before` when the
   card's momentum integrator is NEMO's RK3, the leapfrog's continuity
   prediction otherwise. An unknown form RAISES rather than silently running
   one of the two.
2. The vector-EEN VORTEX card states that it consumes NEMO's own vertical
   velocity (`zad_qco_evaluation = nemo_literal`), which GYRE, DINO and
   ORCA2 already do. It had inherited the generic default from the flux
   card, **which never calls `dyn_zad` at all** — the hidden-choice shape:
   a default that disagrees with the run, on a card built by copying a card
   the default was harmless on.

Neither piece works alone, and that is measured rather than argued: the
card's selection with the old after-SSH gives `2.880e-08`, i.e. nothing.

| arm | completed right-hand side, legoESM − NEMO |
|---|---|
| before (generic vertical velocity) | `2.878294e-08` |
| NEMO's vertical velocity, old after-SSH | `2.880131e-08` — no help |
| **both** | **`2.710505e-20`** |

After the fix every row of section 2 is at the floor
(`2.03e-20 / 6.78e-21 / 6.17e-21 / 0`), and handing the model NEMO's own
vertical velocity is a **no-op** (`2.710505e-20`, unchanged), which is the
sharpest statement available that the operand now agrees.

## 6. Both VORTEX ladders, before and after, every step

Bar `1.0e-15`. "Before" is round 4's published ladder, re-measured there on a
clean tree.

**Vector-EEN card** (`VORTEX_VEC-zco`), status `DEBT` both sides, first over
bar still `kt=2`:

| kt | u before | u after | ssh before | ssh after |
|---|---|---|---|---|
| 1 | `2.2204e-16` | `2.2204e-16` | `1.3553e-20` | `1.3553e-20` |
| 2 | `1.4325e-05` | **`3.3693e-06`** | `2.2685e-05` | **`3.7090e-08`** |
| 3 | — | `5.9832e-06` | — | `6.5057e-06` |
| 4 | — | `4.2479e-06` | — | `8.2200e-06` |
| 5 | — | `6.2107e-06` | — | `8.0021e-06` |
| 6 | — | `8.9156e-06` | — | `8.0199e-06` |
| 7 | — | `1.1073e-05` | — | `5.2308e-06` |
| 8 | — | `1.2745e-05` | — | `6.0089e-06` |
| 9 | — | `1.2944e-05` | — | `4.5933e-06` |
| 10 | `1.4621e-05` | `1.2543e-05` | `5.3334e-06` | `5.2990e-06` |

The `kt=2` height row lands on `3.7090e-08` against the FLUX card's own
`3.7088e-08` — the level round 4's substitution arm predicted, before this
code was written, from NEMO's record alone (`u 3.369e-06`, `ssh 3.709e-08`).
The prediction and the landed ladder agree to three digits in `u` and four
in `ssh`.

**Flux-ENS card** (`VORTEX-zco`), `DEBT`, first over bar `kt=2`: `kt=2`
`u 1.1355e-07 ssh 3.7088e-08`, `kt=10` `u 2.4640e-06 ssh 5.3364e-06` —
**every row equal to round 4's published ladder**, as it must be: the flux
card selects flux-form advection and never reaches `dyn_zad`.

## 7. Which cards execute the changed statement

The after-SSH statement is reached by any card that selects NEMO's own
first-`wzv` operands AND runs NEMO's RK3 stepper.

| card / recipe | selects NEMO's wzv operands | NEMO RK3 | executes the change |
|---|---|---|---|
| `VORTEX_VEC-zco` | yes, **new this round** | yes | **yes** |
| `GYRE-zco` | yes | yes | **yes** |
| `ORCA2-zps` | yes | yes | **yes** |
| DINO `nemo_dino_kamm` | yes | yes | **yes** |
| DINO `nemo_dino_kamm_mlf` | yes | no (leapfrog) | no — keeps the leapfrog form |
| `VORTEX-zco`, `LOCK_EXCHANGE-zco`, `OVERFLOW-zps` | no | — | no |
| DINO `legoesm_default`, `nemo_paper` | no | — | no |

**This is a shared landing.** It is not scoped to VORTEX and was not made so.

## 8. Gates

| gate | its own line |
|---|---|
| **VORTEX vector ladder kt=1..10** | section 6 — `kt=2` `u 1.4325e-05 → 3.3693e-06`, `ssh 2.2685e-05 → 3.7090e-08` |
| **VORTEX flux ladder kt=1..10** | section 6 — every row equal to round 4 |
| `LOCK_EXCHANGE-zco` kt=1..3 | `AT-BAR`, no row over the bar — unchanged |
| `OVERFLOW-zps` kt=1..3 | `DEBT`, first over bar `{T,u}` at `kt=2` — unchanged |
| **GYRE certified ladder, kt=1..10** | `DEBT`, **first over bar `{T,S,u,v,ssh}` at `kt=3` — the certified value, unchanged**; `kt=2` velocities `8.326673e-17` / `9.714451e-17`, the digits the certification receipt publishes. Of the 50 scored rows, **40 MOVED and all 40 moved TOWARD NEMO** (median factor `3.10`, e.g. `kt=3` `u 4.751547e-06 → 1.048051e-06`); **0 worsened** |
| per-term discriminator, clean tree | section 2; closure `1.355e-20` against a bar of `1e-18` |
| the four row plants | each `VISIBLE and ISOLATED` — the named row moves and no other — exit 1 |
| the `rhs-agrees` plant | `COLLAPSED`, exit 1 |
| per-term observer's effect on the model | right-hand side `1.36e-20`, `u`/`v` `2.78e-17`, `T`/`S`/`eta` exactly `0` — the compiled-rounding floor, `2 000×` below what it is used to attribute, and MEASURED rather than asserted |
| ordered control (NEMO's vertical-velocity call leaves the momentum accumulator untouched) | holds |
| `tests/ocean/unit/test_zad_qco_coupled.py` | `5 passed` |
| `tests/test_dispatch_hardening.py` (the new guard is pinned grow-only) | `24 passed` |
| GYRE from-rest year, day 30 / 240 / 360 | section 9 |
| DINO month gate, private work dir | section 9 |
| card battery, citation gate, push gate | section 9 |
| citation map audit over the default receipt | `unmapped: []`, `map entries failing: 0` |

## 9. The shared landing's own gates

GYRE executes the changed statement, so Decision 43/55/59 binds: the year
rows are re-run and registered, and the direction is reported, not the
magnitude alone.

### GYRE, the from-rest year

`nemo_testcase_l2_gyre_year_fromrest.py --member 0 --days 360 --snap-steps 6
--tag round5`, scored by `nemo_testcase_l2_gyre_year_owners.py --day-gap
--days 30,240,360`. The harness's own run-to-run floor is `~2e-10 K`.

| day | certified, T rms [K] | this round | direction |
|---|---|---|---|
| 30 | `2.3276772050683987e-06` | `2.3440e-06` | **AWAY, +0.7% relative** |
| 240 | `6.586171881479517e-05` | `6.5826e-05` | toward, −0.05% |
| 360 | `0.002670992385329469` | `5.4085e-05` | **toward, 49× better** |

**This is the one row that holds the round.** Two of three move toward NEMO,
one of them by a factor of fifty; the day-30 row moves away by seven parts in
a thousand, which is `80 000` floor units and above the `1e-3` relative
allowance note AT gives a statement that takes a certified row to the bar —
and no certified row went from debt to bar here. So the automatic rules do
not cover it and the operator decides. The round does NOT claim it.

### DINO, the month gate

`2.040288957e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03
K) -- PASS`, exit 0, private work directory. Identical to rounds 3 and 4,
i.e. `1.9e-10 K` from the certified value, which is the harness's own
run-to-run floor: **this round is INERT on DINO's from-rest month**, which
the card census predicts — that gate's card runs NEMO's leapfrog, and the
leapfrog form of the after-SSH is unchanged.

### The battery

`254 passed` with three failures, all three in the citation gate and all
three the same cause: this round's inserted lines moved the symbols the
citation map pins by text, so 56 map entries had drifted. Re-anchored by the
lane's own method — each entry to the line its own first anchor identifies,
each span's end from its own last anchor, the pinned extent recomputed from
the two, and the receipts' prose moved with the map in the same change. The
gate is green: `16 passed`, `map entries failing: 0`.

### Provenance of the year arm

The year ran at `34da4f2af`. Two commits landed after it: the dispatch
guard's message and its ratchet entry, and the freshwater scoping. Neither
is asserted to be inert — it is **measured**: the per-term discriminator and
BOTH VORTEX ladders were re-run at the final tip on a clean tree and are
identical to the digit (`2.710505e-20`; `kt=2` `u 3.3693e-06`, `ssh
3.7090e-08`; every other step equal).

## 9b. How far the transcription actually reaches, stated rather than implied

READ OFF THE CODE, not measured: NEMO's extrapolated guess is
`2*ssh(n) - ssh(n-1)`, and the second operand is the PREVIOUS step's entry
height. legoESM's RK3 lane does not carry it — the leap-frog lane does, and
every RK3 consumer falls back to the step-entry height when it is absent
(and the harnesses that drive these cards do not seed it). So on an RK3 card
the transcription currently evaluates to "the after slot holds the
step-entry height", i.e. the scale-factor term is exactly zero.

* At the FIRST step that is EXACTLY NEMO, because the extrapolation has
  never run — which is what NEMO's own recorded vertical velocity shows, and
  it is the whole of this round's measurement.
* At later steps it is an approximation of NEMO's extrapolation, not the
  extrapolation. Everything measured here is still measured: the VORTEX
  ladder re-seeds every `kt` from NEMO's own record, and the improvement is
  reported per step; GYRE's 40 moved rows all moved toward NEMO.

**Completing it requires legoESM's RK3 lane to carry the previous step's
entry sea surface height, which is a change to the state a run carries.**
That needs the operator's decision and is NOT taken here (section 11).

## 10. Choices made this round

| choice | ASKED or UNASKED | note |
|---|---|---|
| the vector-EEN VORTEX card consumes NEMO's own first-`wzv` vertical velocity instead of legoESM's generic z-star one | ASKED in effect | decision 73 makes the card a transcription of NEMO's vector-invariant momentum program; this is which routine's output that program feeds to `dyn_zad`, read from `stp2d.f90:153`, not a preference |
| the first `wzv` call's after-SSH follows the card's own time-stepping program (RK3 extrapolation, leapfrog continuity prediction) rather than one shared form | ASKED in effect, and it is the FIX | operator note BI: "a NEMO option that is right for one time-stepping program is NOT a shared default — it is selected per card, and the card's stepper is checked". Both forms are NEMO's, each cited |
| keying that form on the card's momentum time integrator rather than adding a new config field | UNASKED, and stated | a new field would be a knob whose default keeps the bug on GYRE, DINO and ORCA2, which rule 3 forbids. The key is NEMO's own relationship between two of its sources, not a coupling of unrelated choices. Revert on request |
| the FLUX VORTEX card is left on the generic vertical velocity | UNASKED, and stated | it runs flux-form advection and never calls `dyn_zad`, so the statement is not executed there; changing it would be a second variable in the same round. Its own `kt=2` owner is still open (section 11) |
| `zad_bottom_face_mask` is left at `min_rule` on the vector card | UNASKED, and stated | GYRE and DINO set `nemo_faithful`; this deck is flat-bottomed, so the two agree here and flipping it would be a second variable with nothing to measure. Named rather than silently carried |
| round 4's magnitude exclusion is retracted and the bound corrected in the instrument | not a choice | a defect fix; its non-vacuity is that it un-excludes the term |

## 11. OPEN

**ORCA2 IS UNMEASURED THIS ROUND AND THAT BLOCKS ITS CLAIMS.** ORCA2-zps
selects NEMO's first-`wzv` operands and NEMO's RK3 stepper, so it executes
the changed statement at every step, and its ladder was NOT re-run here.
Run it before any ORCA2 fidelity number from before this round is spent.

**DECISION NEEDED — finish the after-SSH transcription on the RK3 lane.**
NEMO's guess is `2*ssh(n) - ssh(n-1)`; legoESM's RK3 lane carries only
`ssh(n)`, so the term evaluates to zero at every step instead of only the
first (section 9b). Carrying the previous step's entry height is a change to
the state a run carries, which the lane's standing rule says is never taken
without the operator. One line: **carry the previous step's entry sea
surface height on the RK3 lane (Y), or leave the after-SSH slot equal to the
step-entry height (X, today)?** Current value X. My pick: Y, because the
statement is NEMO's and the round has already shown what this operand is
worth — but the measurement that settles it is cheap and should come first:
the same per-term discriminator at `kt=2` rather than `kt=1`, where NEMO's
extrapolation is no longer zero.

**Round 6 — note BK's resolution ladder, if the vector card is at bar level.**
It is not yet: `kt=2` is `3.37e-06` against a bar of `1e-15`, and the flux
card's own `1.14e-07` is now the nearer target. The vector card's remaining
`kt=2` velocity residual is `30×` the flux card's, so ONE difference remains
between the two momentum programs at that step, and the same walk that found
this one will find it — the per-term rows are now all at the floor, so the
remainder is NOT in the completed pre-stage right-hand side and must be in
what the stages do with it. Round 4's own stage table already says where to
look: with NEMO's pre-stage right-hand side handed over, stage 1 was EXACT
(`1.1e-16`), so the residual is stage 2 or stage 3.

**The flux card's `kt=2` owner is still unnamed** (stage-3 residual
`1.204e-07`), unchanged by this round, with the implicit vertical viscosity
still on its candidate list and still unmeasured by a committed instrument.

**Then the resolution ladder** (decision 74, note BK) — 15 km and 10 km,
still owed.

**What this bears on ORCA2 and on the developed state.** ORCA2, GYRE and
DINO's RK3 recipe all execute the changed statement at EVERY step of a
developed state, where the after-SSH extrapolation is not zero. GYRE's
ten-step ladder shows the size of that: 40 of 50 scored rows moved, all
toward NEMO, median factor `3.10`. Any ORCA2 finding that rests on a
developed-state momentum row, on the first RK3 stage, or on the barotropic
slow forcing is downstream of this and should be re-measured before it is
spent.

## 12. Review

The codex quota guard reads `80.0% used`. At the 80% line the brief puts the
review on a Claude reviewer, so ONE fresh Claude code-reviewer was the
review, told to refute, told to verify every cited NEMO line verbatim, and
told specifically to attack the plants and the row map — because that is
what rounds 3 and 4 got wrong.

**Verdict: SHIP, with follow-ups.** Findings and dispositions:

| # | finding | disposition |
|---|---|---|
| 1 HIGH | keying the after-SSH form on the time integrator is a SHARED landing: GYRE and ORCA2 both execute it, and ORCA2 was not re-measured | **ACCEPTED, and it is a gate, not a note.** GYRE's certified ladder was re-measured here (40 of 50 rows moved, all toward NEMO, 0 worsened) and its year is in section 9. **ORCA2 is UNMEASURED this round and no ORCA2 fidelity claim may be spent until its ladder is re-run** — carried into section 11 |
| 2 HIGH | the freshwater term was added after BOTH after-SSH branches, but NEMO's RK3 extrapolation carries none — only the leapfrog's `ssh_nxt` does | **CONFIRMED AND FIXED** (`88785a0b6`). Measured inert rather than argued: the operand is reachable only through one keyword and no caller in the repository ever passes it a value, so no number moves today; the deviation would have fired the moment that wiring landed, on a card with real evaporation minus precipitation |
| 3 MEDIUM | the extrapolation collapses to a no-op past the first step because the RK3 lane carries no previous-step height, so the `kt=2..10` improvement is carried by the OTHER half of the fix | **AGREED, and it was already section 9b.** Restated in the reviewer's sharper words: **the ladder tables past `kt=1` do not validate the extrapolation term itself.** The decision that would complete it is in section 11 |
| 4 LOW | the reviewed range did not register the new guard in the grow-only dispatch ratchet | already fixed in `442671ec0`, which landed while the review was running |

What the reviewer checked and did NOT fault, which is the part the
attribution rests on: the two-sided bound is the right correction and its
numbers are internally consistent; the row map matches the code's own
bundling and the closure check reads the array captured BEFORE any plant
mutates it; all five plants perturb values that actually flow into the owner
decision rather than a printed label, and the plant size is correctly placed
between the total and the term peaks; asking the tendency call for its
decomposition is structurally additive and cannot feed back into the
tendency; both new tests are non-vacuous; and **every cited NEMO source line
is verbatim-correct** — which this campaign has not always managed.

**Single reviewer, stated plainly.** The campaign's default is two. This
round had one, per the brief's instruction at the quota line. That is a
weaker review than the default and is recorded as such.
