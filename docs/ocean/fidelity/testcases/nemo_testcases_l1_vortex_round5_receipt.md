# RECEIPT — VORTEX round 5: the vector card's second step is owned by the vertical velocity, not by either surviving term

Date 2026-09-30. Lane tip at the start `5add31a068ad`. Preregistration
`PREREG_nemo_testcases_l1_vortex_round5.md`, frozen before any measurement.
Evidence `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round5`.

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
