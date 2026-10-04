# Round 220 / VORTEX_SMT round 9 — SMT-1: ORCA2 rung 0's background vertical mixing on the seamount

**Decision 93 (user, operator note CE), rung 1 of the seamount mini-ladder.** One
module of the seamount vector deck moves to ORCA2 rung 0's values; nothing else
changes, and the round measures what that module does on both sides.

**Headline.** The background vertical mixing is **faithful**: with the tracer
diffusivity switched on for the first time on this deck (`rn_avt0` 0 → 1.2e-5,
so the implicit tracer vertical solve stops being identically inert), the whole
50-row registry is the SMT-0 registry to four significant digits, and the
first-over-bar row does not move. The enhanced vertical diffusion rung 0 runs
beside it **never fires here** (minimum N² over the run is +9.0e-06 s⁻², ten
orders the wrong side of the trigger) — and, separately and as a **FINDING**,
the card can state it and the executed model does not run it.

---

## 1. The deck, line by line, cited to the rung-0 namelist

`namelist_cfg_smt1_vec_een.patch`, applied to the shipped
`tests/VORTEX/EXPREF/namelist_cfg`. Against the certified seamount vector deck
(`namelist_cfg_smt_vec_een.patch`) the diff is **exactly one hunk, five lines**,
and the preflight printed it before the build:

| line | SMT vector deck | SMT-1 deck | rung-0 source |
|---|---|---|---|
| `ln_zdfevd` | `.false.` | `.true.` | `namelist_cfg:409` |
| `nn_evdm` | (absent, ref default 0) | `0` | `namelist_cfg:410` |
| `rn_evd` | (absent) | `100.` | `namelist_cfg:411` |
| `rn_avm0` | `1.e-4` | `1.2e-4` | `namelist_cfg:417` |
| `rn_avt0` | `0.e0` | `1.2e-5` | `namelist_cfg:418` |

Rung 0 also pins `nn_avb = 0` (`:419`), `nn_havtb = 0` (`:420`) and
`ln_zdfcst = .true.` (`:421`); the seamount deck already resolves all three, so
they are **not** in the diff — the background is uniform, with no latitude shape
and no profile. `cn_exp` is deliberately unchanged: this is the same experiment
with one module switched, and only the build and evidence directories are new
(the `smtvecr3` pattern).

NEMO's own control print, read back out of `ocean.output` rather than assumed:

```
enhanced vertical diffusion             ln_zdfevd =  T
   applied on momentum (=1/0)             nn_evdm =            0
   vertical coefficient for evd           rn_evd  =    100.00000000000000
vertical eddy viscosity                 rn_avm0   =    1.2000000000000000E-004
vertical eddy diffusivity               rn_avt0   =    1.2000000000000000E-005
zdf_evd : Enhanced Vertical Diffusion (evd)
```

## 2. Build, record, admission

New configurations `VORTEX_SMT1_VEC_R8_OMIP_L1` (plain) and
`VORTEX_SMT1_VEC_R8_OMIP_L1_P3` (instrumented), copied from the certified
seamount vector pair with its recorder patches and the seamount `usrdef_zgr`
hook; cpp keys unchanged (`key_qco key_RK3 key_vco_1d3d`). Round 3's and round
7's builds and records are untouched.

* **ADMITTED**, `restart_byte_identical: true` — the kt=10 restarts of the plain
  and the instrumented build are byte-identical, which is the additions-only
  proof (note AS).
* Header plant fired (`PLANT_FIRED header`).
* Ten `oracle_step_entry` frames, ten barotropic frames, the stage records at
  kt=1 (s1/s2/s3) and the stage-term records.
* Wall time: the two builds plus the two runs completed in ~14 minutes
  (15:12 → 15:26 UTC), the runs themselves in seconds at this size.

## 3. NEMO sanity, and what the module did on NEMO's side

From the committed probe
(`nemo_testcase_l1_vortex_smt_round9_smt1_probe.py`), reading NEMO's own
frames:

| kt | max abs ssh [m] | max abs u [m/s] | NaN | max abs (SMT-1 − SMT-0) T [K] | u [m/s] | ssh [m] |
|---|---|---|---|---|---|---|
| 1 | 0.9162 | 0.8723 | 0 | 0 | 0 | 0 |
| 2 | 0.8868 | 0.8842 | 0 | 2.294e-06 | 4.582e-08 | 0 |
| 5 | 0.8510 | 0.9461 | 0 | 9.065e-06 | 6.599e-07 | 2.294e-08 |
| 10 | 0.8000 | 1.0113 | 0 | 1.916e-05 | 3.923e-06 | 4.737e-08 |

No NaN at any step; the vortex is the same vortex. The initial state is
**bit-identical** to SMT-0's (R9-P8 CONFIRMED) — nothing in the initial state
reads `namzdf`. By kt=10 the module has moved NEMO's own temperature by
1.9e-05 K, which is the scale any fidelity error in it would have had to
compete with.

**Where avt differs from the SMT-0 run:** everywhere wet, and by construction
rather than by measurement — `ln_zdfcst` with `nn_avb = 0` makes `avt` the
single scalar `rn_avt0`, so it goes from **identically zero** (round 7's
finding, `zdfphy.f90:206-208,227,349`) to 1.2e-5 m²/s at every wet interface,
and `avm` from 1.0e-4 to 1.2e-4.

**Does enhanced vertical diffusion fire? NO — 0 cells, every step.**
`zdfevd.F90:93` writes `p_avt = rn_evd * wmask` wherever
`MIN(rn2, rn2b) <= -1.e-12`; under RK3 both arms are the step-entry tracer
(`stprk3.F90` forms `rn2b` on Nbb and copies `rn2 = rn2b` before any stage), so
one arm per recorded step is the whole trigger. Scored with **this deck's**
S-EOS (decision 69: `rn_a0 = 0.28`, every other coefficient zero), the minimum
N² over all ten steps is **+9.0e-06 s⁻²** — ten orders the wrong side of the
threshold. **R9-P3 CONFIRMED.**

## 4. The legoESM card

`VORTEX_SMT1_VEC-zps` — the certified seamount vector card with the same five
values stated as executed configuration, not as defaults:
`A_v = 1.2e-4`, `K_v = 1.2e-5`, and an explicit enhanced-vertical-diffusion
selection with `K_conv = 100` (`rn_evd`), `nu_conv = 0`
(`nn_evdm = 0`; `zdfevd.F90:106` guards the momentum arm), the trigger
`MIN(rn2, rn2b) <= -1.e-12` and the deck's own S-EOS as the fluid it sees.
`ln_zdfcst` with a uniform background is those two scalars, so the card selects
**no** vertical-mixing closure, and every other physics module is stated OFF
rather than inherited. The rung dispatch is fail-closed (an unknown rung and a
flux-form SMT-1 both raise — the flux deck is not carried up the ladder because
ORCA2 is vector-invariant), and the validator refuses a card that drops any of
the three rung-0 values.

A field-by-field test proves the only configuration rows that moved are `A_v`,
`K_v` and the physics block.

**Geometry identity: 0 ULP on all twelve fields** against the new record's own
`mesh_mask.nc` (`k_bot`, `e3t_0`, `e3u_0`, `e3v_0`, `e3f_0`, the four card
operands, the two resolved qco-arm faces, the three masks), with the gate's own
non-vacuity row live (1164 cells where the resolved `e3u_0` differs from
`e3t_0`). **R9-P7 CONFIRMED.**

## 5. The ladder, row by row against the SMT vector registry

kt=1 is at the bar on all five fields (T and S exactly 0, u/v one quantum,
ssh 1.4e-20). First over bar: **kt=2, T/u/v/ssh** — the same row as SMT-0.

| kt | field | SMT-0 (vector) | SMT-1 | ratio |
|---|---|---|---|---|
| 1 | T / S | 0 | 0 | — |
| 1 | u / v | 2.220446e-16 | 2.220446e-16 | 1.000 |
| 1 | ssh | 1.355253e-20 | 1.355253e-20 | 1.000 |
| 2 | T | 3.618737e-12 | 3.618737e-12 | 1.000 |
| 2 | u | 1.267541e-09 | 1.267539e-09 | 1.000 |
| 2 | v | 2.548881e-10 | 3.058414e-10 | **1.200** |
| 2 | ssh | 2.664535e-15 | 2.664535e-15 | 1.000 |
| 10 | T | 1.484450e-08 | 1.484277e-08 | 1.000 |
| 10 | u | 1.237234e-06 | 1.237110e-06 | 1.000 |
| 10 | ssh | 1.672757e-08 | 1.673112e-08 | 1.000 |

43 of 50 rows change in the last digits; **exactly two families change
structurally**:

* **v at kt=2, 3, 4** — ×1.200, ×1.211, ×1.212. `2.548881e-10 × 1.2 =
  3.0587e-10` against the measured `3.058414e-10`: the row is **proportional to
  `rn_avm0`** to four digits. That is a scaling test, and it says the owner is a
  **pre-existing** momentum error the new viscosity merely rescales, not a new
  statement. By kt=5 the ratio is back to 1.000, i.e. the advective error has
  overtaken it.
* **S** — the uniform-salinity rows move from 4-5 quanta to 6-8 quanta of
  2.030e-16 and cross the 1e-15 bar at kt=5 and kt≥7 (1.015e-15 → 1.218e-15 /
  1.421e-15 / 1.624e-15). Registered as a status change; S carries no signal on
  this deck (`rn_b0 = 0`), so these are last-bit composition in a field the
  physics does not read.

**R9-P5 is REFUTED and reported as such:** the first-over-bar row is unchanged
rather than becoming a tracer row, because the tracer solve did not introduce a
measurable error.

## 6. The owner walk: the implicit tracer solve over partial cells

The statement Decision 93 said this rung would make measurable is
`trazdf.F90:219-221`, the association of the tridiagonal:

```fortran
zwi(ji,jk) = - p2dt * zwt(ji,jk  ) / e3w(ji,jj,jk  ,Kmm)
zws(ji,jk) = - p2dt * zwt(ji,jk+1) / e3w(ji,jj,jk+1,Kmm)
zwd(ji,jk) =   e3t(ji,jj,jk,Kaa) - ( zwi(ji,jk) + zws(ji,jk) )
```

Under this build's keys the two thicknesses are **different objects**:
`domzgr_substitute.h90:80` makes `e3w_0(i,j,k) = e3w_1d(k)` — the uniform 500 m
ladder, even at a partial bottom cell — while `:97` makes
`e3t_0(i,j,k) = e3t_3d(i,j,k)`, the thinned cell. The off-diagonals divide by
the 1-D ladder; the diagonal carries the 3-D thickness.

**legoESM's association is NEMO's, and the measurement is non-vacuous.**
`nemo_e3w_kmm` resolves `e3w_0` from the card's raw mesh field when the
coordinate carries one (`nemo_e3w_mesh_reference`, which
`create_z_star_from_thicknesses` sets by default), and the SMT card supplies the
uniform 500 m ladder; only a card with no mesh `e3w_0` falls to the
midpoint-of-the-live-thickness arm. The result is the registry above: with the
solve switched from identically inert to active, **T at kt=2 does not move at
all** (3.618737e-12 both sides) and kt=3 moves by 0.011 %.

That it could have moved is arithmetic, not assertion. The per-step diffusive
increment is `K dt / dz² = 1.2e-5 × 2880 / 500² = 1.4e-7` relative, and NEMO's
own two runs differ by 2.294e-06 K at kt=2. The midpoint arm's divisor at the
seamount's thinnest bottom cell is `(500 + 50.67)/2 = 275.3` m against NEMO's
500 m — a **45 % wrong divisor** on those interfaces, which would have put an
O(1e-6 K) error into exactly the cells the seamount creates, six orders above
the 3.6e-12 the row actually carries. **First non-bit statement from this
module: none.** The rung is faithful.

## 7. FINDING — the card's stated enhanced vertical diffusion is not executed

Pre-registered as a sanity check, found by plant, and reported as a finding
rather than fixed inside this round:

* **The plant.** Multiplying `rn_evd` by ten thousand (100 → 1e6) and running
  one full step changes the temperature and the velocity by **exactly 0.0**.
  The selection the card states is not reaching the solve.
* **It is inert here**, by §3: NEMO's `zdfevd` never fires on this rung, so
  nothing is wrong with this round's numbers.
* **It would not have been inert if it had run.** The model's `nemo_bn2` EVD
  trigger builds alpha/beta from `NemoSEOSConfig()` **defaults** — the DINO
  coefficient set — because no recipe threads its own (the comment above the
  call says exactly this). On this deck that fluid is wrong, and the trigger
  fires on **61 cells at every step, including the pristine initial state**,
  where the deck's own S-EOS gives N² = +9.0e-06 s⁻² and NEMO fires on none.
  A constant count on a pristine stably-stratified state is an instrument
  signature, not physics.

Both facts are now **tests**, not sentences.

**DECISION_NEEDED (operator).** Rung 0's `ln_zdfevd` is real for ORCA2, where it
does fire. Options, my pick first:

1. **Wire it in a dedicated round, before SMT-2.** Two statements: thread the
   card's S-EOS coefficients into the EVD trigger, and give the implicit solve
   NEMO's composition (`zdfevd` **replaces** `avt` where it fires; the pipeline
   path would **add** the background). Both are shared-path model edits with
   their own gates, and ORCA2 needs both.
2. **Carry SMT-2 first** and wire EVD when a rung makes it fire.
3. Drop the EVD declaration from the SMT-1 card (I do not recommend it: the
   deck resolves it, and a card that hides a resolved switch is the hidden
   choice the lane bans).

## 8. Gates

| gate | result |
|---|---|
| geometry identity, SMT-1 vs its own record | 12/12 fields, 0 ULP |
| initial state vs SMT-0 | bit-identical, 7 fields |
| SMT-1 ladder / 50-row registry | first over bar kt=2 T/u/v/ssh; status DEBT |
| SMT vector, SMT flux, VORTEX-zco registries | 0/50 rows moved |
| the other five flat VORTEX cards + two tanks | see §9 |
| card unit tests | 13 passed |
| NEMO admission | ADMITTED, restarts byte-identical, plant fired |

No file under `packages/` or `src/` changed except the fidelity recipe module
(a new card and its validator branch), so every other card is inert **by
construction**; the registries above are the measurement of that, not a
substitute for it.

## 9. Predictions, scored

| # | outcome |
|---|---|
| R9-P1 | **CONFIRMED** — five lines, one hunk, nothing else |
| R9-P2 | **CONFIRMED** — kt=10 restarts byte-identical |
| R9-P3 | **CONFIRMED** — 0 trigger cells with the deck's S-EOS (min N² +9.0e-06) |
| R9-P4 | **CONFIRMED** — avt 0 → 1.2e-5, avm 1.0e-4 → 1.2e-4, read back from NEMO |
| R9-P5 | **REFUTED** — the first-over-bar row is unchanged; the tracer solve added nothing measurable |
| R9-P6 | **not reached** — there is no first-over-bar row for this module to own; the association was measured faithful instead, with the 45 %-divisor bound as the non-vacuity argument |
| R9-P7 | **CONFIRMED** — 0 ULP, twelve fields |
| R9-P8 | **CONFIRMED** — initial state bit-identical to SMT-0 |

## 10. ORCA2 pointer

* **The association ORCA2 inherits is already right.** ORCA2 is a partial-cell
  configuration and its implicit tracer solve takes the same shared
  `nemo_e3w_kmm`; this round measured that path with a live diffusivity over
  partial cells for the first time and it is faithful. ORCA2's rung-0 walk does
  **not** need to spend a round on `trazdf.F90:219-221`.
* **Two things ORCA2 does need, and they are §7's.** ORCA2 rung 0 runs
  `ln_zdfevd = .true.` with `rn_evd = 100` and `nn_evdm = 0`, on TEOS-10 and on
  a real ocean where convection fires. Its card must (a) have the EVD selection
  actually reach the solve — this round proved a stated selection can be
  silently dropped — and (b) compose it NEMO's way: `zdfevd` **replaces**
  `avt` with `rn_evd` where the trigger fires and leaves `avm` alone at
  `nn_evdm = 0`, which is **not** the additive composition the pipeline path
  gives. ORCA2's card selects the TKE closure, so it takes the implicit
  fallback and the max-floor composition, which is the right one — that should
  be confirmed on its lane by the same plant used here (multiply `rn_evd` and
  watch the step), not assumed.
* **The background pair itself is cheap and safe to carry:** 1.2e-4 / 1.2e-5
  with `nn_avb = nn_havtb = 0` is two scalars on the config, measured faithful
  over partial cells here.

## 11. OPEN

* The 100-day SMT-1 comparison (round-210 scorer + movie) is **preregistered,
  not run**: the deck patch `namelist_cfg_smt1_vec_een_100d.patch` and the
  `smt1vec100d` acquisition variant are committed and reuse this round's
  certified binaries by hash. Prediction: the day-100 T rms stays within a
  factor of two of the SMT-0 vector card's 4.35e-05 K, since the module moves
  NEMO's own T by only 1.9e-05 K over ten steps.
* §7's DECISION_NEEDED.
* The S rows crossing the bar at kt≥5 (6-8 quanta of 2.030e-16) are registered,
  not walked.
