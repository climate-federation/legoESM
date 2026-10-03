# Round 211 / VORTEX_SMT round 1 — the NEMO side: a seamount, partial steps, four records

**Decision 88 (user, 2026-10-03), operator note CC.** This round builds NEMO's
VORTEX with topography and acquires its records. No legoESM card, no legoESM
physics or configuration change, no ladder: those are round 2 and later.

---

## 0. DECISION 89 — ONE cpp KEY MOVES, AND NEMO IS THE ONE THAT SAYS SO

**Status: pending with the user; the operator authorised proceeding under it
for this round, which lands no legoESM statement either way. If the answer is
"keep the key", this round's two builds and four records are discarded and
nothing else in the repository has to be undone.**

The round brief said to copy the certified instrumented builds with their
**cpp keys unchanged** and to set `ld_zps = .TRUE.`. **Those two instructions
cannot both hold, and NEMO is the one that says so.** The certified VORTEX
cards compile `key_qco key_vco_1d key_RK3`, and
`src/OCE/DOM/domzgr.F90:259` is

```fortran
IF( l_zps )   CALL ctl_stop( 'STOP','domzgr: key_vco_1d and l_zps=T are incompatible. Fix usrdef_zgr !' )
```

Under `key_vco_1d` there is no three-dimensional `e3t` array at all — `dom_oce.F90:359-362`
allocates the 1-D vertical arrays only — so a partial cell has nowhere to
live. The key that NEMO itself labels **"z-partial cells"** is `key_vco_1d3d`
(`domzgr.F90:242-243`), and it is the key both shipped partial-step
user configurations (`tests/OVERFLOW`, `tests/IWAVE`) guard their zps branch
with (`lk_vco_1d3d`).

So this round changes **one** cpp key, `key_vco_1d` → `key_vco_1d3d`, and
nothing else: `key_qco` and `key_RK3` are carried, `key_xios` and `key_agrif`
are dropped exactly as every other VORTEX variant drops them. It is reported
here as the first line of the receipt rather than buried, because it is the
single place where the built configuration departs from "the certified build
with a new bathymetry". Decision 88's own words are "partial steps ON
(`ln_zps`)", and partial steps are reachable by exactly one route.

The acquisition re-reads that refusal out of the compiled tree before it
builds anything (`run.sh`, the `SMT_ZGR` preflight block), so if a future NEMO
drops the restriction the script says so instead of quoting this paragraph.

---

## 1. The bathymetry, and where every constant comes from

```
h(x,y) = 5000 m  -  1000 m * exp( -((x-x0)^2 + (y-y0)^2) / (150 km)^2 )
(x0, y0) = (-300 km, 0 km)
```

`H0 = 5000 m`, `A = 1000 m`, `L = 150 km` and the 300 km westward shift are
the **user's**, stated in Decision 88. The only thing this round had to read
out of NEMO is where the vortex centre is and which way west points:

| quantity | value | citation |
|---|---|---|
| vortex centre | `(glamt, gphit) = (0, 0)` km | `tests/VORTEX/MY_SRC/usrdef_istate.F90:79-80` sets `zx = glamt*1.e3`, `zy = gphit*1.e3`, and `:86` puts the anomaly at `exp(-(zx²+zy²)/zlambda²)`; the velocity does the same at `:98-99`, `:105` |
| `glamt` units | kilometres | `tests/VORTEX/MY_SRC/usrdef_hgr.F90:78` "Position coordinates (in kilometers)" |
| west | `-glamt`, for `nn_rot = 0` | `usrdef_hgr.F90:108`: `plamt = zroffsetx + rn_dx*1.e-3*(zti-0.5)`, i.e. `glamt` increases with the i index |
| the deck's `nn_rot` | `0` | `tests/VORTEX/EXPREF/namelist_cfg:24` |

Everything else is the shipped 30 km deck, untouched: `rn_dx = rn_dy = 30000`,
`rn_dz = 500`, `rn_Dt = 2880`, `rn_ppgphi0 = 38.5`, `rn_ppumax = 1.0`,
`nn_e = 48`, both lateral operators OFF, the shipped simplified equation of
state (decision 69). The deck patches change four lines and no physics:
`cn_exp`, `nn_itend`, `nn_stock`, `ln_meshmask`.

---

## 2. The partial-step construction, statement by statement

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/vortex_smt_usrdef_zgr.F90`.
Its module header carries the same list; this is the short form.

| what | transcribed from | note |
|---|---|---|
| 1-D reference coordinate `zgr_z` | `tests/VORTEX/MY_SRC/usrdef_zgr.F90:89-161` | verbatim; ten uniform 500 m levels unchanged |
| land mask `k_top` | `tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:113-115` | `z2d = 1`, `lbc_lnk`, `k_top = NINT(z2d)`; the same mask VORTEX's own `:187-193` builds |
| minimum partial cell | `OVERFLOW:204` | `ze3min = 0.1_wp * rn_dz` = 50 m |
| bottom level | `OVERFLOW:209-212` | `k_bot = jpkm1`; `DO jk = jpkm1,1,-1 ; WHERE( zht < pdepw_1d(jk)+ze3min ) k_bot = jk-1` |
| land applied to `k_bot` | `OVERFLOW:143 / :160 / :197` | `k_bot = k_bot * k_top` |
| partial `e3t` | `OVERFLOW:221-225` | `e3t(ik) = MIN(zht, pdepw_1d(ik+1)) - pdepw_1d(ik)`; `e3t(ik+1) = e3t(ik)` |
| `e3u`, `e3v` | `tools/DOMAINcfg/src/domzgr.F90::zgr_zps:1162-1170` | `MIN` of the two neighbouring `e3t` |
| `lbc_lnk` on `e3u`, `e3v` before `e3f` | `zgr_zps:1176-1177` | |
| `e3f` | `zgr_zps:1191-1197` | `MIN` of the two neighbouring `e3v` |
| `lbc_lnk` on `e3f` | `zgr_zps:1198` | |

**The exclusion that mattered, named per the reuse rule.** OVERFLOW's zps
branch finishes with `pe3u = pe3v = pe3f = pe3t` and documents why that is
legal *there*: `OVERFLOW:228-231`, "HERE OVERFLOW configuration: e3 increases
with i-index and identical with j-index". A seamount varies in **both**
directions, so that shortcut is wrong here and was **not** ported; NEMO's
general two-dimensional rule from the DOMAINcfg tool was transcribed instead.

**Three more exclusions, deliberate:** `zgr_zps`'s duplication of row `jj=1`
onto `jj=2` (`:1203-1207`, which NEMO itself flags `!!gm bug ?` and which is
ORCA-specific), its ice-shelf branches (no cavities here), and its
`WHERE(e3 == 0) e3 = e3_1d` repair (`:1179-1184`, `:1200-1202`) — nothing can
be zero here because every array is initialised to the 1-D reference first.

**What stays 1-D.** Under `key_vco_1d3d`, `dom_zgr` calls `usr_def_zgr` with
exactly four optional arrays, `e3t_3d e3u_3d e3v_3d e3f_3d`
(`domzgr.F90:265-269`), and applies `lbc_lnk` to all four itself afterwards
(`:273-274`). `e3w`, `gdept` and `gdepw` remain the 1-D reference. This is
NEMO's own zps shape, not a simplification taken here.

---

## 3. Can ten 500 m levels carry a 1000 m seamount? — PREDICTED, THEN MEASURED

Pre-registered before any run (`phase3/vortex_smt/round1/predictions.md`):

| # | prediction | outcome |
|---|---|---|
| P1 | bathymetry min 4000.000 m, max 5000.000 m | **CONFIRMED** — `4000.000` / `5000.000` m, NEMO's own control print |
| P2 | `k_bot` 8 at the summit, 10 in the far field; the summit cell FULL | **CONFIRMED** — `k_bot` 8 / 9 / 10 over 5 / 56 / 3908 wet columns; summit `e3t = 500.0000` m |
| P3 | bottom `e3t` ≥ 50 m (the `ze3min` floor), ≤ 500 m | **CONFIRMED** — min `50.6710` m, max `500.0000` m; NEMO prints `minimum thickness of the partial cells = 10 % of e3 = 50.0` |
| P4 | the zps run differs from the flat one at kt=1 | **REFUTED AS WORDED** — see §5; at kt=1 the difference is the MASK, not the partial cells |
| P5 | the vortex still drifts west | **CONFIRMED** — §5 |
| P6 | no NaN at kt=10 or day 100 | **CONFIRMED** — explicit `isfinite` assertion, not a `nan*` reduction |

So the answer to the brief's "if ten uniform 500 m levels cannot represent a
1000 m seamount, STOP" is: **they can, and nothing was tweaked to make them.**
What makes it work is NEMO's own `ze3min = 0.1 * rn_dz` = 50 m floor, which
drops a level rather than create a cell thinner than a tenth of the reference.
There is therefore **no DECISION_NEEDED on representability.**

**The seamount as NEMO resolved it** — the i-row through the summit, from
`ocean.output` (`glamt` in km, depth and thicknesses in m):

| `glamt` | 5000−h | `k_bot` | `e3t(k_bot)` | `e3u(k_bot)` |
|---:|---:|---:|---:|---:|
| −480 | 4763.072 | 10 | 263.0722 | 132.1206 |
| −450 | 4632.121 | 10 | 132.1206 | 132.1206 |
| −420 | 4472.708 | 9 | 472.7076 | 302.3237 |
| −390 | 4302.324 | 9 | 302.3237 | 147.8562 |
| −360 | 4147.856 | 9 | 147.8562 | 147.8562 |
| −330 | 4039.211 | 8 | 500.0000 | 500.0000 |
| **−300** | **4000.000** | **8** | **500.0000** | **500.0000** |
| −270 | 4039.211 | 8 | 500.0000 | 500.0000 |
| −240 | 4147.856 | 9 | 147.8562 | 147.8562 |
| −210 | 4302.324 | 9 | 302.3237 | 302.3237 |

Two things to read off it. The summit sits exactly on a w-level, so its cell
is **full**, not partial — the 300 km offset lands on a T-point because the
mesh puts T-points at −900 + 30 n km. And `e3u` is visibly **not** `e3t`: at
`glamt = −480` the U-face carries 132.1 m where the T-cell carries 263.1,
which is the `MIN` of the two neighbours. That is the one place OVERFLOW's
shortcut would have been wrong, and the row is the evidence that it was not
taken.

The seamount removes **66 T-cells**, all at levels 9 and 10, in a disc
spanning `glamt` −420 … −180 km and `gphit` −120 … +120 km — i.e. centred on
(−300, 0) km, which is where Decision 88 put it.

---

## 4. The four records

Built, run and admitted by the committed acquisition
(`run.sh --variant smtflx|smtvec|smtflx100d|smtvec100d --run`). Four NEMO
configurations were compiled: `VORTEX_SMT_OMIP_L1{,_P3}` and
`VORTEX_SMT_VEC_R8_OMIP_L1{,_P3}`, the `_P3` ones carrying the same writers
their certified parents carry (the step record on both; the stage-2/3 term
writer additionally on the vector card, which is the round-192 build).

| record | run length | frames | restart byte-identical to the PLAIN build | header plant | wall (instrumented / plain) |
|---|---|---:|:--:|:--:|---:|
| `VORTEX_SMT_OMIP_L1_P3/kt1_10` | 10 steps | 24 | **yes** | fired | 2 s / 2 s |
| `VORTEX_SMT_VEC_R8_OMIP_L1_P3/kt1_10` | 10 steps | 26 | **yes** | fired | 2 s / 3 s |
| `VORTEX_SMT_OMIP_L1_P3/day100` | 3000 steps = 100 d | 3064 | **yes** | fired | 112 s / 121 s |
| `VORTEX_SMT_VEC_R8_OMIP_L1_P3/day100` | 3000 steps = 100 d | 3066 | **yes** | fired | 101 s / 101 s |

All four `ADMITTED`; geometry `67 × 67 × 11` with the two-cell halo on each
side, i.e. the shipped 63 × 63 × 11 box. **Additions-only is proved the way
note AS requires** and not by a stream comparison: for every record the
instrumented build's restart at the full run length is byte-identical to the
un-instrumented build's. The checker parses each record's own header and
predicts no size (note BD); its header plant turns it red on all four, so the
guard the admission rests on is shown able to fail.

**One acquisition bug, found and fixed rather than worked around.** The first
100-day admission REFUSED a record that cannot exist: it asked for 3000
step-entry records when the writer fires only for the first sixty
(`stprk3_step_record.patch:19`, `IF( lwp .AND. kstp >= nit000 .AND. kstp <=
nit000 + 59 )`). `run.sh` now carries the record count separately from the run
length; the restart it compares is still the one at step 3000. The NEMO runs
themselves were correct the first time — 100 daily restarts per arm — and were
re-run from scratch anyway so the committed tool is the one that produced the
admitted evidence.

The daily cadence `nn_stock = 30` is instrumentation and is stated, not
hidden: it is round 210's own cadence for the flat cards, and the deck
otherwise carries NEMO's shipped `nn_itend = 3000`.

---

## 5. NEMO's own output — the sanity the brief asks for

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_smt_round1_nemo_sanity.py`,
committed; its results are in `phase3/vortex_smt/round1/sanity_{flux,vec}.json`.
Every field is asserted finite with an explicit `isfinite` test — **no
`nanmax`**, which would have hidden exactly the failure being tested for.

| | flux card | vector card |
|---|---:|---:|
| kt=10 max \|ssh\| [m] | 7.805441e−01 | 8.024420e−01 |
| kt=10 max \|u\| [m/s] | 1.011721e+00 | 1.014659e+00 |
| kt=10 max \|v\| [m/s] | 1.008228e+00 | 1.011055e+00 |
| day 100 max \|ssh\| [m] | 3.405582e−01 | 4.577891e−01 |
| day 100 max \|u\| [m/s] | 3.306587e−01 | 8.304996e−01 |
| day 100 max \|v\| [m/s] | 3.646732e−01 | 9.466118e−01 |
| NaN anywhere | none | none |

**The vortex still drifts west**, on both cards — the longitude of the ssh
extremum, in km:

| day | 1 | 10 | 30 | 60 | 100 |
|---|---:|---:|---:|---:|---:|
| flux | 0 | −30 | −90 | −120 | −240 |
| vector | 0 | −30 | −60 | −90 | −150 |

Two NEMO-side observations, recorded and **not** interpreted here: the flux
card has lost two thirds of its peak velocity by day 100 where the vector card
has lost only a sixth, and the flux card's vortex has travelled 240 km west
against the vector card's 150 km. Both are NEMO-vs-NEMO differences between
two momentum programs on the same topography; nothing in this round compares
either against legoESM.

### Does the partial-step run differ from the flat run, and where?

Prediction P4 said yes, "because partial steps change e3t near the seamount
from step 1". **Measured, that wording is wrong and is retracted.**

| | flux card | vector card |
|---|---|---|
| kt=1 (the INITIAL state) max \|ΔT\| | 5.707767 K | 5.707767 K |
| … on the 66 cells the seamount dries | 5.707767 K | 5.707767 K |
| … on cells wet in BOTH runs | 2.79e−05 K | 2.79e−05 K |
| kt=1 max \|Δssh\| | **0.0, identical** | **0.0, identical** |
| kt=2 (after ONE step) max \|Δssh\| | 2.2980e−03 m | 2.2997e−03 m |
| kt=2 max \|Δu\| / \|Δv\| | 5.871e−04 / 6.884e−04 | 6.146e−04 / 7.002e−04 |

At kt=1 the O(1) difference is **entirely the land mask**: the set of cells
where the seamount run has `T = 0` and the flat run does not is *exactly* the
set where the two `tmask` fields differ (66 cells; set equality tested, not
eyeballed). `ssh` is bit-identical. So the partial cells have not acted yet —
nothing has been differenced.

They act at kt=2, and the first thing that moves is `ssh`, by 2.3e−03 m, with
its maximum 268 km from the summit. That is the expected shape: the
topography is under the vortex's *path*, not under the vortex, so one step of
the free surface over the new bottom is what shows first.

**ONE FINDING, carried OPEN.** The initial states are *not* bit-identical on
cells wet in both runs: T differs by up to 2.79e−05 K over 3023 cells in 330
columns, spread through the vortex's own anomaly disc and reaching 532 km from
the summit, including columns whose `k_bot` is still 10. Nothing in the
initial state should see the bottom — `usr_def_istate` reads only `gdept`,
which `key_vco_1d3d` leaves 1-D. 2.8e−05 K is 8 mm of depth at this
stratification, far above fp64 rounding. **PLAUSIBLE, not confirmed:** the
depths handed to `istate` are rebuilt from the 3-D scale factors on the
`vco_1d3d` path rather than taken from the 1-D reference. The discriminating
check is cheap and is deferred to round 2: dump `gdept` from both builds and
diff it. Until then no round may treat the two cards' initial states as the
same state.

---

## 6. What is NOT in this round

No legoESM card, no legoESM physics or configuration change, no ladder, no
registry, no owner walk, no comparison of anything against legoESM. Both
VORTEX flat cards, GYRE, the tanks and DINO are untouched by construction:
nothing outside `scripts/validate/ocean_fidelity/testcases/` and `docs/` is
edited, and the four NEMO configurations are new directories beside the
certified ones, which were not read except to copy their instruments.

## 7. OPEN

1. **Decision 89** (above, §0): `key_vco_1d` → `key_vco_1d3d` on the two SMT
   builds. Forced by `domzgr.F90:259`; pending with the user; nothing lands as
   a legoESM statement from this round either way.
2. The initial-state difference on commonly-wet cells (§5). Diff `gdept`
   between the two builds; one cheap run.
3. **Round 2** — the two legoESM cards, `VORTEX_SMT-zps` and
   `VORTEX_SMT_VEC-zps`, with the bathymetry as an explicit field recipe (the
   same Gaussian, the same five constants, printed) and partial-cell geometry
   built from the card's own resolved mesh, every option explicit.
4. Then the ladders, the 50-row registries, the first-over-bar row and its
   owner walk in NEMO's stage order; expect new statements (partial-cell
   `e3f`, HPG over steps, bottom-level loops), each cited and landed under the
   gates.
5. Then the 100-day comparison, scored like round 210. The NEMO side of it is
   already acquired and admitted here.
6. `nn_stock = 30` remains instrumentation, as in round 210.
