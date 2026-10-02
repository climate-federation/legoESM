# NEMO testcase fidelity: round 198 / VORTEX round 14 — the barotropic Coriolis thickness, landed in the shared builder

**Status: LANDED.**  Decision 84 (user) authorised landing round 197's cited
statement over the unchanged two-ULP ratchet's red, with every moved row
registered.  This round lands it in the SHARED split-explicit coefficient
builder — not as a card edit — re-certifies both VORTEX cards' 50-row
registries at the new values, and shows the certified GYRE card is
byte-identical for the reason NEMO's own ENE branch gives.

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round198/`.
Decision 82 (user): round 194's two-solve WZV candidate stays HELD; it is
re-tested here as a measurement arm only.

**On preregistration.** This is a landing round, not a walk, so its
predictions are not new: they are round 197's committed numbers, which
pre-date every measurement here and are quoted back as reproduction targets
in the verdict table below.  No prediction was written after the fact.

## Compiled program, and why the fix is shared

Inside one sub-time-step of the split-explicit solve NEMO applies eight
frozen 2-D Coriolis coefficients
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:503` calls the
routine; the coefficients are built once in
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:933`).  That builder
has three arms, and **every one of them divides by `e3f_0vor`, not by the
F-point reference thickness `e3f_0`**:

| NEMO branch | selected by | the statement |
|---|---|---|
| EEN | `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:956` | `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:960` |
| ENE and MIX | `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1012` | `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1016` |
| ENS | `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1042` | `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1046` |

`dyn_vor_init` allocates and freezes that array for all four curl-point
schemes in ONE `SELECT CASE` arm
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:890`).  At the decks'
resolved `nn_e3f_typ = 0` it is the four surrounding T cells' MASKED
reference thickness divided by FOUR — by four, not by the number of wet
cells, which is the `nn_e3f_typ = 1` branch
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:897` against
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:905`) — and the
"insure e3f_0vor /= 0" sweep then restores the card's own reference
thickness at a fully dry vertex
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:920`).

Because the statement is shared across NEMO's four schemes, the fix is
shared too.  It is **not** round 197's held card edit.  The new
`nemo_e3f_0vor_from_tmask` transcribes `dyn_vor_init`'s two `nn_e3f_typ`
branches once — NEMO's `((N + NE) + (C + E))` bracketing, the boundary
condition, then the restore — and the one literal coefficient builder calls
it for EVERY card, with that card's own `e3f_0` as the restore operand.  The
builder now RAISES if a card cannot supply the mesh operands; there is no
silent fallback to the plain thickness.

## The ENE proof: why GYRE does not move

GYRE's deck selects `ln_dynvor_ene = .true.`
(`GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:167`), so GYRE runs the ENE arm, and
that arm DOES consume `e3f_0vor`
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1016`).  The reason
GYRE's coefficients are nevertheless unchanged is in the pairing that
statement writes: in the ENE arm each `e3f_0vor` index is divided into a
term whose only mask comes from the SAME F row — `e3f_0vor(ji,jj)` with
`vmask(ji,jj)` and `vmask(ji+1,jj)`, `e3f_0vor(ji,jj-1)` with
`vmask(ji,jj-1)` and `vmask(ji+1,jj-1)` — and the host u point contributes
`r1_hu_0(ji,jj)`, which is zero on a dry face.  On a rectangular closed box
every vertex whose thickness moves has a dry corner that forces one of those
masks to zero, so the moved thickness is multiplied by nothing.  The EEN arm
has no such pairing: its `zpvo` triads sum three different F points
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:960`), so a wall
vertex reaches interior faces.

That argument is READ OFF THE CODE.  It is MEASURED by a committed probe,
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_round198_e3f_0vor_scope.py`
(`phase3/round198/e3f_0vor_scope.json`), which compares the eight frozen
coefficients built from `e3f_0vor` against the eight built from the plain
thickness, at each card's own geometry, and REFUSES to report if no card
moves:

| card | branch | vertices whose `e3f_0vor` differs | the eight coefficients | largest change |
|---|---|---:|---|---:|
| `GYRE-zco` | ENE | 3000 of 21120 | **bit-identical** | `0.0` |
| `VORTEX-zco` | EEN | 2440 of 39690 | change | `1.7818667402025312e-05` |
| `VORTEX_VEC-zco` | EEN | 2440 of 39690 | change | `1.7818667402025312e-05` |

Both VORTEX decks select `ln_dynvor_een = .true.`; they differ only in
momentum advection form, so "the flux card" is flux-form UBS advection with
EEN vorticity, not an ENS card.

## The certified registries, before and after

Both arms re-measured at this round's own commits, same oracle, same gate,
one variable; the BEFORE arms were taken with the production files restored
to the lane tip `d67bb835a` and reproduce round 197's reference exactly
(0 of 50 rows differ, both cards).  Full 50-row tables:
`phase3/round198/registry_before_after.md`; artefacts
`phase3/round198/traj_*_{before,after}.json`.

| row | `VORTEX_VEC-zco` before | after | `VORTEX-zco` before | after |
|---|---:|---:|---:|---:|
| kt2 u | `3.369330e-06` | `3.369330e-06` | `1.135534e-07` | `1.135534e-07` |
| kt2 v | `3.337024e-06` | `3.337024e-06` | `1.135465e-07` | `1.135465e-07` |
| kt2 ssh | `3.709010e-08` | `2.831069e-15` | `3.708794e-08` | `2.664535e-15` |
| kt10 u | `4.865476e-06` | `4.851118e-06` | `2.464017e-06` | `4.888403e-07` |
| kt10 v | `4.724707e-06` | `4.716725e-06` | `2.191673e-06` | `4.859896e-07` |
| kt10 ssh | `5.356565e-06` | `4.879769e-07` | `5.336426e-06` | `1.899224e-07` |

**The ledger, not the selection.**  On the vector card 34 of 50 rows move:
23 improve and 11 worsen, and NO row changes status.  On the flux card 39
move: 33 improve and 6 worsen, and TWO rows change status — the flux card's
`kt6` salinity crosses AT-BAR to DEBT by one quantum
(`8.120488408686856e-16` to `1.015061051085857e-15`) and its `kt7` salinity
crosses back the other way (`1.0150610510858566e-15` to
`8.120488408686853e-16`).  `first_over_bar` stays at `kt=2` on both cards
and every `kt=1` row is unchanged.  The vector card's `kt=2` velocity rows
do not move at all, because their owner is the stage vertical velocity
round 194 named and this statement does not touch it.

## The ratchet's own numbers, red and registered

Decision 84 authorises landing over this, and the ratchet itself is
UNCHANGED (limit 2 row-scale ULPs, Decision 82).  Both reports reproduce
round 197's to the digit (`phase3/round198/ulp_*.json`):

```
OFFLINE_ORACLE_RELATIVE_COMPARE FAIL: rows=50 max_worsening_ulps=2132967438.5
  first_over_bar={'fields': ['T','u','v','ssh'], 'kt': 2}->{'kt': 2, ...}   (VORTEX_VEC-zco)
OFFLINE_ORACLE_RELATIVE_COMPARE FAIL: rows=50 max_worsening_ulps=540700912.625
  first_over_bar={'fields': ['T','u','v','ssh'], 'kt': 2}->{'kt': 2, ...}   (VORTEX-zco)
```

| card | cells improved | cells worsened | max worsening, row-scale ULPs | largest move the fix makes | worst row |
|---|---:|---:|---:|---:|---|
| `VORTEX_VEC-zco` | 1,015,156 | 143,924 | `2132967438.5` | `37070406141.19629` | `kt10` ssh: 106 worse, 3615 better of 3721 |
| `VORTEX-zco` | 1,045,802 | 112,378 | `540700912.625` | `37011552618.32715` | `kt8` v: 382 worse, 36218 better of 36600 |

The largest move the statement makes is more than an order of magnitude
larger than the largest worsening, on both cards.

## Round 194's held WZV candidate, re-tested on the NEW reference

Decision 82 keeps it HELD and nothing about it is landed here.  It is
applied to a scratch arm on top of this round's reference, measured, and
reverted (`git status --porcelain` clean afterwards); artefacts
`phase3/round198/traj_*_wzv_arm.json` and `phase3/round198/ulp_*_wzv_arm.json`.

**The trade that held it has disappeared.**  On the old reference the
candidate cost `kt5` salinity its AT-BAR status and the ratchet read in the
billions.  On the new reference it costs nothing and the ratchet reads
**six** ULPs.

| `VORTEX_VEC-zco`, this round's reference | reference | + held WZV candidate |
|---|---:|---:|
| kt2 u | `3.3693298e-06` | `1.3044375e-15` |
| kt2 v | `3.3370237e-06` | `1.3356829e-15` |
| kt10 u | `4.8511182e-06` | `6.7335205e-15` |
| kt10 v | `4.7167245e-06` | `6.6045730e-15` |
| rows moved of 50 | — | 40 |
| rows changing status | — | 6, ALL toward AT-BAR |
| `first_over_bar` | `T, u, v, ssh` at kt=2 | `u, v, ssh` at kt=2 |
| two-ULP ratchet | — | `FAIL, max_worsening_ulps=6` |

The six status moves are `kt2`, `kt3`, `kt4` and `kt5` temperature and
`kt6` and `kt7` salinity, every one DEBT to AT-BAR; the `kt6`/`kt7`
salinity pair this round's landing moved is moved straight back.  On the
flux card the candidate is exactly inert (0 of 50 rows move, ratchet PASS),
which is correct — it is a vector-card statement.

**This is a FINDING, not a landing.**  The candidate remains held under
Decision 82, and it is now a far better trade than when it was held: a
six-ULP ratchet red against four temperature rows and two salinity rows
returning to the bar, and the vector card's kt=2 velocities falling by
2,600x to the bar's own order.  It needs its own decision.

## GYRE: byte-identical, measured two ways

* **The certified `kt=1..10` ladder.**  Re-run at this round's commit and
  compared row by row against round 191's certified ladder
  (`phase3/round191/gyre_ladder.json`): **0 of 50 rows differ**, status
  `DEBT` on both sides, `first_over_bar` unchanged at `kt=3`
  (`phase3/round198/gyre_ladder_after.json`).
* **The from-rest year.**  `nemo_testcase_l2_gyre_year_fromrest.py --member 0
  --days 360 --snap-steps 6 --tag r198`, compared file by file against the
  certified carried arm `phase3/year_fromrest/lego_seed0_r6carried`:
  **all 360 daily snapshots byte-identical, 0 differing**.  The three
  certified day digests are equal on both sides —
  `day030` `95f336ef7ef9e1e0…`, `day240` `018b75e12bc52968…`,
  `day360` `446a9041887a2c85…` — so the certified numbers
  `2.3432510206121264e-06`, `6.581707093530567e-05` and
  `5.407735418221895e-05` K stand unchanged to every printed digit.  They are
  carried by byte-identity of the snapshots they are computed from, which is
  a stronger statement than re-scoring; the day-gap scorer was not re-run,
  and that is said here rather than implied.

## Predictions, against round 197's committed numbers

| # | reproduction target, from round 197 | verdict |
|---|---|---|
| P1 | the shared builder reproduces the held card patch exactly on both VORTEX cards | **CONFIRMED** — 0 of 50 rows differ from round 197's `after` arm, both cards |
| P2 | the BEFORE arms reproduce the certified reference at the lane tip | **CONFIRMED** — 0 of 50 rows differ, both cards |
| P3 | the ratchet reads `2132967438.5` and `540700912.625` | **CONFIRMED** to the digit |
| P4 | GYRE is byte-identical | **CONFIRMED** — coefficients bit-identical, ladder 0 of 50, year 360 of 360 |
| P5 | the flux card's `kt6` salinity crosses AT-BAR to DEBT and `kt7` crosses back | **CONFIRMED**, both quanta |
| P6 | the vector card's `kt=2` velocity rows do not move | **CONFIRMED** |

## Gates and tests

* **Citation gate**: `status PASS`, `citations_found 274`,
  `unmapped_citations []`, `map_entries_failing_audit 0`, and all **9**
  planted-shift self-tests fired (`phase3/round198/citations.json`).
* **Citation re-anchor**: the helper adds sixty lines above two cited
  definitions in the vertical module and the coefficient edit adds twenty
  above one in the barotropic module, so **five receipt citations and three
  pinned map anchors** were moved by rigid shift and every shifted endpoint
  re-verified to land on the same definition.  None weakened, none removed.
* **Round-198 unit module**: `6 passed`
  (`tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round198_e3f_0vor.py`),
  with the certified-card pin run as an in-process old-against-new
  comparison.  **Non-vacuity**: with the production change reverted all six
  arms fail (`phase3/round198/nonvacuity_tests_reverted.txt`); five of those
  failures are an ImportError, which is a weak control, and the real one is
  the in-process comparison, which cannot pass if the statement is inert.
* **Barotropic null-mode module**: `26 passed` together with the above, after
  its two literal-builder tests were given the mesh the builder now requires.
* **Source-rounding change is bit-inert here**: both VORTEX registries 0 of
  50 rows moved and GYRE's eight coefficients stayed bit-identical after it.

## Independent adversarial review (fresh reviewer, this round)

Codex is reserved for the ORCA2 lanes, so the mandatory second opinion was a
fresh reviewer agent given the diff, the compiled sources and the evidence
root with no knowledge of how any of it was produced.  Verdict as delivered:
**DO-NOT-SHIP, with three cheap fixes named**.  All three are fixed and the
verdict's conditions are met.

* **DEFECT (accepted, fixed).**  The new docstring had the fully-dry-vertex
  restore operand backwards; `dynvor.F90:945-951` restores `e3f_0` in both
  arms and it is the substitution header that expands it per build.
* **DEFECT (accepted, fixed).**  Two committed unit tests built the operand
  bundle with no mesh and so hit the new fail-closed raise.  They now hand
  the builder the mesh and use its own frozen array.
* **DEFECT (accepted, fixed).**  Three scripts called the shared builder
  without the card's grid, silently skipping the north fold; the ORCA2
  external gate is tripolar and its own comment claims it runs the
  production builder, so it was diverging on the folded row.  All three now
  pass the grid (the DINO probe's bridge names it `geometry`).
* **DEFECT (accepted, fixed).**  The helper used a bare
  `optimization_barrier` where this module documents that only the shared
  source-rounding identity survives optimised HLO.
* **GAP (accepted, reported, NOT fixed here).**  The coefficient builder
  applies the live `r3f` through `fmask` where NEMO uses `fe3mask` in all
  three branches.  It is equal on every card measured here because they
  resolve `rn_shlat = 0`; it is pre-existing, it is not this statement, and
  it is carried to OPEN rather than bundled into a landing round.
* **GAP (accepted, corrected above).**  The reviewer is right that "a mask
  from the same F row" does not by itself cover `ffu_ne`, which pairs
  `e3f_0vor(ji,jj)` with `vmask(ji+1,jj)`; what zeroes that term is the dry
  U point's `r1_hu_0`.  The ENE section names both, and the identity was
  measured at `eta = 0` only.
* The reviewer independently CONFIRMED: the `nn_e3f_typ = 0` transcription is
  exact in cells, mask, divisor, bracketing and the fold-then-restore order;
  `raw.e3f_0` is the right restore operand for every card; the placement is
  shared with no per-card switch; the probe's branch resolution is identical
  to production and its non-vacuity control is real (GYRE's 3000 of 21120 is
  the arithmetically right count for a 22x32 closed box over 30 levels); and
  the scope JSON's recorded `diff_sha256` is reproducibly the hash of the
  probe commit that produced it.
* **Minor, recorded not fixed**: the `nn_e3f_typ = 1` branch is unreachable
  from production (both call sites pin 0) and is exercised only by the new
  test; and the probe's branch-resolution expression carries the same dead
  clause production carries.

## Option choices made this round

| choice | ASKED / UNASKED | note |
|---|---|---|
| landing the `e3f_0vor` statement over the ratchet's red | ASKED | Decision 84 |
| the fix lives in the shared builder, every card, no per-card switch | ASKED | Decision 84's wording |
| `nn_e3f_typ = 0` | n/a | resolved by the decks' own `namelist_ref:1072`, not chosen here |
| the fully-dry-vertex restore operand is each card's own `e3f_0` | n/a | NEMO's own operand, cited |
| the two-solve WZV candidate stays HELD | ASKED | Decision 82 |
| the two-ULP ratchet is unchanged | ASKED | Decision 82 |
| the `fmask`/`fe3mask` gap is reported, not fixed | n/a | pre-existing, not this statement |

Nothing on the UNASKED list.

## Landing verdict: LANDED

## OPEN — round 199

1. **The held two-solve WZV candidate is now a different trade and needs a
   DECISION.** On this round's reference it costs six row-scale ULPs and
   returns six certified rows to the bar while taking the vector card's
   `kt=2` velocities from `3.37e-06` to `1.30e-15`.  It stays HELD under
   Decision 82; the measurement is above.
2. **The flux card's own `kt=2` owner.**  Its `kt=2` velocity rows are
   unchanged at `1.135534e-07` / `1.135465e-07`; the vertical-profile
   momentum residual named in round 4 is still unattributed.
3. **`fmask` where NEMO uses `fe3mask`** in the frozen coefficient's live
   `r3f` factor, in all three branches.  Inert on every `rn_shlat = 0` card
   and unmeasured elsewhere.
4. **Decision 74's 30/15/10-km ladder**, once the vector card's `kt=2` rows
   are at the bar or proven harness floor — which is now item 1's decision.
