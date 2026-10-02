# NEMO testcase fidelity: round 199 / VORTEX round 15 — NEMO's two continuity solves, landed on the vector card

**Status: LANDED.**  Decision 85 (user, operator note BT) authorised landing
round 194's two-solve WZV candidate on `VORTEX_VEC-zco` over the unchanged
two-ULP ratchet, with every moved row registered.  This round applies it as
two explicit card lines, re-certifies the vector card's 50-row registry as
the new reference, shows the flux card inert and GYRE byte-identical, and
retires the held patch manifest as landed.

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round199/`.
Lane tip this round starts from: `413984f3b` (round 198's landing).

**On preregistration.**  This is a landing round, not a walk.  Its
predictions are round 198's committed re-test numbers, measured before this
round existed and quoted back below as reproduction targets.

## The statement, in NEMO's compiled source

Inside one RK3 stage NEMO solves the continuity equation TWICE, and the two
calls are not the same statement.  In the vector-invariant arm
(`ln_dynadv_vec = .true.`, which the `VORTEX_VEC` deck selects) the momentum
program is handed the solve built from the RAW stage velocity:

* `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:289-300` — the
  `IF( ln_dynadv_vec )` block.  At `kstg /= 1` the vector arm calls
  `wzv( kstp, Kbb, Kmm, Kaa, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, np_velocity )`
  (`:293`), on the velocity, while the flux arm at `:300` calls the same
  routine on the transports `zFu, zFv` with `np_transport`.
* `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traadv.f90:268` — the tracer
  program then RE-SOLVES it, `wzv( kt, Kbb, Kmm, Kaa, pFu, pFv, ww,
  np_transport )`, and builds its own `pFw` from that second solve.
* `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90:236-311` — `wzv_RK3_t`,
  the solve itself (`div_hor` on the operands it is handed, then the
  vertical integral), which is why the operand choice changes the answer.

The flux card runs only the transport form, so the split does not exist
there; that is why it must be, and is, inert.

## The card lines

The two fields are STATED on the vector arm of the testcase recipe
(`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py`, the arm
selected by `whole_step_identity == "vortex_vector_een_c2"`), never inferred
from the time integrator or the momentum form (Decision 75):

```
wzv_call2_evaluation="nemo_literal",
nemo_stage_momentum_wzv_split=True,
```

`VORTEX-zco` is untouched and keeps the library values (`"generic"`,
`None`), which is what NEMO's flux arm resolves to.  The held manifest
`manifests/nemo_testcase_l1_vortex_round194_literal_wzv_held.patch` is
renamed `..._round194_literal_wzv_LANDED.patch` with its status header
rewritten — retired, not deleted.

## The vector registry, before and after, re-certified

Measured on this tip with the certified ladder
(`nemo_testcase_phase3_trajectory_gate.py --max-step 10
--continue-after-first`), before on the clean tip and after with the card
lines (`phase3/round199/traj_VORTEX_VEC-zco_{before,after}.json`; the full
50-row table is `phase3/round199/registry_before_after.md`).

**40 of 50 rows move; all 40 improve; none worsens.  Six rows change
status, every one DEBT to AT-BAR, and none crosses the other way.**

| row | `VORTEX_VEC-zco` before | after | factor |
|---|---:|---:|---:|
| kt2 u | `3.3693297862846805e-06` | `1.3044375150352006e-15` | 2.6e9 |
| kt2 v | `3.3370236514063123e-06` | `1.3356828663935172e-15` | 2.5e9 |
| kt2 ssh | `2.831068712794149e-15` | `2.831068712794149e-15` | unchanged |
| kt10 u | `4.85111821934675e-06` | `6.7335204832662586e-15` | 7.2e8 |
| kt10 v | `4.716724547932066e-06` | `6.604573001238271e-15` | 7.1e8 |

`first_over_bar` loses temperature and stays at `kt=2` on `u, v, ssh`:
`{'fields': ['T','u','v','ssh'], 'kt': 2}` -> `{'fields': ['u','v','ssh'],
'kt': 2}`.  The `kt=2` sea surface row does not move at all, because its
owner is the barotropic solve round 196/197 walked, not this statement.

### Every moved row, registered

| row | before | after | status |
|---|---:|---:|---|
| kt2.T | 3.6254894e-09 | 3.4661404e-16 | **DEBT -> AT-BAR** |
| kt2.u | 3.3693298e-06 | 1.3044375e-15 | DEBT |
| kt2.v | 3.3370237e-06 | 1.3356829e-15 | DEBT |
| kt3.T | 3.2611027e-08 | 6.9346162e-16 | **DEBT -> AT-BAR** |
| kt3.S | 6.0903663e-16 | 4.0602442e-16 | AT-BAR |
| kt3.u | 4.0100848e-06 | 2.0211968e-15 | DEBT |
| kt3.v | 3.9838919e-06 | 2.2204460e-15 | DEBT |
| kt3.ssh | 1.1778077e-07 | 2.6645353e-15 | DEBT |
| kt4.T | 1.1403733e-07 | 6.9380883e-16 | **DEBT -> AT-BAR** |
| kt4.u | 2.3865613e-06 | 3.0886616e-15 | DEBT |
| kt4.v | 2.3408030e-06 | 3.2481342e-15 | DEBT |
| kt4.ssh | 1.0245750e-07 | 2.8865799e-15 | DEBT |
| kt5.T | 1.6921264e-07 | 8.6777447e-16 | **DEBT -> AT-BAR** |
| kt5.u | 3.7174699e-06 | 3.8340760e-15 | DEBT |
| kt5.v | 3.5887628e-06 | 3.8758059e-15 | DEBT |
| kt5.ssh | 6.9030488e-07 | 2.7755576e-15 | DEBT |
| kt6.T | 1.7251205e-07 | 1.0419641e-15 | DEBT |
| kt6.S | 1.0150611e-15 | 8.1204884e-16 | **DEBT -> AT-BAR** |
| kt6.u | 4.7338094e-06 | 4.8608578e-15 | DEBT |
| kt6.v | 4.5937960e-06 | 4.4415833e-15 | DEBT |
| kt6.ssh | 9.7400508e-07 | 4.3853809e-15 | DEBT |
| kt7.T | 1.5821503e-07 | 1.0425569e-15 | DEBT |
| kt7.S | 1.0150611e-15 | 8.1204884e-16 | **DEBT -> AT-BAR** |
| kt7.u | 5.4398238e-06 | 5.2180482e-15 | DEBT |
| kt7.v | 5.3107247e-06 | 4.8099274e-15 | DEBT |
| kt7.ssh | 8.1675466e-07 | 2.6873306e-15 | DEBT |
| kt8.T | 1.4746070e-07 | 1.0430726e-15 | DEBT |
| kt8.u | 5.9104080e-06 | 7.0887340e-15 | DEBT |
| kt8.v | 5.8658458e-06 | 5.3431109e-15 | DEBT |
| kt8.ssh | 5.1115893e-07 | 2.7200464e-15 | DEBT |
| kt9.T | 1.4542248e-07 | 1.2174170e-15 | DEBT |
| kt9.S | 1.2180733e-15 | 1.0150611e-15 | DEBT |
| kt9.u | 5.5863877e-06 | 6.5377263e-15 | DEBT |
| kt9.v | 5.4865737e-06 | 6.4517892e-15 | DEBT |
| kt9.ssh | 3.4300831e-07 | 3.4972025e-15 | DEBT |
| kt10.T | 1.4955843e-07 | 1.3917916e-15 | DEBT |
| kt10.S | 1.2180733e-15 | 1.0150611e-15 | DEBT |
| kt10.u | 4.8511182e-06 | 6.7335205e-15 | DEBT |
| kt10.v | 4.7167245e-06 | 6.6045730e-15 | DEBT |
| kt10.ssh | 4.8797690e-07 | 3.1554620e-15 | DEBT |

The six status moves are `kt2`, `kt3`, `kt4`, `kt5` temperature and `kt6`,
`kt7` salinity, all DEBT -> AT-BAR.  They reproduce round 198's re-test
exactly (`phase3/round198/ulp_VORTEX_VEC-zco_wzv_arm.json`), which predicted
`kt2 u/v 1.3044375e-15 / 1.3356829e-15`, 40 moved rows and 6 status moves
before this round started.

## The ratchet's own numbers, red and registered

The cellwise two-ULP ratchet is UNCHANGED (limit 2 row-scale ULPs).
Decision 85 authorises landing over its red:

```
OFFLINE_ORACLE_RELATIVE_COMPARE FAIL: rows=50 max_worsening_ulps=6
  first_over_bar={'fields': ['T','u','v','ssh'], 'kt': 2}->{'fields': ['u','v','ssh'], 'kt': 2}   (VORTEX_VEC-zco)
OFFLINE_ORACLE_RELATIVE_COMPARE PASS: rows=50 max_worsening_ulps=0
  first_over_bar unchanged                                                  (VORTEX-zco)
```

| card | cells improved | cells worsened | max worsening, row-scale ULPs | largest move the statement makes | worst row |
|---|---:|---:|---:|---:|---|
| `VORTEX_VEC-zco` | 1,078,359 | 74,259 | `6.0` | `26618111479.5` | `kt10` T: 465 worse, 36308 better of 37210 |
| `VORTEX-zco` | 0 | 0 | `0.0` | `0.0` | none |

The largest move the statement makes is **4.4 billion times** the largest
worsening.  The worsening is six last bits on rows that are themselves at or
near the bar.

## The flux card is inert, and the inertness is not vacuous

`VORTEX-zco`: **0 of 50 rows move**, 0 cells move, ratchet PASS.  An
inertness claim measured by a gate that cannot go red would be worthless, so
the same comparison was re-run with the gate's own planted violations on the
same inert pair (`phase3/round199/ratchet_plants.txt`):

| plant | verdict |
|---|---|
| `worsen-3ulp` | `FAIL, max_worsening_ulps=3` |
| `at-bar-to-debt` | `FAIL` |
| `improve` (benign) | `PASS` |

## GYRE: byte-identical

GYRE already states this program — its card has carried
`wzv_call2_evaluation="nemo_literal"` and
`nemo_stage_momentum_wzv_split=True` since round 163 — so the vector VORTEX
arm cannot reach it.  Measured anyway, both ways:

* **The certified `kt=1..10` ladder**, re-run at this round's tree and
  compared row by row against round 191's certified ladder
  (`phase3/round191/gyre_ladder.json`): **0 of 50 rows differ**, status
  `DEBT` on both sides, `first_over_bar` unchanged at
  `{'fields': ['T', 'S', 'u', 'v', 'ssh'], 'kt': 3}` (`phase3/round199/gyre_ladder_after.json`).
* **The from-rest year.**  `nemo_testcase_l2_gyre_year_fromrest.py --member 0
  --days 360 --snap-steps 6 --tag r199`, compared file by file against the
  certified carried arm `phase3/year_fromrest/lego_seed0_r6carried`
  (`phase3/round199/gyre_year_byte_identity.txt`):

```
certified arm files: 360 | compared: 360 | differing: 0 | missing: 0
day030.npz certified 95f336ef7ef9e1e0 round199 95f336ef7ef9e1e0
day240.npz certified 018b75e12bc52968 round199 018b75e12bc52968
day360.npz certified 446a9041887a2c85 round199 446a9041887a2c85
```

  The three certified day numbers therefore stand unchanged to every printed
  digit: **day 30 `2.3432510206121264e-06`**, **day 240
  `6.581707093530567e-05`**, **day 360 `5.407735418221895e-05` K**.  They are
  carried by byte-identity of the snapshots they are computed from; the
  day-gap scorer was not re-run, and that is said here rather than implied.

## Tanks and the other cards

Both tanks re-run on this tree with the same 50-row ladder and compared
against the last certified tank measurement
(`phase3/merge_main_2026-09-29/after/`):

| card | rows differing | status | first_over_bar |
|---|---:|---|---|
| `LOCK_EXCHANGE-zco` | 0 of 50 | `DEBT` -> `DEBT` | `{'fields': ['u'], 'kt': 4}` -> `{'fields': ['u'], 'kt': 4}` |
| `OVERFLOW-zps` | 0 of 50 | `DEBT` -> `DEBT` | `{'fields': ['T', 'u'], 'kt': 2}` -> `{'fields': ['T', 'u'], 'kt': 2}` |

The ORCA2 and DINO cards cannot see this change: it is written inside the
arm selected by `whole_step_identity == "vortex_vector_een_c2"`, which only
`VORTEX_VEC-zco` resolves.  DINO is measured anyway by the month gate that
`land.sh` runs on every landing that touches model code.

## Gates and tests

* **Citation gate**: `status PASS`, `unmapped_citations []`, run on the
  gated receipt at this round's tree (`phase3/round199/citations_default_receipt.json`).
* **Citation re-anchor**: the card edit inserts eleven lines at line 169 of
  the testcase recipe module, so the old-to-new line map was built with
  `difflib` over the pre-round and post-round text and applied once: four
  pinned spans in the gate's map (`341,586,2034` -> `352,597,2045`;
  `558-645` -> `569-656`; `408-410` -> `419-421`; `626` -> `637`) and the one
  prose citation in the gated receipt, moved in the same commit.  The
  reviewer independently audited the whole map: 0 non-OK rows.
* **Focused card battery**: `74 passed`
  (`tests/ocean/unit/test_nemo_vortex_card.py`,
  `tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round198_e3f_0vor.py`,
  `tests/ocean/fidelity/test_fidelity_card_constructibility.py`;
  `phase3/round199/focused_pytest.log`).
* **Non-vacuity of the new pin**: with the card edit reverted and the test
  kept, `test_the_vector_card_states_nemos_two_continuity_solves` FAILS
  (`nemo_literal` expected, `generic` resolved;
  `phase3/round199/nonvacuity_test_reverted.log`).  The test also asserts the
  stepper's own predicate resolves the split as executing on the vector card
  and not executing on the flux card, so the stated line cannot go inert
  unnoticed.
* **Non-vacuity of the inertness gate**: the flux card's own before/after
  pair, which the gate calls clean, goes RED under both of the gate's
  planted violations and stays green under the benign one
  (`phase3/round199/ratchet_plants.txt`).
* **Generic NEMO-GYRE recipe gate and the lane's required battery** run at
  the push gate (`land.sh`), whose decisive line is recorded with the push.
* **DINO month gate** runs inside `land.sh` with a private work directory;
  its number is recorded with the push.

## Independent adversarial review (fresh reviewer, this round)

One fresh `code-reviewer` agent, given the diff and no other context.

**First verdict: DO NOT SHIP**, on five findings.  Two were refuted, one was
accepted and fixed, two were procedural and are now discharged:

| finding | response |
|---|---|
| "Decision 85 and note BT do not exist; the commit cites its own authorisation" | **REFUTED, and the reviewer retracted it.**  Both live outside the git tree, in the campaign ledger and the standing brief (user, 2026-10-02 07:15; "Decision 82 is thereby superseded").  The reviewer had grepped only `docs/`. |
| "landing over a red ratchet is a concealed violation" | **REFUTED, retracted.**  The red is the authorised term of Decision 85 and every moved row is registered here. |
| the card comment cited the tracer re-solve at `traadv.f90:274`, a blank line | **ACCEPTED and FIXED** — the call is at `:268`, as this receipt says.  Comments are outside the citation gate's reach, so nothing would have caught it. |
| "the test only restates the constructor" | **ACCEPTED in part.**  It is the Decision 75 statement pin and the registry is the behavioural evidence; the reviewer's suggested executed pin was added. |
| "receipt has unfilled placeholders; GYRE and OVERFLOW unmeasured" | **DISCHARGED** — the reviewer read the tree mid-round; both measurements are above and this receipt is complete. |

**Re-verdict after the retractions**: every number in this receipt
reproduces exactly from `phase3/round199/`; scope, override-freedom and all
four citation re-anchors verified clean.

## Choices made this round

| choice | ASKED / UNASKED | where |
|---|---|---|
| `VORTEX_VEC-zco` states `nemo_stage_momentum_wzv_split=True` and `wzv_call2_evaluation="nemo_literal"` | **ASKED** | Decision 85 (user), note BT |
| landing over the two-ULP ratchet's red at 6 ULPs | **ASKED** | Decision 85 |
| the ratchet limit itself is unchanged | **ASKED** | Decision 82/85 |
| the flux card is left on the library values NEMO's flux arm resolves to | **ASKED** | note BT: "nothing else changes" |

UNASKED list: **empty**.

## Landing verdict: LANDED

The statement is NEMO's, cited in the compiled source of the card's own
build, and it is STATED on the card rather than inherited.  On the vector
card 40 of 50 registry rows move, every one toward NEMO, six of them from
DEBT to AT-BAR, with the `kt=2` velocities falling by 2.6 billion to the
bar's own order; nothing moves away at row level.  The flux card is exactly
inert, GYRE is byte-identical over the certified ladder and all 360 days of
the from-rest year, and the tanks are unchanged.  The two-ULP cellwise
ratchet is red at six ULPs, which Decision 85 authorises and this receipt
registers.  **LANDED.**

## OPEN, in the order note BT sets

1. **The flux card's own `kt=2` owner.**  `VORTEX-zco` `kt=2` u/v
   `1.1355340046037554e-07` / `1.1354649764871994e-07`, unmoved by rounds
   198 and 199 — a stage-3 residual with no named statement since round 4.
2. **`fmask` versus `fe3mask` in the `r3f` factor.**  Inert only at
   `rn_shlat = 0`; it must be cited and tested at the cards' own `rn_shlat`.
3. **Decision 74's resolution ladder** (30 / 15 / 10 km, both cards).  The
   vector card's `kt=2` velocities are now at the bar's own order, so the
   remaining precondition is item 1.
