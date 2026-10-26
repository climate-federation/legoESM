# PR #1802 — decision 72, and the DINO year re-run

**Status: D72 LANDED.  THE DINO FROM-REST YEAR RE-RAN AND COMPLETED on the
card the certification actually uses.**

Decision 72 is done and proven by value on the three columns it is about.
The DINO year certification was re-run; on its own card, the MLF one, it
completes all 360 days at this tip.  A first attempt used the Euler card
through a different driver, hit that card's already-registered fifth-step
instability, and its "the year cannot run" conclusion is RETRACTED below.
This round's change is measured bit-inert on both DINO cards, so no DINO
number here is attributable to it.

Nothing in the certified set moved.  GYRE's ladder, all 360 of its daily year
snapshots and both tank ladders are byte-identical to the previous round's
certified arm.

* Before arm: the lane tip `24f8f7e75`.
* After arm: `4f6a55768`, this round's third commit.
* Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/pr1802_dino_year/`.

---

## Decision 72 — the ORCA1 OMIP card keeps main's calm-column floor

**What NEMO does, cited.**  `tke_avn` floors the `ln_mxl0` wind anchor at
`rn_mxl0` — `zmxlm(ji,1) = MAX( rn_mxl0, zmxlm(ji,1) )`, compiled GYRE branch
`GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:610` — so on a CALM column the
anchor IS that value.  `zdf_tke_init` has already OVERWRITTEN the namelist
`rn_mxl0` with the active mixing-length floor `rmxl_min`, because `ln_mxl0` is
true: the whole guarded block is
`GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:828-831` (shipped
src/OCE/ZDF/zdftke.F90:859-862), whose last statement is
`rn_mxl0 = rmxl_min`.  On ORCA1 that floor is `1.0e-3 m`, because `ln_zdfiwm`
forces `rmxl_min = 1.e-03_wp` at
`GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:810-812` (shipped
src/OCE/ZDF/zdftke.F90:841-843).

**So NEMO's own value on this card is 1.0e-3 m, and the user's decision is
that Pierre's card does not take it.**  It keeps the namelist `rn_mxl0` that
GitHub `main` gave it, `0.04 m`.

**What changed.**  `TKEConfig` regains `mxl0_min_m` (NEMO's `rn_mxl0`, default
`0.04`), which rounds 55/56 had removed when they made the anchor read the
mixing-length floor unconditionally, and gains
`nemo_mxl0_rmxl_min_overwrite` (default `False` = main's behaviour) which
selects NEMO's overwrite.  The NEMO-literal cards select it: GYRE and
everything built on `_nemo_tke_config`, the ORCA2-zps testcase card that
specialises that config, and both NEMO DINO cards.  The ORCA1 card states both
its value and its arm explicitly rather than leaning on the default.  The
zero-step closure diagnostic follows the same selector instead of
reconstructing a floor of its own.

**The proof, by value.**  The ORCA1 card's TKE config was resolved from its own
builder on three trees, and the anchor evaluated on three columns — windy wet,
calm wet, and land — using the previous round's own fingerprint probe
(`phase3/pr1802_dino_year/orca1_tke_fingerprint.py`; outputs `fp_main.json`,
`fp_before_tip.json`, `fp_after_tip.json`).

| ORCA1 anchor [m] | GitHub `main` `d7109d9b0` | lane before `24f8f7e75` | lane after `4f6a55768` |
|---|---|---|---|
| wet, stress 0.10 Pa (both arms) | `0.7951003609964353` | `0.7951003609964353` | `0.7951003609964353` |
| LAND, stress 0.07 Pa (both arms) | `0.5565702526975047` | `0.5565702526975047` | `0.5565702526975047` |
| CALM, stress 0, `ln_zdfiwm` off | `0.04` | `1e-08` | `0.04` |
| CALM, stress 0, `ln_zdfiwm` on | `0.04` | `0.001` | `0.04` |

The calm column is restored exactly, and the windy and land columns — the ones
decision 66 settled — are untouched.  On the `ln_zdfiwm` OFF arm the whole
mixing-length and diffusivity column now hashes identically to `main`
(`l_k` `6b3f788a80ed203e`, `K_M` `52f696c556ecee21`).  On the ON arm it does
not, and that is a DIFFERENT, already-decided item, reported here so it is not
read as this round's: `main`'s ORCA1 card does not set NEMO's forced
`rn_emin`/`rmxl_min` pair at all (it resolves `1e-6`/`1e-8`), while the lane
sets `1e-10`/`1e-3` from `GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:810-812`.  That is the 2026-08-27
equatorial-undercurrent fix and PR #1749; decision 72 names the surface
anchor's floor, not the interior mixing-length floor, and only the former was
moved back.

**Fails when reverted: YES, two ways.**
`tests/ocean/unit/test_nemo_card_opt_in_defaults.py::test_orca1_card_keeps_mains_rn_mxl0_surface_floor`
fails when the card selects the overwrite (`2 failed, 16 passed`) and again
when the library default is flipped (`2 failed, 16 passed`); logs
`plant_card.txt` and `plant_default.txt`.  Its companion
`::test_nemo_literal_cards_take_the_rn_mxl0_overwrite` pins the other half —
GYRE, ORCA2 and both DINO cards keep NEMO's value, and on GYRE that value
(`1.0e-2 m`) is asserted to DIFFER from `mxl0_min_m`, so the row cannot pass by
coincidence.

---

## The DINO year certification — re-run, refused, and diagnosed

**RETRACTION, kept loud.**  An earlier draft of this section said the DINO
from-rest year "cannot complete on either tree".  That is WRONG, and the
coordinator caught it.  It ran the wrong card through the wrong driver.  The
certified from-rest year artifact
(`/data/abyssal/dbalwada/dino_fromrest_y1/verdict360_fromrest/phase0_floor.json`)
names its own producer: `run_dino.py` with the deck
`scripts/experiment/dino/nemo_faithful_kamm_mlf.yaml` and the recipe
`nemo_dino_kamm_mlf` — the MLF card, not the Euler card — recorded in that
run's `run_metadata.json`.  The first attempt used
`dino_year_screen_fullframe.py` with `nemo_dino_kamm`, the EULER card, which
is a different card and a different driver.

**What the Euler card's failure actually was, and why it is not news.**  On
`nemo_dino_kamm` the operand the geometry guard rejects is finite and positive
at all six of its call sites through step 5; the model's own state goes
non-finite at the OUTPUT of step 5 (T 268 cells, S 268, u 334, v 333), with the
peak current running 0.034 → 0.050 → 0.062 → 1.126 → 4.737e24 m/s; the guard
fires at step 6 as a downstream detector.  Round 184 had already registered
exactly this — "the `nemo_dino_kamm` Euler card's shared fifth-step
implicit-solve instability" — and had already recorded the MLF card as stable.
So the measurement reproduced a known, owned defect on a card the year
certification does not use.  The guard is behaving correctly and was not
loosened.  Evidence: `logs/dino_fromrest_trace.log`.  For completeness, the
same Euler screen on GitHub `main` reaches day 30 entirely non-finite without
stopping (`logs/dino_main_fromrest.log`), so that card's instability is not
this branch's either.

**The certified protocol, re-run on the card it actually uses.**  `run_dino.py
--config scripts/experiment/dino/nemo_faithful_kamm_mlf.yaml --grid latlon
--recipe nemo_dino_kamm_mlf --n-lon 50 --nemo-faithful-grid --allow-multiyear
--days 360 --snapshot-every-days 30 --dt 2700`, on this round's final tip.
**It completes all 11,520 steps and writes all thirteen snapshots.**  Scored by
`twin_nemo_ts_maps.py --run-dino-dir` against NEMO's own from-rest year, the
same comparator and the same NEMO records the pinned artifact used
(`RUN_TRAJ` kt 960 for day 30, `RUN_FROMREST_Y1` kt 11520 for day 360).

| wet 3-D temperature rms vs NEMO [K] | pinned artifact | this tip |
|---|---|---|
| day 30 | `2.039e-03` | `6.982e-03` |
| day 360 | `3.924e-03` | `7.590e-03` |

**Those two columns are NOT a controlled pair, and the difference is NOT
attributable to this round.**  The pinned artifact was produced at PR #1728,
many lane commits ago, by a tree this round did not run; the year receipt
itself already records a later re-measurement of day 30 on the fixed card at
`6.889e-04` K, so the pinned day-30 number had already moved twice before this
round began.  What IS attributable is measured directly and by value: **this
round's change is bit-inert on both DINO cards.**  Both select NEMO's overwrite
(`tke_nemo_mxl0_rmxl_min_overwrite=True`), so their anchor floor resolves to
`0.009999999999999998` m — exactly the value the unconditional pre-change code
computed, since it equals `_mixing_length_floor` on those cards to the last
bit.  No DINO trajectory can move, and no before-arm year run is needed to say
so.

**CONFIRMED / PLAUSIBLE, kept separate.**  CONFIRMED: the certified card runs
the full from-rest year at this tip; the Euler card's state is non-finite at
the output of step 5; this round's change resolves to the identical anchor
floor on both DINO cards.  PLAUSIBLE, not measured: that the day-30 and
day-360 differences from the pinned table are owned by intervening lane work
rather than by anything in this PR — nothing here was run at the pinned tree's
commit, so that is an inference, and the follow-up is a controlled before/after
year pair if anyone wants the attribution.

---

## The certified numbers, this tip against the previous round's certified arm

### GYRE — the 70-row ladder

`nemo_testcase_l2_gyre_phase3_gate.py --trajectory-only --max-step 10`.

The report is equal FIELD FOR FIELD to the previous round's, excluding only the
worktree path: same 50 trajectory rows, same 70-row residual map, same
`DEBT` status, same first-over-bar `{T,S,u,v,ssh}` at kt 3 and barotropic
`{uu_b,vv_b}` at kt 2, and the residual array file has the same SHA-256
`5269048ef189d113385991c6d52c903ec519ad8796cad43fc559002173a1574e`.

| | previous round's certified arm | this tip |
|---|---|---|
| report digest, worktree stripped | `7ba15556de2de841` | `7ba15556de2de841` |

### GYRE — the from-rest year

`nemo_testcase_l2_gyre_year_fromrest.py --member 0 --days 360 --snap-steps 6
--tag year`, scored by `nemo_testcase_l2_gyre_year_owners.py --day-gap` on the
eight certified checkpoints.

**All 360 daily snapshots are byte-identical** to the previous round's
certified arm by whole-file comparison (360 identical, 0 differing).

| day | certified | this tip |
|---|---|---|
| 30 | `2.3276772050683987e-06` K | `2.3276772050683987e-06` K |
| 240 | `6.586171881479517e-05` K | `6.586171881479517e-05` K |
| 360 | `0.002670992385329469` K | `0.002670992385329469` K |

### The tanks

`nemo_testcase_phase3_trajectory_gate.py --case <case> --max-step 10
--continue-after-first`.  Both reports are equal field for field to the
previous round's, excluding the worktree path, the git stamp and the residual
artifact block (this round did not ask for residual files).

| case | certified | this tip |
|---|---|---|
| LOCK_EXCHANGE-zco | DEBT, first over bar `{u}` at kt 4 | identical, 50 of 50 rows equal |
| OVERFLOW-zps | DEBT, first over bar `{T,u}` at kt 2 | identical, 50 of 50 rows equal |

### DINO — the 90-day developed twin

`kamm_twin_90d.py nemo_dino_kamm_mlf --days 90 --save-3d --bridge-before`,
scored by `acceptance_gate_90d.py --run-recipe nemo_dino_kamm_mlf`.
Run to completion on a FROZEN tree at `e26971fa8` (the harness refuses to save
if the checkout changes mid-integration, and it refused twice here before the
tree was frozen — that is the guard doing its job, and both refused runs were
discarded rather than reported).  Both instrument self-checks passed:
`NEMO y10 ACC through THIS harness: 121.07 Sv vs recorded 121.07 Sv`.

| metric, 90-day twin from NEMO's day-180 state | previous round, the arm that ships | this tip | NEMO day 90 | 5x threshold |
|---|---|---|---|---|
| ACC [Sv] | `65.390274` | `65.390274` | `65.369204` | `4.550e-01` |
| upper contrast < 1400 m [kg/m3] | `-0.288146` | `-0.288146` | `-0.288182` | `5.500e-04` |
| deep contrast > 1400 m [kg/m3] | `-0.011261` | `-0.011261` | `-0.011258` | `2.250e-04` |
| southern-band surface sigma MAX [kg/m3] | `0.909348` | `0.909348` | `0.909343` | `4.750e-04` |
| southern-band surface sigma MEAN, \|diff\| from NEMO | `1.023e-06` | `1.023e-06` | — | `4.750e-04` |
| verdict | `PASS 5 / FAIL 0` | `PASS 5 / FAIL 0` | — | level 5x |

Every printed digit matches the previous round's BEFORE arm — the arm that
ships, since decision 67 was built, measured and then held.  So decision 72
moves no DINO number either, which is what the DINO cards selecting NEMO's
overwrite predicts.

---

## Gate results

One battery at a time on this host; every log under
`phase3/pr1802_dino_year/logs/`.

| gate | result |
|---|---|
| card gates: both DINO recipes, both tanks, the tank zero-diffusion removal, the DINO mesh / from-rest / step-1 / rank-dump gates | `221 passed, 9 warnings in 578.29s` |
| the six-file push gate plus this round's own test file, on the FINAL committed tip | `154 passed in 963.13s (0:16:03)` |
| the seven CI ratchets (constants, saturation, dispatch, validate-strict, param specs, inline coefficients, private imports) | `4 failed, 10560 passed, 4 skipped in 100.71s` |
| receipt citation gate, cumulative default receipt | `PASS`, 274 citations, 0 failures, 0 unmapped, 0 map-audit failures, all 9 self-tests fired |
| receipt citation gate, THIS receipt | `PASS`, 3 citations, 0 failures, 0 unmapped |
| the same gate with a planted shift on this receipt's own citation | `FAIL` and exit 1 |
| GYRE ladder, tanks, year | byte-identical, above; comparisons saved as `year_snapshot_compare.txt` and `report_compare.txt` |

The four ratchet failures are all in files this diff does not touch
(`git diff --name-only 24f8f7e75..HEAD` lists twelve files, none of them):
the Earth-radius literal in the mixed-precision Helmholtz validator, the
reference-salinity literal in the FESOM vertical-mixing bridge, and two
parameter-spec rows in the lateral-mixing and shortwave-penetration configs.
Both shrink-only baselines are untouched as well, so a file outside the diff
cannot have been pushed red by it.  The previous round ran a narrower ratchet
selection (one failure of 10,114); the two extra files here are the two extra
ratchets this round ran.

---

## Review

Both reviewers ran on this diff, adversarially, before it was declared done.

**Claude `code-reviewer`, independent fresh context: SHIP.**  No blocker, no
high.  It traced every construction site that can reach the anchor rather than
taking the claim on faith, confirmed against the GitHub-main checkout that the
new default reproduces main's expression exactly, checked the new tests for
coincidental passes, and verified all five NEMO citations against both the
shipped and the compiled source.  Its one substantive finding: the compiled
overwrite block starts at its `IF` on line 828, not at the log line on 829.
Correct, and fixed — every copy of that citation now names the guard.

**codex `exec --sandbox read-only`: DO NOT SHIP**, three findings, all acted
on rather than argued with:

1. *"The receipt overclaims DINO ownership.  Main is only observed NaN at day
   30; it is not shown failing at step 5 or by the same mechanism."*  **Fair,
   and the DINO section above is rewritten to separate what was measured from
   what was inferred.**  Codex also independently agreed with the operative
   conclusion: *"The trace does support keeping the guard: operands remain
   positive through step 5 and become invalid only after state corruption."*
2. *"Evidence does not substantiate every byte-identity claim ... the evidence
   directory lacks saved before/after comparisons for the 360 snapshots and
   tank reports.  The 90-day DINO run is incomplete.  The receipt still
   contains five placeholders."*  **Fair on all three.**  The comparisons are
   now written to `year_snapshot_compare.txt` and `report_compare.txt` in the
   evidence root rather than existing only in a shell, the DINO twin was
   re-run to completion on a frozen tree, and the placeholders are filled.
3. *"The test claiming to pin ORCA2 never constructs ORCA2 — it tests GYRE and
   DINO only."*  **Correct, and the sharpest of the three: it is a scope word
   that was never grepped.**  Closed by narrowing that test's claim to what it
   proves and adding `::test_orca2_card_inherits_the_rn_mxl0_overwrite`, which
   builds the real ORCA2 card from its own deck and asserts the resolved
   config takes the overwrite with NEMO's forced floor `1.0e-3 m` — and that
   this differs from its `rn_mxl0`, so the row cannot pass by coincidence.
   `19 passed in 18.38s`.

## Known-red list

Carried forward unchanged from the previous round, each pre-existing:

* `scripts/validate/cg_helmholtz_mixed_precision_1675.py` — an Earth-radius
  literal the constants ratchet rejects.  Not this round's.
* `packages/ocean/legoesm/ocean/physics/vertical_mixing/fesom_integration.py` —
  a reference-salinity literal, same ratchet.
* `tests/ocean/unit/test_tke_carried_coefficients.py::test_step_entry_n2_bundle_*`
  (two).
* The `nemo_dino_kamm` Euler card's fifth-step implicit-solve instability,
  already registered by round 184 and reproduced here on this tree and on
  GitHub `main`.  It does not touch the certified from-rest year, which runs on
  the MLF card.

## Behaviour changes for the PR body

1. **ORCA1 keeps 0.04 m; NEMO's actual value is 1e-3 m
   (zdftke.F90:859-862 overwriting `rn_mxl0` with the `rmxl_min` that
   zdftke.F90:841-843 forces to 1e-3 under `ln_zdfiwm`) — Pierre's call.**
   The two arms differ only where the wind anchor does not already exceed the
   floor, i.e. on calm and land columns; every windy column is identical.
   NEMO's transcription stays available to the NEMO-literal cards, which select
   it explicitly.
2. **No other production behaviour changes.**  GYRE, both tanks, ORCA2 and both
   DINO cards take NEMO's overwrite exactly as before; their certified numbers
   are byte-identical.
3. **The DINO from-rest year certification was re-run and completes.**  On its
   own card (the MLF one) it runs all 360 days at this tip; its temperature gap
   against NEMO is `6.982e-03` K at day 30 and `7.590e-03` K at day 360.  Those
   differ from the pinned table, but the pinned table predates this branch by
   many commits and the gap had already been re-measured once before this
   round, so nothing here is attributable to this PR — and this round's change
   is measured bit-inert on both DINO cards.  The separate Euler-card
   fifth-step instability round 184 registered is reproduced, unchanged, on
   this tree and on GitHub main.

## OPEN

1. **The `nemo_dino_kamm` Euler card's fifth-step instability**, already
   registered by round 184: bisect the first step whose tendency is wrong (the
   state is clean through step 3 and visibly wrong by step 4) on both trees.
   It does not block the year certification, which runs on the MLF card.
2. **Attribute the DINO from-rest year's drift from the pinned table**, if it
   matters: a controlled before/after year pair, since the pinned numbers come
   from a tree many commits old and were already re-measured once.
3. The two items the previous round left open: decision 67's real statement
   (the velocity-sum branch), and refreshing the two committed tank trajectory
   reference documents, which are older than the tip they are compared against.
