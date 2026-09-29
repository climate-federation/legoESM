# PR #1802 — decision 72, and the DINO year re-run

**Status: D72 LANDED.  THE DINO YEAR RE-RUN WAS ATTEMPTED AND IS STILL
REFUSED — and the refusal is neither a mesh defect nor a guard defect.**

Decision 72 is done and proven by value on the three columns it is about.
The DINO year certification was then re-run as asked; it refuses, and this
round measured WHY rather than repairing the detector.  The answer changes the
question: the geometry operand the guard rejects is finite and positive at
every call through the fifth step, the model's own STATE goes non-finite at
the output of that fifth step, and GitHub `main` does the same thing silently.
The guard is doing its job; the DINO from-rest run is what is broken, and it
was broken before this branch existed.

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

**The protocol.**  `dino_year_screen_fullframe.py nemo_dino_kamm <out.npz>`,
11,520 steps of 2,700 s from the analytic rest state on NEMO's own DINO mesh,
scored by `compare_fullframe.py`; the year-5 form of it is the table in
`docs/ocean/fidelity/dino_handoff_2026_07.md` (ACC 65.6 Sv, SST correlation
0.995, bias −0.17 K, sea-surface-height small-scale ratio 0.86).

**The guard that refuses.**  The fail-closed positivity check on the raw-mesh
thickness operand inside the shared NEMO `bn2` routine,
packages/ocean/legoesm/ocean/eos.py
(`compute_buoyancy_frequency_nemo_bn2`, the `e3w_source="mesh_reference"`
arm): *"raw-mesh e3w_int must contain only finite values > 0"*.

**The measurement.**  The round-182 instrument was reused rather than
re-written; it needed one argument (`--script`) so it could wrap the from-rest
year screen as well as the developed-state 90-day twin it already wrapped.  It
labels every Python call site that reaches that routine and prints, per
execution, the operand's minimum and its non-finite / zero / negative counts,
alongside a non-finite census of the model state at the step boundaries.
Log: `phase3/pr1802_dino_year/logs/dino_fromrest_trace.log`.

Six call sites reach the routine per step: the TKE step-entry N² bundle, and
five GM/Redi native-slope sites (the Tréguier kappa, the slope N², and the K33
assembly).  Per step, in execution order:

| step | operand non-finite cells, all six calls | state at step OUT | max abs u [m/s] |
|---|---|---|---|
| 1 | 0 | all finite | `3.405e-02` |
| 2 | 0 | all finite | `5.024e-02` |
| 3 | 0 | all finite | `6.175e-02` |
| 4 | 0 | all finite | `1.126e+00` |
| 5 | 0 | **T 268, S 268, u 334, v 333 non-finite** | `4.737e+24` |
| 6 | 347,200 | — | refusal |

**So: NEITHER.**  The raw-mesh operand is genuinely finite and positive
wherever the model is; it is not a DINO mesh defect, and the guard is not
reading the wrong operand.  It is reading the right operand one step after the
model destroyed itself.  The velocity grows by a factor of eighteen between
steps 3 and 4 and overflows at step 5: that is an explosive numerical
instability in the from-rest DINO run, and the guard is its first detector.
**The guard was therefore not loosened and no geometry "fix" was written.**

**The control that settles ownership.**  The identical screen, same recipe,
same NEMO mesh, same environment, run from the GitHub `main` checkout
(`d7109d9b0`) with main's own committed harness:

> `y1 day  30.0  T[nan,nan] usurf[nan,nan] max|u|=nan max|v|=nan max|eta|=nan finite=False`

`main` reaches day 30 with the whole field NaN and does not stop.

Stated no more strongly than it was measured, because the second reviewer
pushed back on exactly this:

* **CONFIRMED** — the lane's state is non-finite at the output of step 5, and
  `main`'s whole field is non-finite by day 30 on the same card, the same
  mesh and the same harness.  So a from-rest DINO run that completes a year
  does not exist on either tree, and the year certification is not blocked by
  something this round or this branch introduced.
* **PLAUSIBLE, not measured** — that `main` fails at step 5 too, and by the
  same mechanism.  Only the lane arm was instrumented, and the two trees
  differ in far more than this guard; `main` simply has no fail-closed
  positivity check on this path to stop it.  Attributing the two failures to
  one cause would need `main` instrumented as well, which is the first step of
  the follow-up round named in OPEN.

Log: `logs/dino_main_fromrest.log`.

**What this means for the DINO numbers.**  There are no new DINO year numbers
to report, old or new: the run cannot complete on either tree, so no ACC,
correlation or ratio exists to compare against the certified 65.6 Sv / 0.995 /
0.86.  The DINO evidence that DOES exist at this tip is the developed-state
90-day twin, re-run here on the after arm (below).  The year certification
stays BLOCKED, now with its root named instead of attributed to geometry.

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
* **NEW, and it is a finding rather than a red test: the DINO from-rest year
  screen goes non-finite at step 5 on this tree and on GitHub `main`.**  See
  the DINO section.

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
3. **The DINO year certification is still not re-run, and the reason is now
   measured.**  The from-rest DINO run destroys its own state at step 5 on this
   branch AND on GitHub `main`; this branch refuses loudly where main produces
   NaN silently.  No geometry guard was loosened.

## OPEN

1. **The from-rest DINO instability.**  Its own round: bisect the first step
   whose tendency is already wrong (the state is clean through step 3 and
   visibly wrong by step 4), on both trees, and decide whether the certified
   year-5 numbers can be reproduced at all on the current card.  Until then the
   DINO evidence for this PR is the 90-day developed twin.
2. The three items the previous round left open: decision 67's real statement
   (the velocity-sum branch), and refreshing the two committed tank trajectory
   reference documents, which are older than the tip they are compared against.
