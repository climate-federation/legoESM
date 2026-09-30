# GYRE NEMO-fidelity merge-main receipt — 2026-09-29

## Scope and provenance

Merge commit `f3607976a48259a4db9ad25ec800174ea0bf1ce4`, parents the lane
`d80355887` (tip `5301122fe2ef` plus the preregistration commit, docs only) and
GitHub `main` `aade0a44478bbc35447c351836010bea4e514079`.  Merge base
`e413aee94d3da166a5883a917a3fd55fda70d9ec` — the lane tip Pierre merged as PR
#1802 on 2026-09-28.  `main` has advanced **141** commits since that base; the
lane carries **29** of its own on it.  Ordinary merge: no squash, no rebase,
two parents.

Commit range `5301122fe..<final tip>`: the preregistration, the merge, the
citation re-anchor, and this receipt.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_merge_main_2026-09-29.md`,
committed before the merge ran.  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/merge_main_2026-09-29/`
(`before/`, `after/`, `wt_before/`, `wt_after/`, `wt_main/`, `wt_battery/`,
`logs/`, `citation_gate*.json`, `ladder_compare.json`, `compare_arms.py`).

## Textual conflicts

**None**, exactly as the operator's dry run predicted.  No deviation from the
preregistration.

## Files edited on BOTH sides — ONE, and it merged as a clean union

`packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py`.  The two
sides touch different NamedTuples, and the merged file carries both:

| compared against | what the merged file adds |
|---|---|
| the lane parent | ONLY main's deletion of `KPPConfig.c_b` and its `__param_spec__` note |
| the main parent | ONLY the lane's `TKEConfig.mxl0_min_m`, `nemo_mxl0_rmxl_min_overwrite` and their note |

Nothing was silently combined and nothing was dropped.

## Structural proof that nothing else was combined at all

Every path in the merged tree was compared, by blob hash, against BOTH parents:

| class | count |
|---|---:|
| blob identical to BOTH parents (neither side touched it) | 8487 |
| blob identical to the LANE parent only | 17 |
| blob identical to the MAIN parent only | 293 |
| blob identical to NEITHER parent | 2 — the file above, and this round's preregistration |
| paths in the MAIN parent missing from the merge | 0 |
| paths in the LANE parent missing from the merge | 2 |

The two missing lane-parent paths are **main's own deletions**, not merge
losses: `config/templates/coupled/amip.yaml` (renamed by `5b02a3d53`) and the
atmosphere lat-lon C-grid compressible-Euler model (parked under `_future/` by
`d4649266b`).  Both are off the ocean path; neither is touched by the lane.

All seventeen lane files — the barotropic step, the DINO experiment, the NEMO
recipe, the TKE closure, the ORCA1 OMIP card, the DINO month gate and its test,
the citation gate, and four receipts — survive verbatim.

The seams the previous merges had to preserve were re-checked, and the check is
stronger than a re-read: `ocean_model_latlon_cgrid.py`, `nemo_testcase_recipe.py`,
`ice/config.py`, `ice/sea_ice.py`, `grids/tripole.py` and `core/bulk_flux.py`
are byte-identical to BOTH parents, so the shortwave selector precedence, the
`nemo_stage_momentum_wzv_split` `bool | None` sentinel (GYRE `True`, ORCA2
`False`, unset RAISES), the sea-ice refusals from the 2026-09-25 merge and the
D68 tripole options are not merely unchanged in effect — they are unchanged in
bytes.  The TKE flags live in the one combined file and keep their `False`
library defaults there.

## Main-side changes on a shared path, with an execution verdict each

| main commit | what it changes | verdict on the certified cards |
|---|---|---|
| `a09a27f66` (PR #1810) | in `_bc_external_surface_forcing`, the net-heat denominator moves from `dz_ref[0] * J` to the live top thickness `h_k[..., 0]` | **EXECUTES on GYRE** (its gate passes `q_net`) and is **bit-identical**: `compute_layer_thickness` returns `dz_ref * J` on a pure z\* coordinate and `h_partial * J` on a partial-cell one, and NO certified card has a partial TOP cell (measured by the codex review from the card constructors: LOCK top `1.0 m`, OVERFLOW `20.0 m`, GYRE `10.003514801805068 m`; DINO's masked-zco construction gives `h_partial` in `{0, dz_ref}`). Same two operands, same product |
| `7317ed528` | `shortwave_penetration_tendency` gains `dz_live=`, and the Jerlov and generic branches pass `h_k` | **does NOT execute** on GYRE: the card selects `physics.shortwave_penetration.scheme = "nemo_qsr_2bd"`, so the generic call sits behind `if not _nemo_qsr_2bd:` and is skipped |
| `ffd6201599` | `BiharmonicConfig.enforce_cfl` DEFAULT `False` → `True`, `cfl_dt_estimate` deleted, `dt` threaded through the lateral-mixing `physics_fn` | inert: the NEMO recipe sets `lateral_mixing=LateralMixingConfig(scheme="none")`, so the helper that would receive `dt` is never built and the per-call dispatch is byte-identical. See OPEN for the latent problem this default flip activates elsewhere |
| `3609622f80` | deletes `KPPConfig.c_b`, `LinearDragConfig`, `QuadraticDragConfig` and their re-exports | no remaining reference anywhere in `packages/`, `scripts/` or `tests/`; GYRE runs TKE, not KPP |
| `646d6be8b8`, `dfcc1430b8`, `11d8b23107`, `fb70c6f3d7`, `d43453661d`, `1ac479a808`, `c1f778c44e`, `2815f86634` | the WOA / JRA55-do / Dai-Trenberth loaders now FAIL on a missing cache (`allow_synthetic` default `False`), and the caches are validated | no certified NEMO card calls them, and `run_omip_core2.py` already passes `allow_synthetic=False` at both of its call sites |
| `789f30bfc4`, `c2e81f43cd`, `35a6bde191`, `a2f96c8bb6`, `940abf1cef`, `3ad4a172f3` | cube-halo corner fill becomes a raising run-config field; dead YAML keys and mixed-precision names deleted | the deleted keys are atmosphere-side; the DINO card YAML sets none of them |
| `6a1873f41a` | new `redi_aht0` / `redi_coefficient` lateral-mixing fields | additive, defaults reproduce existing behaviour |
| `core/precision.py` (in `3ad4a172f3`) | deletes dtype-policy entries (`pressure_gradient`, `coriolis`, `ocean_timestepping`, …) as unconsumed | no ocean dynamics module consults that registry; the certified runs force fp64 anyway |
| `0e63f36cb5`, `f755a73432`, `9d5e9c631d`, `ac13e7455f`, `5b9ee3b6d9`, `fb730e4516`, `7a76d9a596` | MPAS, FV3-duo and AMIP lanes | off the lat-lon NEMO path |

**RETRACTION, kept loud.** The preregistration attributed the `dz_live=h_k`
insertions to `a09a27f66` (PR #1810).  That is wrong and the first reviewer
caught it: `a09a27f66` touches this file with three insertions and one
deletion — the denominator and its two comment lines — and its `dz_live` work
landed in `mpas_physics.py`, a different dycore.  The two `dz_live` lines here
come from `7317ed528`.  The conclusion is unchanged (the GYRE card skips that
branch), but the commit named in the preregistration was the wrong one.

## Trajectory proof — bit-identical

Both arms ran from CLEAN, COMMITTED, DETACHED worktrees with no
`LEGOESM_GATE_ALLOW_DIRTY` escape.  Before arm `wt_before` at `d80355887`
(the lane tip plus the docs-only preregistration); after arm `wt_after` at the
merge commit `f3607976a`.  Identical commands on both: the ladder gate
`--trajectory-only --max-step 10`; both tank gates `--max-step 10
--continue-after-first`; the from-rest member `--member 0 --days 360
--snap-steps 6 --tag year`; and the day-gap scorer on the eight certified
checkpoints against `phase3/year_fromrest`.

| comparison | result |
|---|---|
| `ladder.json` — all 14 content keys | **identical**; only `worktree` differs |
| `ladder.json` digest, provenance stripped | `7ba15556de2de841` on BOTH arms |
| `ladder.residuals.npz` | 210 arrays, **0 unequal**; whole-file sha256 `5269048ef189d113` on BOTH arms |
| certified 70-row ladder (`nemo_testcase_offline_compare.py`) | `OFFLINE_ORACLE_RELATIVE_COMPARE PASS: rows=70 max_worsening_ulps=0`, first-over-bar `{T,S,u,v,ssh}` kt 3 → unchanged. **0 rows moved** |
| the same instrument with `--plant worsen-3ulp` | `FAIL … max_worsening_ulps=3` — the comparison can fail |
| LOCK_EXCHANGE-zco, 50 rows | `DEBT`, first over bar `{u}` at kt 4; every content field equal except the `legoesm_git_sha` provenance stamp; 150 residual arrays **0 unequal**, file sha256 `a28d1f8ac5482b0f` both arms |
| OVERFLOW-zps, 50 rows | `DEBT`, first over bar `{T,u}` at kt 2; same, file sha256 `43e297c25b34b950` both arms |
| 360 daily snapshots `day001.npz`…`day360.npz` | **360 of 360 byte-identical**; `day030` `edd7297597965639`, `day240` `e3b784ba3e715825`, `day360` `e6a25563c7e10703`, each equal on both arms |
| the eight scored day-gap rows | **0 non-provenance field differences** |

The eight year rows, identical on both arms:

| day | T rms [K], both arms |
|---:|---|
| 30 | `2.3276772050683987e-06` |
| 60 | `1.4786e-05` |
| 90 | `1.6289e-05` |
| 120 | `1.0964e-04` |
| 180 | `6.1133e-05` |
| 240 | `6.586171881479517e-05` |
| 300 | `5.4749e-05` |
| 360 | `0.002670992385329469` |

Day 30, day 240 and day 360 reproduce the lane's certified values to every
digit.  The harness's run-to-run floor is ~`2e-10` K (round 129); this claim is
stronger — the two arms agree to the BYTE, not to the floor.  No row moved, so
no `main` commit had to be routed out of the NEMO identity and the
preregistration's HOLD branch was not taken.

NOTE on the tank digests: the tank report embeds `legoesm_git_sha` in the
document itself, so a whole-document digest necessarily differs between two
commits.  The field-for-field comparison with that one provenance key excluded,
and the byte-equal residual files, are what carry the tank claim — not a digest.

## DINO

Both DINO rows were measured on the merged tree.  Their before values were
measured earlier this week at commits whose diff to the before arm touches no
model code, exactly as the preregistration declared.

| card | before (lane, committed measurement) | after (merged tree) |
|---|---|---|
| from-rest month, day-30 wet 3-D T rms vs NEMO `kt=960` | `2.040288765e-03` K | **`2.040288765e-03` K**, `PASS` against the bar `2.244317642e-03` K |
| 90-day developed twin, ACC [Sv] | `65.359517` | `65.359517` |
| — upper contrast < 1400 m [kg/m3] | `-0.288137` | `-0.288137` |
| — deep contrast > 1400 m [kg/m3] | `-0.011256` | `-0.011256` |
| — southern-band surface sigma MAX | `0.909327` | `0.909327` |
| — southern-band surface sigma MEAN | `0.802911` | `0.802911` |
| — verdict | `PASS 5 / FAIL 0` at level 5x | `GATE 90D-TWIN: PASS 5 / FAIL 0 / level 5x / total 5` (the gate's own line, pipes rewritten as slashes for this table) |

Both of the twin's instrument self-checks passed on the merged tree:
`NEMO y10 ACC through THIS harness: 121.07 Sv vs recorded 121.07 Sv`, and band
volume `2.694775e+16 m3` at relative `4.98e-08`.  Every metric reproduces the
lane tip's re-certification to every printed digit.

## ORCA1 card fingerprint (D66 / D72)

The previous rounds' own probe (`phase3/pr1802_final/orca1_tke_fingerprint.py`)
was run on the merged tree and on GitHub `main`.  Main's own tree RAISES on the
probe's four-argument anchor call, because main still selects the masked
anchor and its TKE closure then demands a surface land mask; the run on main
therefore used a copy with the fallback widened to catch that refusal
(`phase3/merge_main_2026-09-29/orca1_tke_fingerprint_bothtrees.py`, one
character of difference, recorded rather than hidden).

| ORCA1 anchor column | GitHub `main` `aade0a444` | merged tree | D66 / D72 target |
|---|---|---|---|
| wet, windy, stress 0.10 Pa | `0.7951003609964353` m | `0.7951003609964353` m | unchanged |
| calm, stress 0 Pa, `ln_zdfiwm` off | `1e-08` m | **`0.04` m** | `0.04` m (D72) |
| calm, stress 0 Pa, `ln_zdfiwm` on | `0.001` m | **`0.04` m** | `0.04` m (D72) |
| LAND, stress 0.07 Pa | masked to the floor | **`0.5565702526975047`** m | `0.5565702526975047` m (D66) |

Read this the right way round.  `main` today carries the state PR #1802 landed;
D66 and D72 are the corrections the user took AFTER that PR merged, and they
live on this lane.  So the merged tree deliberately DIFFERS from `main` on
Pierre's card, on exactly the calm and land columns the two decisions name, and
reproduces both decisions' certified values to every digit.  Carrying that
difference to `main` is the purpose of the follow-up PR.  The windy column —
where the anchor is not sitting on its floor — is identical on both trees.

The probe's `--plant` arm (anchor doubled) moves every reported hash on the
merged tree, so the zero difference above is readable rather than vacuous.

## Citation gate

Main's insertion into the shared lat-lon C-grid step is five lines above line
4530.  Exactly three citations sit below it and each moved by a RIGID `+5` with
its extent unchanged; the remaining ten citations into that module sit above
the insertion and did not move at all.  Each shift was accepted only after the
cited text at the old lines in the lane parent was verified byte-identical to
the text at the new lines in the merged tree.

All three are in the shared lat-lon C-grid step.  The "before" column gives
the lane parent's line numbers as bare numbers, deliberately: written in
`file.py:NNNN` form they would read as live citations to lines that no longer
carry that text, and the gate would (correctly) flag them.

| citation | lane-parent lines | citation after the merge | extent |
|---|---|---|---:|
| ZAD operand barriers | 4968 to 4990 | `ocean_pe_latlon_cgrid.py:5021-5054` | 23 lines, unchanged |
| the lateral-friction velocity operands | 5307 to 5308 | `ocean_pe_latlon_cgrid.py:5425-5426` | 2 lines, unchanged |
| the `rho_prime` / `h_k` argument | 5330 | `ocean_pe_latlon_cgrid.py:5448` | 1 line, unchanged |

Applied to `CITATION_MAP` and to the round-8 receipt's prose.  No NEMO
compiled-source citation moved — the merge does not touch the oracle builds.

After the re-anchor the gate reports `status PASS`, **274 citations, 0
failures, 0 map entries failing audit, 0 unmapped**, and **all nine self-test
plants fired**; the worktree stamp is `clean: true`.  Planting a shift on
`ocean_pe_latlon_cgrid.py:5448` makes the gate `FAIL` and exit 1, so it can
fail.

## Tests

Every summary line below is pytest's own last line.

**Broad ocean battery** (`tests/ocean/fidelity tests/ocean/unit
tests/unit/test_run_omip_cli.py`, `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu`, one
battery at a time on this host).

The FIRST attempt is DISCARDED, not quoted: it ran while other agents' jobs
were on the host, a worker died with `MemoryError` while pytest was rendering a
traceback, and xdist turned that into an `INTERNALERROR` that aborted the
session:

> `317 failed, 5622 passed, 126 skipped, 2 xfailed, 34 warnings, 54 errors in
> 995.93s` — with `MemoryError` and five `node down: Not properly terminated`

A second attempt on a quiet host crashed the same way at a different test
(`29 failed, 5521 passed`, one `MemoryError`, four nodes down), which located
the problem: the crash is in pytest's TRACEBACK RENDERING, not in the tests.
The run that is quoted was taken with `--tb=no -rf`, which removes the
rendering step, and it COMPLETED:

> `205 failed, 8508 passed, 179 skipped, 2 xfailed, 84 warnings, 14 errors in
> 3180.64s (0:53:00)`

Under twelve parallel workers most of those are execution artefacts, so all
**205** flagged IDs were re-run on the same merged tree at `-n 4`.  The node-id
list was passed as ARGV by a committed helper rather than through a shell,
because a bracketed parametrisation glob-expands in bash and the batch then
collects nothing — the 2026-09-25 merge's vacuous batch, reproduced here once
before it was fixed (`no tests ran in 2.12s`, discarded).

> `21 failed, 184 passed, 3 warnings in 552.95s (0:09:12)`

Those 21 were then classified, ONE PROCESS PER ID, against the lane parent
(`wt_before`) and against GitHub `main` (`wt_main`):

| class | count | reading |
|---|---:|---|
| fails on the LANE PARENT **and** on `main` | **20** | pre-existing debt on both sides, untouched by the merge |
| passes on both parents, flagged merged | 1 | **execution artefact** — see below |
| genuine merge interaction | **0** | — |

The twenty are exactly the known-red families: the worktree-stamp ratchet, the
four float32 advection-gradient underflow cases, the three stpmlf
call-coverage rows, the two step-entry buoyancy-bundle rows, the SI3
scalar-math v2 gate, and main's own MPAS/diagnostics rows (the recipe case
board, `test_bbl_adv_mpas`, `test_diag_omip_nemo_battery`,
`test_k_zeta_bih_resolution_scaling`, `test_nemo_zdfmxl_transcription`,
`test_teos10_rab_bn2`), plus the round-129 spread-floor gate, the 501 config
nesting row and the AB3-AM4 live-split row — every one red on BOTH parents.

The single remaining ID,
`tests/ocean/unit/test_ocean_scm.py::test_coriolis_rotation_conserves_kinetic_energy`,
was run ALONE on the merged tree and **PASSES** — `1 passed in 115.78s`.  It
takes 116 seconds by itself against this suite's 120-second per-test timeout,
so under worker pressure it times out; it is a parallel-execution artefact, not
a merge defect.  **So the merge introduces no new failure.**

**Citation gate**, at the final tip: `status PASS`, 274 citations, 0 failures,
0 map entries failing audit, 0 unmapped, all 9 self-test plants fired,
worktree `clean: true`.  The planted control `FAIL`s and exits 1.

**Push gate** (the six files named in `autopilot/autopilot_max.sh`), at the
FINAL tip on a clean committed tree:

> `135 passed in 925.09s (0:15:25)` at `1c9db3c62`, and again
> `135 passed in 917.74s (0:15:17)` at the final tip `9c9cb889c`

Same count as the 2026-09-25 merge round's push gate.  The citation gate was
re-run at the final tip as well — `status PASS`, 274 citations, 0 failures, 0
map entries failing audit, 0 unmapped, 9 of 9 plants fired, worktree
`clean: true` — and this receipt's own citations pass their own run of the
gate: `PASS`, 3 citations, 0 failures, 0 unmapped.  The first run against this
receipt returned `FAIL` with 3 unmapped, which is the gate doing its job: the
shift table's pre-merge line numbers were written in `file:line` form and so
read as live citations to text that had moved.  Recorded rather than quietly
fixed; the table now gives those as plain numbers and says why.


## Independent reviews

Two, both adversarial, both on the merge and its diff against both parents, and
both returned before this receipt was written.

**Claude `code-reviewer` subagent — SHIP.**  It re-derived the blob arithmetic
independently (294/17/2 by its own counting convention, the same partition),
confirmed the union in the one combined file, confirmed both claimed deletions
were main's, and found no dropped hunk and no dangling reference.  It CORRECTED
the preregistration's commit attribution for the `dz_live` lines (retracted
above).  It confirmed the denominator equality from `compute_layer_thickness`
and confirmed GYRE skips the `dz_live` branch.  It raised one forward-looking
item, carried to OPEN below.  It recorded as UNVERIFIED that DINO was inert by
code reading only — this receipt closes that by measurement.

**codex `exec --sandbox read-only` — SHIP.**  Per claim: merge composition, no
defect; certified execution paths, no defect, and it went further than asked by
reading the top-cell thickness out of every card constructor (LOCK `1.0 m`,
OVERFLOW `20.0 m`, GYRE `10.003514801805068 m`, DINO `h_partial` in
`{0, dz_ref}`) to prove no certified card has a partial top cell; off-path
changes, pre-existing and no newly exposed defect; citation re-anchor, exact
`+5` with byte-identical text.  It also noted, correctly, that the merge's
literal first parent is the preregistration commit rather than the lane tip —
the preregistration is docs only, which is why the before arm was run there.

Neither reviewer found a merge defect, so nothing had to be fixed; the one
correction and the one forward-looking item are recorded rather than acted on.

## Choices

Merge glue only.  No configuration, scheme, default, threshold or carried-state
change on any card; no default value moved anywhere by this round.

UNASKED: **none.**  The three merge choices offered for revert by the
2026-09-25 receipt were KEPT by note AX and are untouched here.

## Verdict

**SHIP.**  The GYRE NEMO identity is proven bit-identical across the merge —
the certified 70-row ladder with zero rows moved and the same digest
`7ba15556de2de841` on both arms, 210 residual arrays with none unequal, all 360
daily snapshots byte-identical, and day 30 / 240 / 360 reproducing the certified
values to every digit.  Both tanks are byte-identical.  Both DINO cards
reproduce their certified numbers exactly.  Pierre's ORCA1 card carries D66 and
D72 exactly as the user decided them.  Two independent adversarial reviews found
no merge defect.

## OPEN

- **Latent, PRE-EXISTING on `main`, reported not fixed.**  Main's flip of
  `BiharmonicConfig.enforce_cfl` to `True` by default, together with the new
  requirement that the run's timestep be passed, means a lat-lon C-grid card
  that selects FACTORY biharmonic lateral mixing now raises: the lat-lon step
  calls its physics function without a timestep, and the biharmonic closure
  refuses when the cap is enforced and the timestep is absent.  Both reviewers
  confirmed it independently.  No card in this campaign selects it (the NEMO
  recipe sets lateral mixing to `"none"`), so nothing here is affected; it
  belongs to whoever owns the factory lateral-mixing path on `main`.
- **Forward-looking, raised by the Claude reviewer.**  The RGB-chlorophyll
  branch of the external surface forcing now carries the live thickness, and
  ORCA2-zps is the one card that selects the RGB scheme.  It is inert today only
  because no ORCA2 gate builds a forced surface at all.  Whoever extends ORCA2
  (or DINO's NEMO-literal lat-lon path) to real forced steps should know that
  this branch will be live on its FIRST measurement — it will not be silently
  un-pinning an existing number, but it must be pinned when it arrives.
- **Carried forward, not opened here.**  The ORCA2 second per-stage continuity
  solve question (round 163) is still unanswered, and the merge does not touch
  it.  The pivot-mesh `nemo_avg` precedence item from the 2026-09-25 merge is
  still open for the ORCA2 and eORCA025 lanes.
- The day-gap scorer needs its `--days` list stated explicitly; a call without
  it refuses rather than silently scoring a shorter list.  Both arms were scored
  on the same eight certified days.
