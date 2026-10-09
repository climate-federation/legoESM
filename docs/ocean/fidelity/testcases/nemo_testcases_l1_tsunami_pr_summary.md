# PR SUMMARY — TSUNAMI (NEMO tests/TSUNAMI, key_RK3 deck), standalone on main

Date 2026-10-09 (round 9, DECISION 105). Branch
`fidelity/nemo-testcases-tsunami-standalone`, based on `origin/main-pin` =
climate-federation main `a06629f58`. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round9/`.

**Independent of the seamount PR #1910.** The lane branch
(`fidelity/nemo-testcases-tsunami`, PR #1913) was stacked on the seamount tip
`0575754c0`. This branch replays the 66 TSUNAMI commits of rounds 1-8
(`git log --no-merges 0575754c0..1acdf3fad ^origin/main-pin`) onto main with
`git cherry-pick -x`, in order, with every seamount statement left out, then
re-anchors the citation map to main's lines. Skipped: the main-merge commit
`9cb8e6695` and the two main commits it carried (`5e368e87b`, `4e13ac2a2`;
already on main). The operator test-allowlist commit `5b9bde8a1` was kept: its
test (`tests/ocean/unit/test_nemo_prognostic_barotropic_state.py`) exists on main.

Base note: `origin/main-pin` has moved 4 commits past `1e1c2c890` (MIT
relicense, a vertical-diffusion microbenchmark). None touches `packages/` or
`src/`. Every certified number below is unchanged.

## 1. What this PR carries vs main

10 files under `packages/`, 37 elsewhere (scripts, tests, docs); 47 files,
+5912 / -200.

**The TSUNAMI card** (`build_tsunami_zco_card`): NEMO 5.0.2 `tests/TSUNAMI`,
2000 km x 2000 km doubly periodic, 10 km, f-plane 38.5 N, jpk = 2, one wet
level of 100 m, 100 steps of 1000 s. Split-explicit free surface, EEN
vorticity, constant vertical mixing, everything else off (tracer and momentum
lateral diffusion off via `K_h = 0`, `A_h = 0`). It sets no seamount-only option.
Declared deviation `TSUNAMI_DEVIATIONS = (("key_RK3", False, True, "DECISION 100"),)`.

**Five card-selected options.** The TSUNAMI card is the only caller that
selects each one. Library defaults and every other card are unchanged. Caller
grep on this branch (main + TSUNAMI) over `packages src scripts`:
`caller_grep.txt`, sha256 `748424f31c00aca4`.

| option | library default | selected by (recipe line) | other callers on main |
|---|---|---|---|
| momentum advection OFF: `momentum_flux_scheme="none"` + `vertical_momentum_scheme="none"` (ln_dynadv_OFF; the validator makes them one selection) | `"upwind"` / `"upwind_perturbation"` | TSUNAMI card (3178) | none; only comments in `ocean_pe_latlon_cgrid.py` |
| tracer advection OFF: `tracer_advection="none"` (ln_traadv_OFF) | `"tvd"` | TSUNAMI card (3181) | none |
| j-periodic step: model config `meridionally_periodic` + card `j_periodic` | `False` | TSUNAMI card (3142, 3187, 3197; the validator ties both to the deck's ln_Jperio) | none. The model scopes its config value (True or False) around every entry point. Helpers called outside the model read the context flag, default `False` |
| one wet level: `allow_single_level` | `False` (min 2 levels) | TSUNAMI card (3128) | none |
| density depth `eos_depth="geometric"` (DECISION 101) | `"insitu"` | TSUNAMI card (3174) | existing certified arm, also set by other testcase cards (recipe 491, 610) and DINO paths; unchanged |

**Shared-code edits** (all inert when the option is off, checked by the
closed-card hashes in section 3): the y-wrap context and its scoping in
`halo_latlon.py`, `latlon_cgrid_operators.py`, `ocean_model_latlon_cgrid.py`
(step body, tendency entry points, runtime mask check, the four outer
integrators), `init_latlon_cgrid.py` (`replace_land_mask`), `vertical.py`
(one wet level), `ocean_pe_latlon_cgrid.py` + `state.py` (advection-OFF arms).

**RK3 acquisition** (`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_tsunami/run_rk3.sh`)
with `check_records.py --program rk3` (110 records admitted), the geometry
identity gate, the kt = 1..10 / 100-step ladder, the offline hpg_sco replay,
receipts for rounds 1-5, 7, 8 with their PREREG files.

## 2. Replay conflicts (every hunk)

| original commit | replayed as | file | hunks | kept |
|---|---|---|---|---|
| `7e8622e32` | `790b50728` | GYRE round-8 receipt | 1 | TSUNAMI side (a line number) |
| `7e8622e32` | `790b50728` | citation gate map | 4 | TSUNAMI side (line-number keys) |
| `2f9c994fa` | `c18e7f384` | citation gate map | 1 | TSUNAMI side (line-number keys) |
| `bb6c04c1c` | `86537407e` | citation gate map | 22 | TSUNAMI side (line-number keys) |
| `2946a87b6` | `df917d570` | GYRE round-8 receipt | 8 | TSUNAMI side (line numbers) |
| `53893bfd2` | `84e34f10d` | `ocean_model_latlon_cgrid.py` (end of module) | 1 | the y-wrap step-body wrapper. Dropped the seamount-only `_NEMOWSLdfDiagnosticTrace` class that sat beside it as context (not on main) |

No other commit conflicted. The recipe, `config.py`, `gm_redi_latlon_cgrid.py`
and `vertical.py` applied cleanly. The diff adds no seamount statement: a grep
of the added lines under `packages` and `tests` for `smt`, `seamount`, `closed_bottom`, `LdfDiagnostic` is
empty. The seamount tracer-LDF hook and the closed-bottom mask / live divisor
are not on this branch. Per-commit check: 58 of 66 replayed patches have the
same patch-id as the original. Five are the conflicted commits above. The
other three (`153a11c80`, `98f2f8c42`, `1eb05a9cb`) have identical `+`/`-`
lines; only their context differs, because the seamount lines are absent.

All line-number hunks were then re-anchored once against the final tree:

- 47 map keys and 14 GYRE-receipt citations were moved to the lines the gate
  itself identifies.
- One range (`ocean_model_latlon_cgrid.py:8805-9241`) went back to main's
  437-line extent. The seamount hook had widened it to 453.

## 3. Certification (standalone tree, commit `599265056b00`, clean)

| gate | result |
|---|---|
| geometry identity | `GEOMETRY IDENTICAL`, **35/35** EXACT; json `ef0ce940518b08a2`, equal to round 8's except the sha stamp |
| kt = 1..10: independent, given-entry, stp_2D rhs, substeps, handoff, stages | all six jsons equal to round 5's final jsons except `legoesm_git_sha` |
| 100-step record, kt = 1..100 | `per_kt` 100/100 rows equal to round 8; no field over the 1e-15 bar; **worst ssh 7.693e-16 normalised at kt 90**; json `aa55c8a12fb3e6c1` (differs from round 8 only in the sha stamp) |
| closed cards, state sha256 after 3 steps, at `origin/main-pin` and at this branch | equal on both sides and equal to round 8: LOCK_EXCHANGE `b3aa27fa80dbdefd`, OVERFLOW `72ad6c2d6a9be145`, GYRE `b8cb5ecd4dc4b467`, VORTEX `b0ef8aaf1dc75f91`, VORTEX_VEC `ce76bd63f0441d3f` (VORTEX_SMT4_VEC is seamount-only, skipped) |

No certified number moved. TSUNAMI is at the bar, not bit-identical (1e-16
relative floor). The given-entry rows stay separate from the independent ones
(D52).

## 4. Other gates

| gate | result |
|---|---|
| citation gate, default (GYRE receipt from `## Round 25 —`) | `PASS`, 274 citations, 0 failing map entries; plant `ocean_model_latlon_cgrid.py:8190`: `FAIL`, exit 1 |
| citation gate, TSUNAMI receipts from `## Citations` | rounds 1, 2, 4, 5, 7, 8 `PASS` (29, 30, 11, 11, 2, 2 citations). Round 3 `FAIL` with 2 unmapped citations, same as on the lane (debt 7) |
| push battery (citation gate, TKE terms, recipe, real freshwater closure; ws stage face mask, prognostic barotropic state) | `126 passed` + `10 passed` = 136, as on the lane |
| TSUNAMI tests (card, check_records, geometry gate, ladder) + advection-off arms + barotropic inertial oscillation | `82 passed` |
| CI ratchets (constants, inline coefficients, private imports, dispatch, validate-strict, physics contracts, federation plan) | 3 failed, 5958 passed. The same 3 also fail on `origin/main-pin` (land `restart.py` coefficient, `test_jra55_do.py` constant, unassigned `taxonomy` module). None come from this PR |

## 5. Review history (codex)

1. Operator review of the lane PR diff (round 6): **BLOCK**, three findings.
   - y-wrap config not authoritative: FIXED in round 7.
   - `implicit_solver.py` custom-VJP: UPSTREAM, main's `5e368e87b`, not in this diff.
   - Ladder forced the global scope: FIXED in round 7.
2. Operator re-review (round 7 head): **BLOCK** on one site class: the four
   outer integrators built face masks outside the scope. FIXED in round 8, with
   a planted partial-cell AB2 test.
3. Operator final review (round 8 head): **SHIP**. Draft PR #1913 opened.

In-round single reviews ran in rounds 1-5 and 7. Their CONFIRMED findings
were fixed in the same rounds. This round's single review (codex) of
`git diff origin/main-pin...HEAD -- packages tests` is recorded in section 8.
NO GATE for dual review: GLM was not run.

## 6. ORCA2 pointer (from round 5)

Certified at the bar over 100 steps and shared with ORCA2 rung 0:

- the split-explicit dyn_spg_ts loop;
- EEN in the barotropic Coriolis (planetary-only form);
- the cyclic i-exchange.

Not shared: the j-wrap; ORCA2 is closed south and folds north. Not measured on
ORCA2 itself.

## 7. Registered debts

1. 1e-16 floor: the unattributed 1.09e-19 stp_2D rhs difference is the only gap to bit-identity.
2. Interior one-ulp moves at kt <= 10 under B4j: unattributed.
3. The y-wrap has no MPI/SPMD band exchange; the helpers refuse it.
4. Shipped-deck (leapfrog, no key_RK3) TSUNAMI stays unscored (DECISION 100).
5. D-TSU-1 (PLAUSIBLE): the acquisition does not grep the preprocessed source for the cpp keys' effect.
6. D-TSU-2: the passivity admission compares float32 outputs, so it cannot see a last-bits perturbation of NEMO.
7. The round-3 receipt does not gate from its first heading (2 unmapped citations).
8. Helpers called outside the model (diagnostics, scripts, `_future`) read the context flag, default False.

## 8. Round-9 review and choices

Single review (codex): see the round-9 commit that records it.

| choice | ASKED or UNASKED |
|---|---|
| base on `origin/main-pin` as named (now `a06629f58`, 4 non-package commits past `1e1c2c890`) | ASKED (ref named); the moved tip is UNASKED, inert for every gate above |
| line-anchor conflicts resolved to the TSUNAMI side, then re-anchored to the lines the gate identifies | UNASKED (mechanical; gate PASS + firing plant) |
| keep operator commit `5b9bde8a1` (its test exists on main) | ASKED (rule given in the task) |
