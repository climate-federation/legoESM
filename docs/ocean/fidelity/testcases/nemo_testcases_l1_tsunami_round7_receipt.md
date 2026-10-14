# TSUNAMI round 7 receipt — the y-wrap comes from the model config

Date 2026-10-09. Base `7c5cf10f2` (round 6). Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round7/`.
Trigger: the operator's codex review of the PR diff (`0575754c0...HEAD`,
packages + tests) returned BLOCK with three findings. No NEMO statement is
added or changed this round; this is a wiring fix, so there is no preregistered
measurement, only the bar the round must hold (round 6's numbers, exactly).

## Dispositions

| # | finding | disposition |
|---|---|---|
| 1 | HIGH: `meridionally_periodic=False` does not override the process-global y-wrap; public `tendencies()` / `tendencies_with_diagnostics()` bypass the wrapper; mutable global allows cross-thread contamination | **FIXED.** `ocean_model_latlon_cgrid.py:15955-15979`: the wrapper now traces inside `meridional_periodicity(config.meridionally_periodic)` for True AND False (the effective config, honouring a `config=` override), and wraps `_step_impl`, `tendencies`, `tendencies_with_diagnostics`, `diagnose_vertical_K`, `prime_step_caches` (the vertex-mask N-S exchange). `halo_latlon.py:48`: the flag is a `ContextVar`, so a scope set in one thread is invisible to a trace in another. `init_latlon_cgrid.py:117,178`: `rest_state_latlon_cgrid_ocean(meridionally_periodic=)` builds the face masks from an explicit argument; the TSUNAMI card passes the deck's ln_Jperio (`nemo_testcase_recipe.py:3173`) instead of scoping the global. |
| 2 | HIGH: `implicit_solver.py:209` unconditionally swaps `thomas_solve` for a new custom-VJP solver; tests relaxed | **UPSTREAM, not touched.** Origin is MAIN, not this lane (see below). |
| 3 | MEDIUM: the ladder forces the global scope, so the record test cannot see the wiring removed | **FIXED.** The scope is gone from the ladder (`nemo_testcase_l1_tsunami_ladder.py:234` onward, no `meridional_periodicity` import) and from every test that stepped the model. Planted: removing the wrapper makes `test_record100_stays_at_the_bar_through_kt100_with_the_j_wrap` (`test_nemo_tsunami_ladder.py:172`) fail. |

### Upstream finding (for the operator to raise)

`git log 0575754c0..HEAD -- .../implicit_solver.py` gives one commit,
`5e368e87ba` "Ocean vertical diffusion: build the tridiagonal bands inside the
level sweep", author Pierre Gentine, 2026-10-08, merged to main by PR #1911
(`perf/ocean-thomas-scan`) and reaching this branch only through round 6's
merge of main `1e1c2c890`. What changed: `implicit_vertical_diffusion_ocean`'s
default path calls the new `tridiagonal.diffusion_thomas_solve` (hand-written
custom VJP) instead of `thomas_solve`; the f32 opt-in, pair and batched paths
are unchanged. Results agree with the old path to rounding, not bitwise.
Tolerances relaxed in `tests/ocean/unit/test_implicit_vmix_batched.py`:
pair-vs-single went from `assert_array_equal` to `assert_allclose` at
1e-13 (f64) / 1e-5 (f32), and 1e-13 / 1e-6 on the mixed-precision case (the
bitwise check survives only against `thomas_solve` on prebuilt bands; the
commit message calls this an owner decision). TSUNAMI does not execute this
path (round 6's call-count probe; the card runs the NEMO-literal solver).

## Gates (clean tree; final numbers at `ec049bd48`, after the codex fixes)

| gate | result |
|---|---|
| 100-step record, kt = 1..100 | 100/100 `per_kt` rows equal to round 6; no field over the bar; worst ssh **7.693e-16** normalised at kt 90 (json `d9fe65c5cd66cb71`, stamp `ec049bd482f3`; same at `36f2423a4`, json `d068bdc562176ae5`) |
| six closed cards, state sha256 after 3 steps, no caller scope | equal to round 6: LOCK_EXCHANGE `b3aa27fa80dbdefd`, OVERFLOW `72ad6c2d6a9be145`, GYRE `b8cb5ecd4dc4b467`, VORTEX `b0ef8aaf1dc75f91`, VORTEX_VEC `ce76bd63f0441d3f`, VORTEX_SMT4_VEC `1cc2f09098310c7f` (json `a8c2a1fc07d45b73`, identical at `36f2423a4` and `ec049bd48`) |
| plant A: round 6's wrapper (True only, step only) | `2 failed`: the step-topology test (walled config under a True caller scope) and the new public-tendencies test |
| plant B: no wrapper at all | `1 failed`: the ladder record test (finding 3's non-vacuity) |
| citation gate, this receipt (from its Citations heading) | `PASS`, 2 citations, 0 failures; plant `lbclnk.f90:1868` shifted: `FAIL`, exit 1; default gate `PASS`, 274 |
| plant C: invariant check not wrapped | `1 failed` (the mask/runtime-check test) |
| plant D: `replace_land_mask` ignores its argument | `1 failed` (same test) |
| TSUNAMI tests + advection-off arms + #576 PoC | `80 passed` (final); main's solver tests `49 passed` (at `36f2423a4`, with the arms) |
| push battery | `111 passed` + `25 passed` = 136, as round 6 (final, and at `36f2423a4`) |
| lat-lon halo users + import/constant ratchets (at `36f2423a4`) | `5840 passed, 2 failed`: `test_jra55_do.py` (hardcoded constant) and `land/restart.py` (inline coefficient), per-file checks on files this diff does not touch |

## Boundary (what still reads the context flag directly)

Library functions called outside the ocean model (the atmosphere lat-lon
sharded step, direct calls to the barotropic or operator helpers) read the
context value, whose default is False, the closed wall.
They take no config, so they cannot see the model's. The meridionally-FLAT
mode is still a module global; codex did not flag it and it was not in scope.

## ORCA2 pointer

ORCA2 is closed in the south and folds in the north, so its config keeps
`meridionally_periodic=False`. That value is now enforced at every model entry
point, so a stray periodic scope cannot reach an ORCA2 trace.

## Choices this round

| choice | ASKED or UNASKED |
|---|---|
| ContextVar instead of deleting the global (deleting means threading the config through 80+ helper call sites) | UNASKED (offered for revert) |
| also wrap `diagnose_vertical_K` and `prime_step_caches` | UNASKED (both trace the lat-axis helpers; offered for revert) |
| `rest_state_latlon_cgrid_ocean(meridionally_periodic=False)` new argument; default = the old global default | UNASKED (pure addition; only two callers scoped the global, both updated) |
| #576 PoC test selects the wrap through the config | UNASKED (test-only) |
| codex fixes: wrap `_assert_runtime_invariants`; `replace_land_mask(meridionally_periodic=False)` | UNASKED (acting on CONFIRMED findings; default = old global default; offered for revert) |

## Review

Single review (codex), read-only, on `7c5cf10f2..36f2423a4` in the context of
the full PR diff (`codex_review.log`, sha256 `6f2bd619d1ed8077`): **BLOCK**,
two findings, both CONFIRMED by reading the code, both fixed in `ec049bd48`:

| finding | disposition |
|---|---|
| HIGH: `step_checked` runs the face-mask invariant outside the config scope, so a periodic model with runtime checks on rejects its own valid state (`ocean_model_latlon_cgrid.py:15362`) | FIXED: `_assert_runtime_invariants` joins the wrapped entry points (`ocean_model_latlon_cgrid.py:15974-15979`); plant C fires |
| MEDIUM: `replace_land_mask` derives face masks from the ambient flag (`init_latlon_cgrid.py:361`) | FIXED: explicit `meridionally_periodic` argument, default False; plant D fires |

Codex also reported: no sign/index/staggering defect (the periodic copies
match NEMO's `jpfillperio`); defaults stay closed when nothing is selected; the
new step/tendency/ladder tests are non-vacuous; the model's jit caches are
keyed on a static `self` and the compiled pressure-gradient path carries the
periodicity in its static key. The fix commit was not re-reviewed (one review
per round). NO GATE for dual review: GLM not run.

## Citations

The exchange the flag selects: `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbclnk.f90:1868`,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbclnk.f90:2026-2033`.
