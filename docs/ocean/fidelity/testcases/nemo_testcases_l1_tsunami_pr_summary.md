# PR SUMMARY — TSUNAMI lane (NEMO tests/TSUNAMI, key_RK3 deck)

Date 2026-10-09 (round 6, DECISION 102). Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round6/`.

**Stacking.** This branch sits on the seamount branch
`fidelity/nemo-testcases-l2-gyre-codex2` (draft PR #1910, not merged); the
first TSUNAMI commit's parent is that branch's tip `0575754c0`. The operator
opens this PR **against that branch** and retargets it to main when #1910
merges. 60 commits (58 from rounds 1-5, the main merge, one test fix); 35 files outside
`packages/`, 11 files under `packages/` + `src/`.

## 1. Main merge

Merged `origin/main-pin` = climate-federation main `1e1c2c890` (only that
ref) as `9cb8e6695`. 0 textual conflicts. The lane's merge-base with it was
`471bee222`; main is 4 commits ahead. Files main changed:

| path | change |
|---|---|
| `packages/core/legoesm/timestepping/tridiagonal.py` | + `diffusion_thomas_solve` (levels-first scan, +130) |
| `packages/ocean/legoesm/ocean/physics/vertical_mixing/implicit_solver.py` | `implicit_vertical_diffusion_ocean` routes through it; new `_diffusion_interface_coeffs` (+/-76) |
| `tests/ocean/unit/test_implicit_vmix_batched.py`, `tests/unit/test_diffusion_thomas_solve.py`, `README.md` | tests, readme |

Nothing else under `packages/` or `src/`.

**Does TSUNAMI execute them? No (MEASURED).** The card runs
`zdf_implicit_solver_evaluation="nemo_literal"`. A call-counting probe over
3 steps of the TSUNAMI card (`r6_solver_exec_probe.py`) records only
`implicit_vertical_diffusion_nemo_momentum` (2 calls) and
`implicit_vertical_diffusion_nemo_tracer_pair` (1 call). Neither
`implicit_vertical_diffusion_ocean`, `_build_implicit_tridiag`,
`thomas_solve` nor `diffusion_thomas_solve` is called. GYRE gives the same
picture. The numerical gates below confirm it: every certified row is
unchanged.

**Composition.** Main's changes and the lane's changes touch disjoint
functions: the lane never edited `implicit_solver.py` or `tridiagonal.py`.
Citation-map spans: none moved (the map cites NEMO sources, which main did
not touch); the default gate passes with no re-anchoring.

## 2. What the branch carries beyond the seamount branch

**The TSUNAMI card** (`build_tsunami_zco_card`): NEMO 5.0.2 `tests/TSUNAMI`,
2000 km x 2000 km doubly periodic, 10 km, f-plane 38.5 N, jpk = 2, one wet
level of 100 m, 100 steps of 1000 s; split-explicit free surface, EEN
vorticity, constant vertical mixing, everything else off. Declared deviation
`TSUNAMI_DEVIATIONS = (("key_RK3", False, True, "DECISION 100"),)`.

**Card-selected options.** Each is selected explicitly by the TSUNAMI card
only; library defaults and every other card are unchanged. Caller grep
(`caller_grep.txt`, sha256 `8a9c416009897524`, `grep -rn ... packages src
scripts`):

| option | library default | selected by | other callers |
|---|---|---|---|
| momentum advection OFF: `momentum_flux_scheme="none"` + `vertical_momentum_scheme="none"` (ln_dynadv_OFF; the validator makes them one selection) | `"upwind"` / `"upwind_perturbation"` | TSUNAMI card only (recipe line 3209) | none; only comments in `ocean_pe_latlon_cgrid.py` |
| tracer advection OFF: `tracer_advection="none"` (ln_traadv_OFF) | `"tvd"` | TSUNAMI card only (line 3212) | none |
| j-periodic exchange: `meridionally_periodic` (model config; the model scopes it, True or False, at every entry point; round 7) and the card's `j_periodic` | `False` | TSUNAMI card only (line 3218; validator requires both equal the deck's ln_Jperio) | none; library helpers called outside the model read the context flag, default `False` |
| one wet level: `allow_single_level` | `False` (min 2 levels) | TSUNAMI card only (line 3159) | none |
| density depth: `eos_depth="geometric"` (DECISION 101) | `"insitu"` | TSUNAMI card (line 3205) | existing certified arm, also set by other testcase cards (recipe lines 491, 610; unchanged) |

Closed-card check that the rewritten statements are inert off-flag: state
sha256 after 3 steps, six closed cards, identical to the round-5 set (section 4).

**RK3 acquisition** (`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_tsunami/`,
`run_rk3.sh`): builds the shipped case with key_RK3 added and key_xios
dropped (documented deviation, DECISION 100), refuses unless compiled keys
are exactly key_RK3 key_qco key_vco_1d. Per-stage record patches; the
operator's run is admitted by `check_records.py --program rk3`: **110
records**, admission json byte-identical to the operator's (round 3).

**Gates** (scripts + unit tests): geometry identity gate (card mesh vs the
run's mesh_mask, 35 arrays, firing plant); kt = 1..10 ladder (independent,
given-NEMO-entry, rhs, substeps, handoff, stages) and the 100-step record
arm; the offline hpg_sco replay.

**Five receipts**: `nemo_testcases_l1_tsunami_round1_receipt.md` (survey,
card transcribed not executable, acquisition) ... `round5_receipt.md`
(B4j j-wrap; at the bar), each with its `PREREG_*` file.

## 3. Certification numbers (this round, merge commit `9cb8e669541a`, clean tree)

| gate | result |
|---|---|
| geometry identity | `GEOMETRY IDENTICAL`, **35/35** EXACT (json `ef0ce940518b08a2`); plant `--plant "grid dx_u"`: `GEOMETRY DIFFERS`, exit 1 |
| kt = 1..10 stp_2D RHS (given NEMO entry) | floor 1.09e-19 at kt 1 (max 1.63e-19 at kt 8); 10/10 rows identical to round 5 |
| kt = 1..10 independent, given-entry | 10/10 and 10/10 rows identical to round 5 (`per_kt` equal) |
| substeps, handoff, stages | `per_kt` equal to round 5 |
| 100-step record, kt = 1..100 | no field over the 1e-15 bar; **worst ssh 7.693e-16 normalised at kt 90**; 100/100 rows identical to round 5 (stamp `legoesm_git_sha` = `9cb8e669541ac25a889f8e0cc270fadb6c8e1579`, json `d7d2bff377142efc`) |
| closed cards, state sha256 after 3 steps | equal to round 5: LOCK_EXCHANGE `b3aa27fa80dbdefd`, OVERFLOW `72ad6c2d6a9be145`, GYRE `b8cb5ecd4dc4b467`, VORTEX `b0ef8aaf1dc75f91`, VORTEX_VEC `ce76bd63f0441d3f`, VORTEX_SMT4_VEC `1cc2f09098310c7f` |

No certified number moved. TSUNAMI is at the bar, not bit-identical (1e-16
relative floor). Label: the numbers are independent of NEMO entry except the
rows named given-entry (D52 separation kept).

## 4. Other gates

| gate | result |
|---|---|
| citation gate, default (seamount receipt, from `## Round 25 —`) | `PASS`, 274 citations |
| citation gate, TSUNAMI round 5 receipt (`## Citations`) | `PASS`, 11 citations, 0 failures, 0 unmapped, 0 map entries failing audit; plant `lbclnk.f90:1868` shifted by 2: `FAIL`, exit 1 |
| citation gate, rounds 1, 2, 4 | `PASS`; round 3 `FAIL` (2 unmapped citations; fails identically at `b1514bf89`, not gated from its first heading, same as round 5's note) |
| push battery (citation gate, TKE terms, recipe, real freshwater closure, ws stage face mask, prognostic barotropic state) | `136 passed in 663.40s` |
| every TSUNAMI test (check_records, geometry gate, ladder, card, advection-off arms) | `72 passed, 1 failed` at the merge; the one red, `test_record100_leaves_the_bar_at_kt15_on_the_j_seam`, fails identically at `b1514bf89` (it still asserted round 4's walled-seam departure that round 5 removed). Expectation was stale, code right: test replaced by "record stays at the bar through kt 100"; ladder file `13 passed` |

## 5. ORCA2 pointer (from round 5)

Shared with ORCA2 rung 0 and now certified at the bar over 100 steps: the
split-explicit dyn_spg_ts loop (boxcar weights, forward start, AB3,
surface pressure gradient, face-depth update, transport accumulation); EEN
in the barotropic Coriolis (planetary-only form; relative vorticity and
non-uniform f not certified); the cyclic i-exchange. NOT shared: the
j-wrap (ORCA2 is closed south, folds north; with the flag off every
rewritten statement is the old expression, and ORCA2's southern `ff_f` copy
keeps its literal statement). Not measured on ORCA2 itself. No held ORCA2 item
is refuted; the rung-0 external-stage ssh debt is **narrowed**: PLAUSIBLY in
what ORCA2 feeds the loop (masks and coast, north fold, relative-vorticity
EEN, slow forcing) rather than in the loop.

## 6. Registered debts

1. 1e-16 floor: the unattributed 1.09e-19 stp_2D rhs difference is the only gap to bit-identity.
2. Interior one-ulp moves at kt <= 10 under B4j: unattributed.
3. The y-wrap has no MPI/SPMD band exchange; the helpers refuse it.
4. Shipped-deck (leapfrog, no key_RK3) TSUNAMI stays unscored (DECISION 100).
5. D-TSU-1 (PLAUSIBLE): the acquisition does not grep the preprocessed source for the cpp keys' effect.
6. D-TSU-2: the passivity admission compares float32 outputs, so it cannot see a last-bits perturbation of NEMO.
7. Round 3 receipt does not gate from its first heading (2 unmapped citations).

## 7. Choices this round

| choice | ASKED or UNASKED |
|---|---|
| merge exactly `origin/main-pin` | ASKED |
| replace the stale kt=15 test by the at-the-bar assertion | UNASKED (test-only, repairs a red the merge did not cause; offered for revert) |

## 8. Review

**Operator codex review of the package diff (round 6 head `7c5cf10f2`): BLOCK**,
three findings, dispositions in `nemo_testcases_l1_tsunami_round7_receipt.md`:

| finding | disposition (round 7) |
|---|---|
| 1 HIGH: `meridionally_periodic=False` did not override the process-global y-wrap; public tendency entry points bypassed the wrapper; cross-thread contamination | FIXED: the model scopes its config value (True and False) around the step body, both tendency entry points, the vertical-K diagnostic, the cache primer and the runtime mask check; the flag is a `ContextVar`; both face-mask constructors take the value explicitly |
| 2 HIGH: `implicit_solver.py` swapped to a custom-VJP solver, tests relaxed | UPSTREAM: main's `5e368e87ba` (PR #1911, Pierre Gentine), arrived with the main merge; untouched here; TSUNAMI does not execute it; recorded for the operator to raise |
| 3 MEDIUM: the ladder forced the global scope | FIXED: no scope in the ladder or tests; removing the model wiring fails the record test (planted) |

After the fixes: 100-step record 100/100 rows equal to round 6 (worst
7.693e-16 at kt 90); six closed cards' state sha256 unchanged; push battery
136 passed; TSUNAMI tests 80 passed.

**Operator second codex re-review of the PR diff (round 7 head `e1940fd76`):
BLOCK** on ONE CONFIRMED site class: the AB2 / leapfrog / NEMO-MLF / unsplit
outer integrators built partial-cell face masks outside the config's y-wrap
scope. FIXED in round 8: the four integrators join the wrapped entry points
(`nemo_testcases_l1_tsunami_round8_receipt.md`, with the full caller list); a
periodic partial-cell AB2 plant fails without the scoping. Record 100/100 rows
and six closed-card hashes equal to round 7. Review history: round 6 BLOCK ->
round 7 fixes -> re-review BLOCK on this one site class -> fixed here.

**Single review (codex) of the round-7 fix: BLOCK** on two new CONFIRMED
findings (the runtime mask check outside the scope; `replace_land_mask`
reading the ambient flag), both fixed and planted in round 7; that last fix
was not re-reviewed. NO GATE for dual review: GLM not run.

UNVERIFIED: the round-6 call-count probe proves the card does not call main's
changed solver in 3 steps, not that no path ever could; batteries ran with 8
xdist workers.
