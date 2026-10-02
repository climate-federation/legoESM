# Preregistration — GYRE NEMO-fidelity lane merges GitHub `main`, 2026-09-29

Written and committed BEFORE `git merge github/main` runs.

## Scope

Lane tip `5301122fe2ef` (branch `fidelity/nemo-testcases-l2-gyre-codex2`) merges
GitHub `main` `aade0a444`.  Merge base
`e413aee94d3da166a5883a917a3fd55fda70d9ec` — the tip Pierre merged as PR #1802
on 2026-09-28.  `main` has advanced **141 commits** since; the lane carries
**29** commits of its own on top of the same base (the PR #1802 review fixes,
the ORCA1 card restorations D66/D72, the DINO month regression fix and its
gate, and the DINO twin re-certification).

No squash, no rebase: one ordinary merge commit with two parents.

## Files edited on BOTH sides — ONE

`git diff --name-only e413aee94 github/main` (295 files) ∩
`git diff --name-only e413aee94 HEAD` (18 files):

```
packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py
```

The two sides touch DIFFERENT NamedTuples in that file:

| side | change |
|---|---|
| `main` (`3609622f80`) | deletes `KPPConfig.c_b` and its `__param_spec__` exclusion note |
| lane (round 187 / D72) | adds `TKEConfig.mxl0_min_m`, `TKEConfig.nemo_mxl0_rmxl_min_overwrite` and one `__param_spec__` exclusion note |

The operator's dry run reports **zero textual conflicts**.  A conflict, anywhere,
is recorded in the receipt as a deviation from this preregistration.

## The real risk, named before the merge: main-side shared-path changes

Zero conflicts does not mean zero semantics.  `main` edited 104 files under
`packages/`, and the lane's certified cards execute some of them.  These are the
main-side changes this round will check for NEMO-card exposure, each with the
verdict it must be given:

| # | main commit | what it changes | why it could reach a NEMO card |
|---|---|---|---|
| 1 | `a09a27f667` (PR #1810) | in `_bc_external_surface_forcing`, the net-heat denominator moves from `dz_ref[0] * J` to the LIVE top thickness `h_k[..., 0]` | the GYRE gate **does** pass `q_net` (`nemo_testcase_l2_gyre_phase3_gate.py:715`), so this line EXECUTES on the certified card. Expected inert because GYRE's top cell is full: `compute_layer_thickness` returns `dz_ref * J` (pure z\*) or `h_partial * J` with `h_partial[...,0] = dz_ref[0]` (partial-cell, top cell full), i.e. the same product |
| 2 | `a09a27f667`, `7317ed528f` | `shortwave_penetration_tendency` gains `dz_live=`, and the two-band / Jerlov branches now pass `h_k` | both GYRE and ORCA2 select `physics.shortwave_penetration.scheme` (`nemo_qsr_rgb`), which the 2026-09-17 seam routes AWAY from these branches; the claim is that the NEMO qsr arm is untouched |
| 3 | `ffd6201599` | `BiharmonicConfig.enforce_cfl` DEFAULT moves `False` → `True`, and `cfl_dt_estimate` is DELETED; the cap now uses the run's real `dt`, threaded through the lateral-mixing `physics_fn` signature | a moved default. Expected inert on the NEMO cards, which set `B_h = 0.0` and `K_bih = 0.0`, and which do not build the cubed-sphere factory path this helper serves |
| 4 | `3609622f80` | deletes `KPPConfig.c_b`, `LinearDragConfig` and `QuadraticDragConfig` (the whole classes, and their `__init__` re-exports) | positional layout of `KPPConfig` changes; a removed public name breaks any lane import. Grep says the lane imports none of them outside the deleted module's own package |
| 5 | `646d6be8b8`, `dfcc1430b8`, `11d8b23107`, `fb70c6f3d7`, `d43453661d`, `1ac479a808`, `c1f778c44e`, `2815f86634` | forcing / observation loaders now FAIL by default (synthetic is opt-in), WOA SSS must have uniform latitude, JRA55-do cache validation refuses Celsius / transposed / missing-year stores | a previously tolerated condition becoming a hard error can refuse a card at load time. DINO and the OMIP cards read WOA |
| 6 | `789f30bfc4`, `c2e81f43cd`, `35a6bde191`, `a2f96c8bb6`, `940abf1cef`, `3ad4a172f3` | cube-halo corner fill becomes a run-config field with raising env validation; dead YAML keys and mixed-precision names deleted from `src/legoesm/config.py` | a deleted YAML key that a lane card sets would silently stop being read |
| 6b | `6a1873f41a` | new `redi_aht0` / `redi_coefficient` lateral-mixing fields | additive; default must reproduce existing behaviour |
| 7 | `0e63f36cb5`, `f755a73432`, `9d5e9c631d`, `ac13e7455f`, `5b9ee3b6d9`, `fb730e4516`, `7a76d9a596` | MPAS / FV3 / AMIP lanes | off the lat-lon NEMO path; verdict recorded, not measured |

The seams the previous merges had to preserve are re-verified unchanged in the
merged tree: the **shortwave selector precedence** (NEMO qsr identities route
through `physics.shortwave_penetration.scheme`, everything else through
`surface_forcing.shortwave_scheme`, a card setting both refuses); the
**`nemo_stage_momentum_wzv_split`** `bool | None` sentinel stated per card
(GYRE `True`, ORCA2 `False`, unset RAISES); the **TKE flags**
`nemo_derived_mxl_min`, `nemo_mxl0_surface_tmask` and
`nemo_mxl0_rmxl_min_overwrite` keeping their `False` library defaults with only
the NEMO-literal cards selecting them (decisions D66 / D72); the **sea-ice
refusals** added by the 2026-09-25 merge; and the **tripole options** from D68,
which stay on the NEMO-literal ORCA2 card alone.

## Resolution rule, stated before any conflict is seen

1. A conflicting hunk carrying a NEMO-transcribed statement is resolved to the
   side matching the compiled NEMO source the lane cites, and the receipt quotes
   that `ppsrc` `file:line`.  Preference, tidiness and recency decide nothing.
2. Where main's change is orthogonal to the NEMO identities, the resolution is
   the UNION: main's addition verbatim, the lane's identities keeping their
   exact routing.
3. No default moves as part of merge glue.  A main-side default that disagrees
   with a NEMO card is routed out through that card's CONFIG, never by editing
   the library default — or the merge HOLDs.

## The claim this round will prove

**The merged tree's certified trajectories are bit-identical to the pre-merge
lane tip's.**  Two arms from CLEAN, COMMITTED, DETACHED worktrees: before = this
preregistration commit, after = the merge commit.  Identical commands.

| quantity | instrument | pass condition |
|---|---|---|
| GYRE certified ladder, 70 rows | `nemo_testcase_l2_gyre_phase3_gate.py --trajectory-only --max-step 10` then `nemo_testcase_offline_compare.py` | `rows=70`, `max_worsening_ulps=0`, **0 rows moved**; report digest (provenance stripped) `7ba15556de2de841` on both arms |
| `ladder.residuals.npz` | same run | 210 arrays, **0 unequal**, whole-file sha256 equal |
| GYRE year, day 30 | `nemo_testcase_l2_gyre_year_fromrest.py --member 0 --days 360 --snap-steps 6 --tag year`, scored by `nemo_testcase_l2_gyre_year_owners.py --day-gap` | `2.3276772050683987e-06` K on both arms, equal to the byte |
| GYRE year, day 240 | same | `6.586171881479517e-05` K |
| GYRE year, day 360 | same | `0.002670992385329469` K |
| the 360 daily snapshots | same | 360 of 360 byte-identical |
| LOCK_EXCHANGE-zco tank, 50 rows | `nemo_testcase_phase3_trajectory_gate.py --case LOCK_EXCHANGE-zco --max-step 10 --continue-after-first` | every field equal; 150 residual arrays, 0 unequal |
| OVERFLOW-zps tank, 50 rows | same, `--case OVERFLOW-zps` | same |
| DINO from-rest month | `LEGOESM_DINO_MONTH_GATE=1 dino_fromrest_month_gate.py --work-dir phase3/dino_month_gate` | `PASS`, day-30 T rms `2.040288765e-03` K, not worse than the pinned bar |
| DINO 90-day developed twin | `acceptance_gate_90d.py --run-recipe nemo_dino_kamm_mlf` | `PASS 5 / FAIL 0` at level 5x, ACC `65.359517` Sv |
| ORCA1 card fingerprint (D66 / D72) | `phase3/pr1802_final/orca1_tke_fingerprint.py` on the merged tree and on `github/main` | Pierre's card equals main's on the wet, land and calm anchor columns |

The harness's run-to-run floor is ~`2e-10` K (round 129).  The claim here is
stronger than the floor: byte equality, not agreement within it.

**If any row moves**, the merge brought a shared-code change that reaches the
NEMO path.  The response is fixed in advance: find the `main` commit with
`git log -S` or a bisect over the merged files, cite it, and either route it out
of the NEMO identity through CONFIGURATION (never by moving a default) or HOLD
the merge with the receipt naming the commit and the row.  Salvaging the number
by adjusting the instrument is out of bounds.

## Before-arm reuse, declared in advance

The DINO 90-day developed twin's before value (`65.359517` Sv, `PASS 5 / FAIL
0`) and the DINO month gate's before value (`2.040288765e-03` K) were measured
THIS WEEK on commits that are ancestors of the before arm and whose diff to the
before arm touches no model code — `5301122fe` for the twin, `a7e853e49` for the
month.  Those committed measurements stand as the before arm for those two
cards; the after arm is run fresh.  Every GYRE and tank row is run fresh on both
arms.

## Also preregistered

- The citation gate is re-anchored by RIGID shift only: a citation moves by a
  constant line delta, extent unchanged, and only after the cited text at the
  old lines in the lane parent is verified byte-identical to the text at the new
  lines in the merged tree.  No citation is re-pointed at different text.  All
  nine self-test plants must fire.
- The broad ocean-fidelity battery runs ONCE, one battery at a time on this
  host, and its failures are diffed against the known-red lists in the DINO
  month-regression and D72 receipts.  Anything new is a merge defect, not lane
  debt.
- Two independent reviews: a Claude `code-reviewer` subagent on the merge diff
  and this round's receipt, and `codex exec` **only if**
  `phase3/codex_quota_guard.sh` exits 0.  If it refuses, the single Claude
  review is the review and the receipt says so.
- Push gate at the FINAL tip: the citation gate plus the five files named in
  `autopilot/autopilot_max.sh`, quoting pytest's own summary line.
