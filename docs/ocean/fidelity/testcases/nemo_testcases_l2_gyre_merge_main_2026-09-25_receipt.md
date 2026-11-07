# GYRE NEMO-fidelity merge-main receipt — 2026-09-25

## Scope and provenance

Merge commit `839e988c83f1d0b86b0e40afd427267d1f4f91d0`, parents lane
`36cc29f0a55eec7c394b7409280f2d8af9ad8bae` (round 169 tip `ce29fc3c9475` plus
the preregistration commit, docs only) and GitHub `main`
`d3f6248410ae6079e50460a8d961de9efdd612a4`.  `main` advanced 374 commits since
the previous merge base `9f4b16d633f0`, which is itself the 2026-09-17
merge-main commit.  Ordinary merge: no squash, no rebase, two parents.

Commit range `36cc29f0a..d60d2cbcd` (five commits): the preregistration, the
merge, the citation re-anchor, the one citation extent main's insertion
changed, and the fix closing the two findings both reviewers raised.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_merge_main_2026-09-25.md`,
committed before the merge ran.  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/merge_main_2026-09-25/`
(`before/`, `after/`, `wt_before/`, `wt_after/`, `citation_gate*.json`,
`codex_review.log`, `push_gate.log`, `card_gates.log`, `broad_ocean.log`).

## Textual conflicts

Six files conflicted — exactly the six the preregistration named from the
operator's dry run, no more and no fewer.  Every one is a single hunk.

| # | file | what each side added | resolution and what decides it |
|---|---|---|---|
| 1 | `packages/core/legoesm/core/bulk_flux.py` | main: `large_yeager_cesm` in the valid-scheme tuple plus its comment. lane: `nemo_si3_constant`. | UNION, both names in one tuple, both comments kept. Orthogonal: the lane's NEMO identity is a separate dispatch arm (`nemo_si3_constant_fluxes`, same file) that main never touches, and the GYRE card selects neither. |
| 2 | `packages/core/legoesm/grids/tripole.py` | comment text only; both sides describe the SAME u-point convention, and the code line below the conflict (`e1u[:, -1:]` prepended) is identical on both sides. | main's block kept, because it records the measurement (`glamu - glamt = +0.5000` deg on eORCA1) behind the convention the lane's block asserts. No executable line differs between the two sides. |
| 3 | `packages/ice/legoesm/ice/config.py` | main: `lead_freeze_source`. lane: `thermo_scheme`, `si3`, `ice_constants`, `Ce_ice`. | UNION. Both sides appended "to preserve positional constructors" and only one claim can survive; main's field keeps its index and the lane's four move one place later (see the review section — this was changed after review). |
| 4 | `packages/ice/legoesm/ice/sea_ice.py` | main: an `ocean_freezing_temperature_K` override at the step entry. lane: the `thermo_scheme` validation and the layered-SI3 dispatch. | UNION, main's override first so it also reaches the layered branch, then the lane's guard. Inert for every card on this lane, which passes the argument nowhere. A refusal was added after review (below). |
| 5 | `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` | main: a `pivot_row_stored` branch in the tripolar-fold vertex-thickness helper for the de-haloed eORCA025 mesh class. lane: `nemo_avg4` added to the NEMO-average branch's selector. | UNION with main's branch FIRST and the lane's second, plus the lane's `nemo_avg4` refusal hoisted above both. See below. |
| 6 | `tests/test_validate_strict_coverage.py` | main: a nested-grid strict-validation test. lane: the ocean shortwave-selector strict-validation test. | UNION, both tests kept. |

### Conflict 5, the only one on the NEMO path

The helper builds the vertex (F-point) thickness the energy/enstrophy vorticity
scheme consumes.  The lane's NEMO statement is `dynvor.F90:918-950` (the
`nn_e3f_typ` construction) with the fully-dry-vertex exit at `dynvor.F90:986`,
and the lane's own code declares `nemo_avg4` **undefined on a tripolar fold**,
because its certified NEMO use is the closed beta-plane GYRE box.

Main's new branch is a tripolar fold too.  Taking main's structure verbatim
would therefore have let a pivot-layout mesh selecting `nemo_avg4` compute a
vertex thickness the lane had declared undefined, instead of raising — a
silent-physics hole the merge would have opened.  The resolution hoists the
lane's refusal above BOTH branches, so it fires on every tripolar fold
exactly as the lane intended, and main's pivot branch keeps its precedence for
every other scheme.  For every non-pivot mesh — which is every card this lane
certifies — the executed statement is the lane's, unchanged.

GYRE never reaches any of this: its grid is a closed beta-plane box, so the
enclosing `fold_is_local(grid) or nmask is not None` is False.  The trajectory
proof below is the gate on that claim.

### The two seams the merge had to preserve, re-verified

- **Shortwave selector.** `main` did not touch it; the 2026-09-17 precedence
  rule stands byte-identical in the merged tree: the NEMO qsr identities route
  through `physics.shortwave_penetration.scheme`, every other value through
  `surface_forcing.shortwave_scheme`, and a card setting both refuses with a
  named error.
- **`nemo_stage_momentum_wzv_split`** is still a `bool | None` sentinel stated
  explicitly per card (GYRE `True`, ORCA2 `False`); an unset card still raises.
  No default moved.

## Auto-merged overlap audit

Twenty other files were edited on both sides and merged cleanly.  Main's hunks
in each were read and given an execution verdict against the resolved GYRE
NEMO card.  Only the executable ones are listed; the rest (MPAS model and
config, the OMIP runner script, an sbatch line, four test modules) are off the
lat-lon GYRE path entirely.

| file | main's change | GYRE verdict |
|---|---|---|
| `constants.py` | new `q_sat_saline_fraction` | additive; unread by the card |
| `core/precision.py` | new `finalize_to_storage` | new function, and a no-op unless storage and accumulate dtypes differ; GYRE runs fp64 |
| `grids/latlon.py` | `FoldDescriptor` gains `pivot_row_stored`, `perm_u`, `perm_f`, all defaulted | additive; GYRE has no fold |
| `grids/operators_latlon_cgrid.py` | pivot-layout fold rows in four operators | every hunk is inside a fold branch; GYRE has no fold |
| `ocean/dynamics/latlon_cgrid_operators.py` | same, plus the pivot case in the density-Jacobian pressure gradient | same |
| `ocean/dynamics/ocean_model_latlon_cgrid.py` | the carried-TKE required set now demands the surface viscosity only under `nemo_z0` **and** mixing-length choice 3 or 4; plus `omega=` on the isoneutral K33 call | **executes, and is inert**: GYRE's resolved mixing-length choice is 3 (read off the built config, not the recipe's `_replace` block), so the required set and the cold-start seeding are exactly what they were |
| `ocean/eos.py` | corrected UNESCO-80 docstring, new `in_situ_temperature` | additive; the card uses TEOS-10 |
| `ocean/physics/lateral_mixing/config.py` | `redi_coefficient` (default `"constant"`), `redi_aht0`, `redi_f_f` | defaults reproduce existing behaviour |
| `ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py` | native-slope active-mask helper extracted; `omega` threaded | refactor plus a threaded argument |
| `ocean/physics/shortwave_penetration.py` | new `top_layer_absorbed_fraction` | new function, called only by the sea-ice lead budget |
| `ocean/restart.py` | four TKE slots classified prognostic | restart I/O only; the certified runs start from rest |

## Trajectory proof — bit-identical

Both arms ran from CLEAN, COMMITTED, DETACHED worktrees with no
`LEGOESM_GATE_ALLOW_DIRTY` escape; each artifact's `worktree` stamp records
`clean: true` and its commit.  Before arm: `wt_before` at `36cc29f0a5`
(the lane plus the docs-only preregistration).  After arm: `wt_after` at the
merge commit `839e988c83`.  Identical commands on both:
the ladder gate `--trajectory-only --max-step 10`, the from-rest member
`--member 0 --days 360 --snap-steps 6 --tag year`, and the day-gap scorer on
the eight certified checkpoints against
`phase3/year_fromrest`.

| comparison | result |
|---|---|
| `ladder.json` — all 14 content keys (`steps`, `barotropic_state_steps`, `first_over_bar`, `scalar_math_root_identity`, …) | **identical**; only `worktree` differs |
| `ladder.json` digest, provenance stripped | `cf06a8fc7d0e90f2` on BOTH arms |
| `ladder.residuals.npz` | 210 arrays, **0 unequal**; whole-file sha256 `43f37831256832c3` on BOTH arms |
| certified 70-row ladder (`nemo_testcase_offline_compare.py`) | `OFFLINE_ORACLE_RELATIVE_COMPARE PASS: rows=70 max_worsening_ulps=0`, first-over-bar `{T,S,u,v,ssh} kt 3` → unchanged. **0 rows moved.** |
| the same instrument with `--plant worsen-3ulp` | `FAIL … max_worsening_ulps=3` — the comparison can fail |
| 360 daily snapshots `day001.npz`…`day360.npz` | **360 of 360 byte-identical** by sha256; `day030` `a66143733bcc9e4e`, `day240` `0d4f16d0c51da705`, `day360` `c6b7e1523b9a6dbb`, each equal on both arms |
| the member manifest | differs only in `wall_seconds` and `worktree` |
| the eight scored checkpoints | every field byte-identical; **0 non-provenance row fields differ** |

The eight year rows, identical on both arms and reproducing round 163's
certified numbers to every printed digit:

| day | T rms [K], both arms | round 163's certified value |
|---:|---|---|
| 30 | `6.572574374770603e-05` | `6.572574374770603e-05` |
| 60 | `2.0808020003520387e-04` | `2.0808020004e-04` |
| 90 | `1.8643018773756063e-03` | `1.8643018774e-03` |
| 120 | `1.04847381054101e-03` | `1.0484738105e-03` |
| 180 | `3.5838854686407826e-03` | `3.5838854686e-03` |
| 240 | `1.644836070117868e-02` | `1.6448360701178680e-02` |
| 300 | `1.3602990041069166e-02` | `1.3602990041e-02` |
| 360 | `1.1225660018551306e-02` | `1.1225660018551306e-02` |

The harness's run-to-run floor is ~2e-10 K (round 129).  This claim is
stronger: the two arms agree to the BYTE, not to the floor.  No row moved, so
no `main` commit had to be routed out of the NEMO identity and the
preregistration's HOLD branch was not taken.

The riskiest auto-merged hunk was checked independently of the measurement as
well, because "the number did not move" and "the code cannot move it" are
different claims: main's carried-TKE guard narrows its required set to mixing-
length choices 3 and 4, and GYRE's resolved choice is 3, so the guard's
behaviour on the card is identical.

## Citation gate

Main's insertions moved fifty-two cited line ranges in the two lat-lon C-grid
modules.  Each was re-anchored by a RIGID shift — the same delta on both
endpoints, extent unchanged — and each shift was accepted only after the cited
text at the old lines in the lane parent was verified byte-identical to the
text at the new lines in the merged tree (51 of 52 verified this way; the
52nd is the exception below).  Deltas, one per distinct insertion point above
the cited lines:

| module | delta | citations |
|---|---:|---:|
| `ocean_model_latlon_cgrid.py` | +1 | 4 |
| | +3 | 8 |
| | +7 | 13 |
| | +8 | 2 |
| | +50 | 3 |
| | +68 | 8 |
| `ocean_pe_latlon_cgrid.py` | +7 | 8 |
| | +30 | 5 |

**One citation is NOT a rigid shift and is recorded as such.**
`ocean_model_latlon_cgrid.py:7932-8364` → `:7834-8267`: main inserted one line
(`omega=_cfg_b.omega,` on the isoneutral K33 call) INSIDE the cited span, so
the first endpoint moves +7 and the last +8, and the map's pinned extent goes
from 433 to 434 lines.  Both pinned endpoint statements are unchanged, and the
map now carries a comment saying why.

Applied to `CITATION_MAP` (52 keys) and to three documents' prose
(the round-8 receipt, 14 citations; the round-162 receipt, 5; the round-162
preregistration, 4).  **No NEMO compiled-source citation moved** — the merge
does not touch the oracle builds.

After the re-anchor the gate reports `status PASS`, 274 citations, 0 failures,
0 map entries failing audit, 0 unmapped, and **all nine self-test plants
fired** (`AMBIGUOUS-ANCHOR`, `widened extent`, `reversed range`, both
single-endpoint shifts, both comma-interior plants, and the two unplanted
baselines that must pass).

## Tests

Every summary line below is pytest's own last line.

**Push gate** (the six files in `autopilot_max.sh`, `JAX_ENABLE_X64=1
JAX_PLATFORMS=cpu`, lane `PYTHONPATH`), at the final tip:

> `135 passed in 939.22s (0:15:39)`

Run three times, the last of them at the FINAL tip `6007a41dc` on a clean
committed tree: `135 passed in 1038.30s`, `135 passed in 992.87s`,
`135 passed in 939.22s`.  Same count every time.

The citation gate was re-run at that same final tip: `status PASS`, 274
citations, 0 failures, 0 map entries failing audit, 0 unmapped, 9 of 9 plants
fired, worktree `clean: true` at `6007a41dc`.

**Card gates** — the same five files round 163 ran, so the counts are
comparable:

> `160 passed, 9 warnings in 350.76s (0:05:50)` (DINO experiment, the L1 tanks
> Rule-12 gate, the lock-exchange slow-forcing owner, the overflow barotropic
> gate)
> `10 passed in 7.50s` (the round-34 tank zdf-removal gate, run separately)

170 passed in total — **the same count round 163 reported**, so DINO, both
tanks, the lock-exchange and the overflow cards are unchanged by the merge.

**Broad ocean battery** (`tests/ocean/fidelity tests/ocean/unit
tests/unit/test_run_omip_cli.py`, `-n 12`), one battery at a time:

> `154 failed, 8475 passed, 179 skipped, 2 xfailed, 79 warnings, 3 errors in
> 3281.47s (0:54:41)`

157 IDs were flagged.  Under twelve parallel workers many of those are
execution artefacts, so all 157 were re-run on a STABLE merged tree with less
worker pressure:

> `21 failed, 136 passed, 3 warnings in 744.44s (0:12:24)`

Those 21 were then classified against the lane parent AND against main, one
process per ID (an `-n 8` batch of node IDs collects nothing when any ID is
absent from the tree, which is how the first attempt returned a vacuous "no
tests ran" — that run is discarded, not quoted):

| class | count | reading |
|---|---:|---|
| fails on the lane parent too | 11 | pre-existing lane debt, untouched by the merge |
| absent from the lane parent, and fails on main's own tree | 9 | main's own debt, arriving with the merge |
| **passes on the lane parent AND on main, fails merged** | **1** | **a genuine merge interaction — below** |

The 11 pre-existing reds are the worktree-stamp ratchet, the four float32
advection-gradient underflow cases, the three stpmlf call-coverage rows, the
two step-entry buoyancy-bundle rows and the SI3 scalar-math v2 gate.  The 9
main-side reds are MPAS and diagnostics rows (`test_bbl_adv_mpas`,
`test_mpas_physics` RGB chlorophyll, both `test_shortwave_scheme_routing` MPAS
rows, `test_teos10_rab_bn2` MPAS wiring, `test_nemo_zdfmxl_transcription`,
`test_k_zeta_bih_resolution_scaling`, `test_diag_omip_nemo_battery`,
`test_recipe_case_board`); each was run on a clean `github/main` worktree and
each failed there.

### The one genuine merge interaction

`tests/ocean/unit/test_mpas_tke.py::TestNemoSurfaceTermsOnMPAS::test_eice3_quarter_ice_maps_to_full_attenuation`
passes on the lane parent, passes on main, and fails merged.  It is on the
MPAS OMIP card, not the GYRE NEMO path, and the GYRE trajectory above is
bit-identical, so it moves no certified number.

Resolved-configuration diff of the card the test builds
(`orca1_zdftke_config()`), the only row that differs:

| field | lane parent | merged (= main) |
|---|---|---|
| `tke_surface_bc_level` | `interior_pinned` | `nemo_z0` |

Main set that value deliberately (`run_omip_core2.py:911`); the lane parent
had the same line present but commented out as reverted.  The file auto-merged
and main's line won, which is the union rule working as written — the lane
expressed no preference in code.

The attribution is ONE VARIABLE and was measured, not argued.  Control first:
the unmodified card reproduces the failure (`1 failed`), with the two
diffusivity fields collapsed to the background floor `1.2e-05` everywhere so
the two ice regimes become indistinguishable.  Then the single row is forced
back to the lane parent's value through a plugin that changes nothing else:
`1 passed`.  Nothing else in the card or the code was touched between the two
runs.

**This is NOT decided here.**  Reverting main's default, or changing the lane's
closure so it responds to `nemo_z0` the way main's does, are both scientific
choices on the MPAS OMIP card and belong to its owner.  It is carried to
DECISION_NEEDED with the measurement above.

## Independent reviews

Two, run on the merge commit and its diff against both parents.

**Claude code-reviewer subagent — SHIP WITH CHANGES.**  No dropped hunk and no
duplicate definition in any of the six files (it diffed the top-level `def`
inventory against both parents); the shortwave precedence rule verified
unchanged; the fold branch confirmed unreachable on GYRE from
`fold_is_local`/`north_fold_mask`.  Its findings: the receipt for the
bit-identity claim did not exist yet (it does now, above); the sea-ice liquidus
override is a silent no-op on the layered branch; the sea-ice field order
changed; the fold-branch precedence is untested for a pivot mesh selecting
`nemo_avg`.

**codex `exec --sandbox read-only` — SHIP WITH CHANGES.**  Per-file: bulk flux
PASS, tripole PASS, the C-grid step PASS ("closed-box GYRE `nemo_avg4`
arithmetic is unchanged; main's pivot handling is isolated to pivot folds,
while `nemo_avg4` still refuses every tripolar fold"), the strict-coverage test
PASS; sea-ice config and sea-ice step CHANGE REQUIRED.  It printed the field
inventory of all three trees, which is what pinned finding 2 below.

Both reviewers independently raised the SAME two sea-ice defects, and neither
touches the GYRE trajectory.

| # | finding | disposition |
|---|---|---|
| 1 | the caller-supplied ocean freezing temperature lands on a configuration field the layered SI3 step never reads, so it is silently ignored there | **FIXED** in `d60d2cbcd`: that combination now refuses with a named error instead of accepting an argument it would ignore. Neither parent could express it (main has no layered branch; no card on this lane passes the argument). Routing the liquidus into SI3's bottom boundary is a physics decision for the sea-ice lane and is deliberately NOT taken in a merge. |
| 2 | the union moved main's `lead_freeze_source` from field index 48 to 52, changing main's positional layout | **FIXED** in `d60d2cbcd`: main's field keeps index 48 and the lane's four move one place later. Every `SeaIceConfig(...)` call site in the tree is keyword-only, so nothing is broken either way; the two "appended to preserve positional constructors" comments, which could not both be true, are replaced by one that says which claim survived and why. |
| 3 | the fold-branch precedence is untested for a pivot mesh selecting `nemo_avg` | **REGISTERED, not fixed.** Neither parent exercised that pair, and it is unreachable from this lane's cards for a reason that does not depend on any mesh's layout class: both NEMO cards select `nemo_avg4`, which now refuses on EVERY tripolar fold, pivot or not. The pivot branch is set only for de-haloed T-pivot meshes, which no card here was measured on. Carried to OPEN for the ORCA2 and eORCA025 lanes. |

The bit-identity measurement was already complete when both reviews returned;
finding 1 and 2's fix lands entirely outside the lat-lon ocean path, so it
cannot move the numbers above.

## Choices

Merge glue only.  No configuration, scheme, default, threshold or carried-state
change on any card; no default value moved anywhere.

UNASKED, and offered for revert — three, all fail-loud guards or ordering on
combinations that could not exist on either parent, the same class as the
2026-09-17 shortwave double-set guard:

1. The `nemo_avg4` refusal now fires above main's pivot-row fold branch as well
   as inside the NEMO-average one, so a pivot mesh selecting `nemo_avg4` raises
   instead of computing. Revert = let main's branch win there.
2. Passing an ocean freezing temperature together with the layered SI3
   thermodynamics now raises. Revert = accept and ignore it, as the merge
   originally did.
3. Main's sea-ice field keeps its index and the lane's four move; the reverse
   ordering is the alternative. No call site is positional either way.

## Verdict

**SHIP.**  The GYRE NEMO identity is proven bit-identical across the merge —
the certified 70-row ladder with zero rows moved and identical digests, all 360
daily snapshots byte-identical, and the day-30, day-240 and day-360 temperature
root mean squares reproducing round 163's certified values to every printed
digit.  The other cards' gates are unchanged at the same 170-test count.  The
one merge interaction found is on the MPAS OMIP card, moves no certified
number, and is carried to DECISION_NEEDED rather than decided.

## OPEN

- **DECISION_NEEDED, opened by this merge.** The ORCA1 MPAS OMIP card's TKE
  surface boundary-condition level is `nemo_z0` on main and was
  `interior_pinned` on the lane; the merge took main's value, and under the
  lane's own closure that flattens the MPAS diffusivity to the background
  floor, so one MPAS test that is green on BOTH parents is red merged. One
  variable, measured both ways (see the tests section). Two ways to close it,
  neither taken here because both are scientific choices on that card:
  (1) keep main's `nemo_z0` and make the lane's closure answer it the way
  main's does, or (2) pin the card back to `interior_pinned` on this lane
  until the closure is reconciled. My pick is (1), because `nemo_z0` is the
  NEMO-faithful boundary and the lane's whole purpose is NEMO fidelity — but
  it needs the MPAS OMIP owner, not a merge.
- **DECISION_NEEDED, carried forward, not opened here:** round 163's question
  of whether ORCA2 should take the second per-stage continuity solve once its
  own ladder is measured is still unanswered, and the merge does not touch it.
- Finding 3 above: a pivot-layout mesh selecting `nemo_avg` now takes main's
  branch rather than the lane's partner-cell formula. Unreachable from this
  lane's cards, which all select `nemo_avg4` and therefore refuse on any
  tripolar fold; the ORCA2 and eORCA025 lanes should measure that pair before
  relying on either branch.
- The day-gap scorer against `phase3/year_owners` still refuses at day 186 for
  a missing NEMO restart, on this tree and on round 163's alike. The certified
  year numbers come from the eight checkpoints against `phase3/year_fromrest`,
  which is the instrument round 163 used; the `year_owners` gap is pre-existing
  and untouched by the merge.
- Round 170's own order (ranking the tracer thickness at the new time level
  against the already-formed tracer content right-hand side) is unchanged by
  this merge.
