# GYRE NEMO-fidelity merge-main receipt — 2026-09-17

## Scope and provenance

This receipt covers merge commit `9f69578592d9ac20f4abef527c989fa3ae9704c7`,
whose parents are lane tip `f02548a28658b760bfa6f86193e8ec5ea0b4416a` and
GitHub `main` `9f4b16d633f0adb52084b84646283238927baa12` (90 commits since the
previous merge base `946351212fb2`), plus two follow-up commits: `d491e264ccb7` re-anchors four lane-source
citations the merge shifted, and `8109f8c2c412` restores main's Sweeney
surface-deposit line after the first review pass (below).
Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/merge_main_2026-09-17/`
(`before/` = lane tip, `after/` = merge commit, `after2/` = final code commit
`8109f8c2c412`, `tests/`, `codex_review.log`, `codex_review2.log`).

## Textual conflicts

Both conflicts sit on the external-forcing shortwave selector of the lat-lon
C-grid step, `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py`
(`_bc_external_surface_forcing` signature and its one call site) and
`packages/ocean/legoesm/ocean/physics/shortwave_penetration.py`
(`ShortwavePenetrationConfig` docstring, one error string).

| Side | What it added on that seam |
|---|---|
| main (`ff325d0e6`, `5431ba645`) | FESOM2 `sweeney_2band` chlorophyll two-band and an explicit `jerlov_2band` water-type route, selected through `surface_forcing.shortwave_scheme` (default `"auto"` = legacy fallthrough); validation tuple of three names; `shortwave_water_type` kwarg. |
| lane | NEMO's `nemo_qsr_2bd` / `nemo_qsr_rgb` identities, selected through `physics.shortwave_penetration.scheme` (the shared pipeline consumes that same field), `_shortwave_surface_composition` (full qsr for NEMO, 0.94 skin otherwise), and a configured heat capacity `c_sw`. |

Resolution: the union.  The signature is
`shortwave_scheme="auto", shortwave_water_type="II", c_sw=None`; the
validation tuple and `SHORTWAVE_PENETRATION_SCHEMES` carry all five names;
main's two branches and the lane's two branches are kept verbatim.  At the call
site the NEMO identity on `physics.shortwave_penetration.scheme` is routed into
`shortwave_scheme`; every other value comes from
`surface_forcing.shortwave_scheme`; a card that sets both refuses with a named
error instead of taking a silent precedence.  Main's Sweeney branch is byte-identical to main's (the merge commit had
switched its surface divisor to the configured `c_sw`; the first review pass
showed that split the branch's column total between two capacities, and
`8109f8c2c412` restored main's line).  The configured `c_sw` reaches, as on the
lane, only the branches the lane wrote.

No NEMO-transcribed statement was touched.  The compiled statement the lane's
NEMO branch reproduces is unchanged: `traqsr.F90:665-712` consumes the complete
`qsr` after `trasbc.F90:299-315` removes it from `qns`, which is what
`_shortwave_surface_composition(..., "nemo_qsr_2bd", ...)` returns (fraction
1.0) and what the GYRE card selects
(`fidelity/nemo_testcase_recipe.py:188`, `scheme="nemo_qsr_2bd"`).

## Auto-merged overlap audit

Main-side hunks in the four remaining both-side files, read from
`git diff 946351212fb2 9f4b16d633f0 -- <file>` and checked against the GYRE
NEMO-identity card:

| File | Main's hunk | GYRE-card execution verdict |
|---|---|---|
| `ocean_tendency_common.py` | zero-safe `sqrt` in `nemo_drag_r_from_speed_sq` (`where(arg>0, sqrt(arg), 0)`) | Executes; bit-inert — where `arg > 0` the value is the same `sqrt`, and `sqrt(0) == 0`. |
| `vertical_mixing/config.py` | new `TKEConfig.surface_flux_coeff` (default `1.0`, FESOM2 card sets 3.75) + spec | Config construction only; the card does not set it. |
| `vertical_mixing/tke.py` | `surface_flux = cfg.surface_flux_coeff * (taum/rho_0)**1.5` on three lines | Multiplication by exactly `1.0` is exact in IEEE-754; bit-inert whether or not the line executes on the card. |
| `state.py` | `LateralViscosityConfig.B_h_gamma0` (default `0.0` = off) | Off. |

Everything else main changed (scripts/cluster, bench, atmosphere physics,
training, config/wb, WOA fill, tripole/MPAS lanes) does not touch the lat-lon
GYRE path.  The trajectory proof below is the gate on this audit.

## Trajectory proof

All arms were run from CLEAN, COMMITTED trees (no `LEGOESM_GATE_ALLOW_DIRTY`
escape): `before/` at `f02548a28658` in `/tmp/gyre-before-f025`; `after/` at
the merge commit `9f69578592d9` and `after2/` at the final code commit
`8109f8c2c412`, both in `/tmp/autopilot-work-mergemain2`; the `worktree` stamps
in the three `ladder.json` files record `clean: true` and those commits.  The
table below holds for `after/` AND for `after2/` against `before/` (both
comparisons were run; the `after2/ladder_comparison.json` is the one that
covers the pushed code).  Commands
(per arm): the ladder gate `--trajectory-only --max-step 10 --output
ladder.json`; the day-gap member `--member 0 --days 30 --snap-steps 6 --tag
daily`; scoring `nemo_testcase_l2_gyre_year_owners.py --day-gap` for days
1-30.

| Comparison | Result |
|---|---|
| `ladder.json` `steps` (10 blocks, 50 rows) | identical (`==` on the parsed documents) |
| `ladder.json` `barotropic_state_steps` | identical |
| `first_over_bar` | identical: `{'fields': ['u','v'], 'kt': 2}` |
| `ladder.residuals.npz` | 210 arrays, 0 unequal (`np.array_equal`) |
| day-gap member snapshots `day001..day030.npz` | 30 of 30 byte-identical (sha256); only `manifest.json` (provenance) differs |
| `day_gap.json` `rows` (30 days) | identical; the only differing key is `worktree` |
| `nemo_testcase_offline_compare.py before after` | `OFFLINE_ORACLE_RELATIVE_COMPARE PASS: rows=70 max_worsening_ulps=0`, `violations: []`; its `--plant worsen-3ulp` exits 1 with `FAIL ... max_worsening_ulps=3` |

Headline rows (identical on both arms, so quoted once):

| Row | value |
|---|---:|
| kt2 U max absolute residual | `2.7377110452773967e-12` |
| kt2 V max absolute residual | `3.2849219221489645e-12` |
| kt3 T max absolute residual | `1.627497246303733e-04` K |
| kt3 S max absolute residual | `6.327735185607253e-06` |
| day-30 T RMS | `1.2397011296352737e-02` K |

The kt2 V and day-30 values differ from the 2026-09-16 merge receipt in their
last digits because the decision-42 card landing (`e92f8aff`) moved them after
that receipt; the lane tip's own numbers are the ones above and the merged
tree reproduces them bit-for-bit.

## Citation gate

Main's insertions into `ocean_pe_latlon_cgrid.py` (+2 lines before the
horizontal-viscosity grid guard at 3208, +132 more between 3535 and 4341)
shifted four citations that pin lane source lines; the gate reported
`SYMBOL-NOT-AT-LINE` for each with the identified line.  Each was re-anchored by
a RIGID shift after verifying the cited lines are textually identical between
the lane parent and the merged tree:

| citation (lane tip) | delta | citation (merged) | extent |
|---|---:|---|---:|
| `ocean_pe_latlon_cgrid.py:3276-3279` | +2 | `:3278-3281` | 4 lines |
| `ocean_pe_latlon_cgrid.py:4762-4783` | +134 | `:4896-4917` | 22 lines |
| `ocean_pe_latlon_cgrid.py:5077-5078` | +134 | `:5211-5212` | 2 lines |
| `ocean_pe_latlon_cgrid.py:5100` | +134 | `:5234` | 1 line |

Applied to `CITATION_MAP` (4 keys) and to the round-8 receipt (3 quotes;
`4762-4783` is a map-only entry).  No NEMO compiled-source citation moved (main
does not touch the oracle builds).  After the re-anchor commit:
`tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py` — `16 passed`
(it was `3 failed, 13 passed` on the merged tree before the re-anchor and
`16 passed` at the lane tip); the gate CLI reports `status PASS`, 274 citations,
0 unmapped, 0 failures, and all nine self-test plants `fired: True`.

## Focused and broad tests

All logs under `tests/` in the evidence directory; every summary line is
pytest's own last line (the shell exit code here is not evidence).

Focused, one process per file, on the merged tree: 13 files
(`test_nemo_testcase_receipt_citation_gate`, `test_tke_nemo_terms`,
`test_nemo_recipe`, `test_real_freshwater_closure`, `test_leapfrog_integrator`,
`test_nemo_testcase_recipe`, `test_shortwave_scheme_routing`,
`test_shortwave_sweeney`, `test_rgb_chl_penetration`,
`test_shortwave_penetration_jerlov_faithful`, `test_biharmonic_gamma0`,
`test_tke_fesom2_card`, `test_nemo_testcase_l2_gyre_phase3_gate`) — all
`passed` (`focused_summary.txt`); the citation gate's `3 failed, 13 passed`
in that battery is the pre-re-anchor run, re-run after `d491e264ccb7` as
`16 passed` (`citation_after_reanchor.log`).

Broad, `tests/ocean/fidelity tests/ocean/unit tests/unit/test_run_omip_cli.py`
with `-n 12` on the merged tree:
`121 failed, 7927 passed, 143 skipped, 2 xfailed, 75 warnings, 16 errors in
3161.99s` (`broad_ocean.log`).  Two commits (`d491e264ccb7`, `8109f8c2c412`)
landed while it ran, so tests that stamp the worktree could have seen a dirty
tree; the 137 flagged IDs (`broad_failing_ids.txt`) were therefore re-run on
STABLE trees, with worker isolation (`-n 8`; a single process aborts in the JAX
compiler inside `test_scm_column_twins.py` on both trees, exit 134):

| tree | commit | re-run summary | failing IDs |
|---|---|---|---:|
| merged | `8109f8c2c412` | `71 failed, 50 passed, 10 warnings, 16 errors in 661.64s (0:11:01)` | 87 |
| lane tip | `f02548a28658` | `71 failed, 50 passed, 10 warnings, 16 errors in 583.09s (0:09:43)` | 87 |

The two failing-ID sets are IDENTICAL (`comm` on `rerun_merged_ids.txt` /
`rerun_lanetip_ids.txt`: 0 merged-only, 0 lane-only, 87 shared; list kept as
`preexisting_red_ids_on_both_trees.txt`).  The other 50 flagged IDs pass on
both stable trees.  So the merge attributes ZERO test failures; the 87 are
pre-existing lane debt in `tests/ocean/unit` (plus the known worktree-stamp
ratchet in `tests/ocean/fidelity`) that the four-file autopilot gate never
ran.  They are reported here, not chased.

## Independent review

Two read-only `codex exec` passes (`codex_review.log`, `codex_review2.log`).

Pass 1, on the merge commit `9f69578592d9`, verbatim:
`VERDICT: DO NOT SHIP — the merged selector silently permits duplicate solar
heating, NEMO RGB cannot traverse the shared pipeline, and configured heat
capacity is applied inconsistently.`  Its four findings and their disposition,
each checked against the parents by `git diff`:

| # | Finding | Disposition |
|---|---|---|
| F1 BLOCKER | `physics.shortwave_penetration.scheme="jerlov_2band"` + `surface_forcing.shortwave_scheme="jerlov_2band"` + `sw_down` deposits solar twice (pipeline `combined.py:336-352` and the external stage). | PRE-EXISTING ON MAIN: `combined.py` is byte-identical between the merge base and main; the surface selector and the explicit Jerlov branch are main's own commits. The merge guard covers the one combination neither parent could express (NEMO identity + non-auto surface selector). Reported, not widened, not fixed here. |
| F2 HIGH | `nemo_qsr_rgb` + `sw_down` raises in the pipeline's two-band-only kernel. | PRE-EXISTING ON THE LANE (kernel and name both predate the merge; main did not touch either). Reported. |
| F3 MEDIUM | Sweeney surface divisor on configured `c_sw`, its penetrative kernel on the module default. | ACCEPTED: `8109f8c2c412` restores main's line. |
| F4 LOW | `penetrating_fraction` rejects the NEMO names that `SHORTWAVE_PENETRATION_SCHEMES` lists. | Fail-loud and unreachable through the merged routing (NEMO names reach only the lane's branches, which do not call it). Left. |

Pass 2, on `8109f8c2c412`, asked to refute each disposition, verbatim:
`F1: STANDS AS CLASSIFIED` / `F2: STANDS AS CLASSIFIED` / `F3: STANDS AS
CLASSIFIED` / `F4: STANDS AS CLASSIFIED` / `d491e264ccb7: VERIFIED — four
same-extent legoESM citation shifts only; no NEMO compiled-source citation
moved.` / `NEW: none — F1/F2 pre-exist on a parent and were not widened by the
merge.` / `VERDICT: SHIP`.

## Choices

- Merge glue only; no configuration, scheme, default, threshold or carried-state
  change on any card.  UNASKED and offered for revert: a card that sets both
  shortwave selectors (NEMO identity on `physics.shortwave_penetration.scheme`
  and a non-`auto` `surface_forcing.shortwave_scheme`) raises instead of one
  silently winning — a combination that could not exist on either parent.

## OPEN

Round 105 as ordered in standing-brief note S/T: the zdfsh2 routing split, its
local proof under NEMO's recorded entry (jit and eager), the Rule-12 ladder
against `after/` here as the before arm, and a MEASURED DINO row.
