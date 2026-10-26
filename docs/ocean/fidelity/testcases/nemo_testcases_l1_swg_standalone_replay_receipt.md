# NEMO SWG level-1 standalone replay receipt

Date: 2026-10-13. Status: **OPEN** pending the operator's independent review,
push, and pull request.

This receipt records the standalone replay of the certified SWG lane onto
`origin/main-ref` at `2b986e2644`. All measurements below were run from the
writable clone
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgsa-2426228382`
and written outside Git under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgsa_rounds/round1`.
The clean measured commit was `a2b01422b`, after the 78-commit replay and the
standalone citation reconciliation.

## Replay identity

The source range is `1acdf3fad..292504bc`: exactly 78 commits. The standalone
replay is the first 78 commits of `origin/main-ref..a2b01422b^`, ending at
`7cd99657ca`. A field-for-field metadata comparison reports source count 78,
replay count 78, and **78/78 exact** author name, author email, author date,
subject, body, and trailer blocks. No replayed message or authorship was
rewritten. The reconciliation commit `a2b01422b` follows the replay.

The whole-series `git range-diff` has 55 `=` pairs, 19 `!` pairs containing
main-relative citation/context resolutions, and four `<`/`>` pairs where an
already-integrated strict citation fix made the historical fix commit empty or
patch-dissimilar. All four retain their source subject and metadata at the same
ordinal. The complete output is `range_diff_78.txt` in the evidence directory.

Reproduction:

```bash
git rev-list --count 1acdf3fad..292504bc
git rev-list --count origin/main-ref..a2b01422b^
git range-diff --no-dual-color \
  1acdf3fad..292504bc origin/main-ref..a2b01422b^
```

## Conflicts and resolutions

Every conflict followed one rule: preserve main's map and source locations,
then add exactly the current SWG commit's own change. Package changes retain
the certified source-range delta; inherited GYRE/VORTEX map context does not
come onto main.

| Source commit | File(s) | Resolution |
|---|---|---|
| 9b4527018 | `nemo_testcase_recipe.py`, citation map, round-1 receipt | Kept the commit's SWG relocation exactly, retained main's stricter disambiguated anchors, and kept main-relative shared citation positions. |
| 7862da6ab | citation map tail | Used the operator's mechanical resolver. Kept main plus the commit-owned final 27 lines beginning with the SWG round-2 block; dropped 120 inherited lines beginning with round 228 and ending in the unmerged VORTEX `dyn_zad` context. |
| fdde82e96 | `ocean_model_latlon_cgrid.py`, `eos.py`, `vertical.py`, citation map, round-3 receipt | Kept the certified code relocation and main's line positions. This restored the shared source anchors without importing the old base's offsets. |
| 78bfc7fb2 | citation map | Kept the commit's SWG P3 entries and vertical shifts. Replaced lane-only ocean-model keys 5779-5781 and 6513-6517 with main-relative 5771-5773 and 6505-6509. |
| 793909316 | `vertical.py`, citation map | Kept the certified line-neutral vertical relocation and main's restored keys 457-555, 773, and 1915. |
| 7c3dff5f7 | citation map, round-8 document | Kept the document split and strict endpoint anchors already integrated to keep earlier commits green; retained occurrence 1 for the meridional declaration. The following 8f0171572 fix remains as an empty historical commit. |

No other package resolution was needed, and no GYRE-only helper, fixture,
receipt, or symbol was required.

## Citation reconciliation

The default citation gate was run after each replayed commit that changed a
package source or the citation map: **34/34 relevant replay points PASS**, each
with zero unmapped citations, source failures, or full-map audit failures.
Their individual JSON reports and `per_commit_gate_summary.tsv` are in the
evidence directory.

The tip-wide audit then ran from the title of every SWG survey,
preregistration, and receipt. All twelve inherited SWG documents PASS. The
historical survey and round-1 documents required qualified source names because
their basename-only and bare continuation syntax could bind a statement to the
wrong same-basename file. The fail-closed map now pins every endpoint.

Line-bearing changes are listed literally below; old, invalid citations are
plain text so the receipt gate does not mistake them for current claims.

| Document or map | Old | New | Reason |
|---|---|---|---|
| GYRE round-8 receipt | ocean_pe_latlon_cgrid.py:5643-5644 | ocean_pe_latlon_cgrid.py:5655-5656 | SWG shared source insertion |
| GYRE round-8 receipt | ocean_pe_latlon_cgrid.py:5666 | ocean_pe_latlon_cgrid.py:5678 | same +12 shift |
| GYRE round-8 receipt | provenance.py:106 | provenance.py:149 | main moved the helper by one line; Round C later added 42 checkout-verification lines above it |
| GYRE round-8 receipt | ocean_pe_latlon_cgrid.py:3561-3564 | ocean_pe_latlon_cgrid.py:3568-3571 | SWG shared source insertion |
| GYRE round-8 receipt | ocean_pe_latlon_cgrid.py:3269-3295 | ocean_pe_latlon_cgrid.py:3276-3302 | SWG shared source insertion |
| conflict map | ocean_model_latlon_cgrid.py:5779-5781 | ocean_model_latlon_cgrid.py:5771-5773 | unmerged GYRE base had eight extra lines |
| conflict map | ocean_model_latlon_cgrid.py:6513-6517 | ocean_model_latlon_cgrid.py:6505-6509 | unmerged GYRE base had eight extra lines |
| SWG survey | usrdef_hgr.F90:80-146 | tests/SWG/MY_SRC/usrdef_hgr.F90:84-145 | qualify authority and pin executable coordinate/metric body |
| SWG survey | usrdef_zgr.F90:110-114 | tests/SWG/MY_SRC/usrdef_zgr.F90:143-147 | replace generic comments with the stated depth values |
| SWG survey | usrdef_zgr.F90:194-205 | tests/SWG/MY_SRC/usrdef_zgr.F90:195-206 | pin the complete wet-bound predicate |
| SWG survey | usrdef_zgr.F90:206-213 | tests/SWG/MY_SRC/usrdef_zgr.F90:214-222 | pin the actual lonely-corner rule |
| SWG survey | stprk3.f90:311-313 | SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:308-321 | cover swap, diagnostic, and restart statements rather than a context-bound fragment |
| SWG survey | ocean_pe_latlon_cgrid.py:2625 | ocean_pe_latlon_cgrid.py:2626-2630 | the replay adds `ens_total` and moves the validated branch |
| SWG survey | nemo_testcase_recipe.py:2894-3200 | nemo_testcase_recipe.py:2886-3200 | TSUNAMI standalone replay removed 31 old-base lines |

The remaining survey and round-1 changes only qualify basenames to their
existing compiled or `tests/SWG/MY_SRC` authorities; their numerical line
ranges do not move. The shifted-citation plant moves the round-8 stage-swap
citation `SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:308-321` by two lines, reports
`SYMBOL-NOT-AT-LINE`, and exits **1**.

Reproduction:

```bash
export PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/home/dbalwada/legoESM/.venv/bin/python
GATE=scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py
for f in docs/ocean/fidelity/testcases/*swg*.md; do
  "$PY" "$GATE" --receipt "$f" --from-heading "$(sed -n '1p' "$f")"
done
"$PY" "$GATE" \
  --receipt docs/ocean/fidelity/testcases/nemo_testcases_l1_swg_round8_multiyear_preregistration.md \
  --from-heading '# NEMO SWG level-1 round 8 — shipped multi-year rung' \
  --plant 'SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:308-321'
```

## Package equivalence and diff disposition

For every changed package file, the stable patch ID of
`git diff 1acdf3fad 292504bc -- <file>` equals the stable patch ID of
`git diff origin/main-ref HEAD -- <file>`. Thus every package delta is the
certified SWG delta, not a cleanly-applied but semantically different hunk.

| Package file | Disposition |
|---|---|
| `latlon_cgrid_operators.py` | Exact source/replay patch ID; ENS and SWG operator work |
| `ocean_model_latlon_cgrid.py` | Exact source/replay patch ID; base gap equals tip gap, preserving main's absence of unmerged tracer-diagnostic work |
| `ocean_pe_latlon_cgrid.py` | Exact source/replay patch ID; SWG production routing and `ens_total` |
| `swe_rk3.py` | Exact new-file patch ID; SWG shallow-water RK3 program |
| `eos.py` | Exact source/replay patch ID; SWG certified shared change |
| `nemo_testcase_recipe.py` | Exact source/replay patch ID; base gap equals tip gap, preserving main's absence of GYRE seamount SMT-4 work; additions are SWG card/validation only |
| `state.py` | Exact source/replay patch ID; SWG carried state |
| `vertical.py` | Exact source/replay patch ID; SWG QCO options with restored shared anchors |

The package/script stat has 21 paths: these eight certified package files,
twelve SWG-named validation files, and the shared citation map. The package
files are expected because the lane implements SWG in shared dynamics; they
are not standalone-resolution additions. The only non-SWG script is the
citation map. Shared tests `test_nemo_ldf_lap_viscosity.py` and
`test_pv_flux_ene.py` are replayed SWG discriminators. The only non-SWG
document change is the five source-line shifts in the GYRE receipt tabulated
above. No defect or unexplained path remains.

Reproduction:

```bash
git diff origin/main-ref HEAD --stat -- packages scripts
git range-diff 1acdf3fad..292504bc origin/main-ref..a2b01422b^
for f in $(git diff --name-only origin/main-ref HEAD -- packages); do
  git diff 1acdf3fad 292504bc -- "$f" | git patch-id --stable
  git diff origin/main-ref HEAD -- "$f" | git patch-id --stable
done
```

## Cross-card digests

The production card digest helper was run under explicit fp64 in both the
standalone tree and a detached `origin/main-ref` worktree. Every pair is equal:

| Card | main | standalone |
|---|---:|---:|
| TSUNAMI | `8bb5ed6c1bc031bd` | `8bb5ed6c1bc031bd` |
| VORTEX | `f7489467bc156303` | `f7489467bc156303` |
| VORTEX_VEC | `5041b91c67bd1443` | `5041b91c67bd1443` |
| GYRE | `f02d2f42a570f413` | `f02d2f42a570f413` |

The earlier expectation that main's GYRE digest could not be
`f02d2f42a570f413` is **REFUTED at the moved main `2b986e2644`**: the digest was
computed independently in the detached main worktree and is not copied from
the old GYRE lane. The replay does not change any of these four cards.

## Fidelity ladders and controls

All commands used the required CPU/x64 environment above and ordinary jitted
production paths. No NEMO executable, MPI launcher, or GPU was used.

| Arm | Result |
|---|---:|
| Given-entry stage replay | **338/338 AT-BAR, 338/338 bit-identical** |
| Chained stage replay | **338/338 AT-BAR, 338/338 bit-identical** |
| Independent from-rest ten-step ladder | **50/50 AT-BAR, 50/50 bit-identical** |
| Carried-state audit | **29/29 AT-BAR, 29/29 bit-identical** |
| Independent thirty-day ladder | **690/690 AT-BAR, 690/690 bit-identical** |
| Recorded stage-entry one-fp64-ULP plant | **DEBT, exit 1** |
| Thirty-day one-fp64-ULP plant | **DEBT, exit 1** |

The optional shipped scorer also ran. It completed all 172,800 one-hour
production steps: **1080/1080 output rows** (360 output times times SSH/U/V)
and **10/10 terminal restart rows** are AT-BAR and bit-identical. All three
first-over-bar fields are `None`. Timed execution excluding JIT warm-up was
2169.0254190154374 seconds at 79.66711615506897 model steps per second. The
JSON SHA-256 is
`31e3d41136c973791886dd48c9b4d112bbe5742c5a967cdea82b3261fa5561f7`.

Reproduction:

```bash
LAD=scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_ladder.py
STAGE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round1/oracle_swg_r1/run
TEN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round0_probe/kt10
MONTH=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round6/oracle_swg_r6_30d
SHIPPED=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round0_probe/year10
"$PY" "$LAD" --arm stages-given --stage-root "$STAGE"
"$PY" "$LAD" --arm stages-chained --stage-root "$STAGE"
"$PY" "$LAD" --arm independent --root "$TEN"
"$PY" "$LAD" --arm carried-state --stage-root "$STAGE"
"$PY" "$LAD" --arm stages-given --stage-root "$STAGE" --plant ulp
"$PY" "$LAD" --arm month --month-root "$MONTH"
"$PY" "$LAD" --arm month --month-root "$MONTH" --plant ulp
"$PY" "$LAD" --arm shipped --shipped-root "$SHIPPED"
```

## Test battery

The requested CPU/x64 battery collected and passed **258/258** tests in
283.48 seconds:

```bash
"$PY" -m pytest -q \
  tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py \
  tests/ocean/unit/test_nemo_tsunami_card.py \
  tests/ocean/fidelity/test_nemo_tsunami_ladder.py \
  tests/ocean/fidelity/test_nemo_swg_check_records.py \
  tests/ocean/fidelity/test_nemo_swg_geometry_gate.py \
  tests/ocean/fidelity/test_nemo_swg_ladder.py \
  tests/ocean/unit/test_nemo_swg_card.py \
  tests/ocean/unit/test_swg_swe_rk3_stage.py
```

The log SHA-256 is
`e49184f1bae4b09c7f2d351436c650d25acaec0e30fc59cc99201773cced47ae`.

## OPEN

1. The operator runs the independent highest-effort review of the range-diff,
   all conflict resolutions, citation audit, and fidelity ladders.
2. After review, the operator pushes this standalone branch and opens the
   approved pull request against main.
