# NEMO SWG double-gyre standalone replay receipt

Date: 2026-10-13. Status: **OPEN** pending independent review, the
reproduction-package round, push, and pull request.

This receipt records the replay of the SWG gradient-gate and double-gyre
series onto the standalone shallow-water base. The base is
`5e6836d506486bb4b41e7efda2a1391a5cd7b82a`; the source tip is
`8f8551c8734947ff5b1489eb169f3cfd4cdfcaf2`; and the clean replay tip measured
here is `9a416fbd6d28b6faf7471631bdcbcd9ff74dd8f9`.

All measurements in this receipt were run from
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgdgsa-960324006`.
Evidence is outside Git under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdgsa_rounds/round1`.
No NEMO build, MPI launch, NEMO integration, or GPU command ran in this
round.

## Replay identity

The source range `292504bc1..origin/dg-lane-ref` and the standalone range
`origin/base-ref..9a416fbd6d` each contain **58 commits**. The first 24 are the
SWG gradient-gate prefix through the replay of `1d7fae766`; the final 34 are
the double-gyre series. The separately continued gradient branch is not in
this history.

A field-for-field comparison reports **58/58 exact** author name, author
email, author date, subject, body, and trailer blocks. Replayed messages and
trailers were not rewritten. The comparison is in `metadata_comparison.json`
(SHA-256
`4462cb834b5da47b90cfe0704760f1ff52f5ebec394f4e11edb451ff68ce3810`).

The default receipt citation gate ran after every replayed commit. Reports
exist for **58/58 current replay commits**, and **58/58 PASS** with no missing
or non-PASS report. The reports are in `per_commit_citation/`. This receipt's
own gate also reports PASS with zero extracted Oracle citations, zero unmapped
citations, zero map-audit failures, and all built-in plants firing.

## Conflict and citation reconciliation

The reconciliation rule is the one certified by the standalone base receipt:
retain the standalone base's map and source authorities, add exactly the
current source commit's change, and re-anchor only by line shifts introduced
by replayed commits. The only Git content conflict was:

| Source / replay | File | Kept | Dropped |
|---|---|---|---|
| `a113b73ac` / `d9c6836b6` | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` | All nine commit-owned additions: the SWG-DG P3D root, `phycst.F90` authority, SWG-DG P3D `stprk3` authority, and their two citation-map entries. The standalone base's qualified SWG survey authorities and disambiguating comment were retained. | The old stacked base's generic short-name comment/context. No line added by `a113b73ac` was dropped. |

Three earlier strict-citation repairs were placed at the first commit where
they became necessary so every historical replay point stayed green:

- `491244bce` adds nine lines to the ocean model. Its GYRE receipt and map
  anchors were shifted by nine in replay commit `a1a2c05db`; the later replay
  of `6cd4a900c` is an empty historical commit with its original metadata.
- `a6d3b91c9` adds three temporary recipe lines. Replay commit `94a86f518`
  temporarily shifts the seven affected recipe citation keys and the one GYRE
  receipt citation by three. Replay commit `d9771eec4` reverses those shifts
  when the source series removes the same three lines.
- The strict SWG-DG geometry key correction from the contiguous 211--221 form
  to the executed four-line 211, 213, 220--221 form was moved from
  `e9e5d92c4` into replay commit `64c47a1a4`, where the receipt first cites it.
  No source assertion was widened or removed.

These are citation timing/context changes only. No package resolution and no
lane-only symbol, helper, or fixture was required.

## Range-diff and package equivalence

The complete range-diff has **51 `=` pairs**, **4 `!` pairs**, and **3
source/replay `<`/`>` pairs**. The non-equal ordinals are fully explained by
the three early strict-citation repairs and the one map conflict above. The
context-only difference at ordinal 56 is inherited from the qualified
standalone authorities retained at ordinal 35. The full output is
`range_diff_58.txt` (SHA-256
`6d097e8cfe42c3857789d6fccae05aa1dde5892969df136fdfb9a8844323cc40`).

Exactly three files under `packages/` change. For each, the stable patch ID of
the source range equals the stable patch ID of the replay against its base:

| Package file | Source and replay stable patch ID | Result |
|---|---|---|
| `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` | `5e72f6a3364d127e0ace0b5cead8dfd6e1a2939d` | EQUAL |
| `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` | `b459b01d591421eb7dee7e10e37184d437e956a0` | EQUAL |
| `packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py` | `10a55786791585a4bdafa4ca53098e8a79868c6a` | EQUAL |

The table was generated as `package_patch_ids.tsv` (SHA-256
`648c7f4f2d9f869a78cfd46d916d4c0fa86a9d56ea0aaa1cbd02f08454ce9998`).
There is no unexplained package difference.

Reproduction:

```bash
git range-diff --no-dual-color \
  292504bc1..origin/dg-lane-ref origin/base-ref..9a416fbd6d
for f in $(git diff --name-only origin/base-ref 9a416fbd6d -- packages); do
  git diff 292504bc1 origin/dg-lane-ref -- "$f" | git patch-id --stable
  git diff origin/base-ref 9a416fbd6d -- "$f" | git patch-id --stable
done
```

## Card digests

The committed card digest helper was run independently in a detached source
worktree and in the clean replay. All six pairs are equal:

| Card/deck invocation | Source | Replay |
|---|---:|---:|
| shipped | `f9d46bfd08f23702` | `f9d46bfd08f23702` |
| Maddison nominal | `b2f60cd83f7d0a00` | `b2f60cd83f7d0a00` |
| Maddison ten-step | `8840b844a13c4921` | `8840b844a13c4921` |
| Maddison month | `5799106c86d9c69d` | `5799106c86d9c69d` |
| Maddison driver 365-day | `4f89aba9a6147da4` | `4f89aba9a6147da4` |
| Maddison driver 1095-day | `5a938efe6fcccdc6` | `5a938efe6fcccdc6` |

The source and replay JSON files have SHA-256
`885f6e63fc0fb0cdc981b4bd7fd8ad2894bfca3e4151ee32e02c35d7ae7486a4`
and
`ea3ab8ffd8d2142f439266340fe73ffa8c045de0a7731ed13a78b57971ed3429`,
respectively. Their `git_sha` stamps differ as expected; their digest mappings
are identical.

## Shipped-deck certification

All commands used CPU, explicit x64, and the production paths. The replay
reproduces the mounted shipped-deck state exactly:

| Arm | Result |
|---|---:|
| Given-entry stages | **338/338 AT-BAR and bit-identical** |
| Chained stages | **338/338 AT-BAR and bit-identical** |
| Independent from-rest ladder | **50/50 AT-BAR and bit-identical** |
| Carried-state audit | **29/29 AT-BAR and bit-identical** |
| Thirty-day ladder | **690/690 AT-BAR and bit-identical** |
| Recorded-stage one-fp64-ULP plant | **DEBT; exit 1; two changed SSH words** |
| Thirty-day one-fp64-ULP plant | **DEBT; exit 1; one changed SSH word** |

No bar was changed.

## Double-gyre certification and carried debt

After removing only the replay commit stamp, the five replay JSON products
(given, chained, month, stage plant, and month plant) are structurally equal
as parsed JSON to the source lane's round-4 products.

| Arm | Exact rows | Reported DEBT rows | Numerical result |
|---|---:|---:|---|
| Given-entry stages | 276/338 | 62/338 | 338/338 within the numerical bar |
| Chained stages | 113/338 | 225/338 | 338/338 within the numerical bar |
| Thirty-day ladder | 249/690 | 441/690 | 524/690 within the numerical bar |

The given and chained ladders first differ at step 2, stage 1 SSH: **1,278 wet
words**, maximum absolute residual
`3.3881317890172014e-21`. Their largest ten-step SSH residuals are
`1.0842021724855044e-19` and `2.168404344971009e-19`, respectively. These are
representation DEBT under the exact contract even though every ten-step row
is numerically within the bar.

The month result contains **90 signed-zero-only rows**. Its first nonzero
debt is day 1 restart SSH: **22,487 unequal words**, maximum absolute residual
`9.367506770274758e-17`. The first row beyond the numerical bar is day 7
restart SSH at `1.2281842209915794e-15`. The maximum month residual is day 30
restart U: `1.6431300764452317e-14`, normalized
`9.20175294902847e-15`. The 390 daily-entry rows split 69 exact / 321 debt;
the 300 restart rows split 180 exact / 120 debt.

Both double-gyre controls bind. The stage plant changes one nonzero wet U word
by one fp64 ULP and changes two SSH words by at most
`1.3234889800848443e-23`. The month plant changes one nonzero wet U word by
one fp64 ULP and changes two SSH words by
`1.3877787807814457e-17`. Both commands exit 1.

The replay result JSON SHA-256 values are:

- given: `4ad0f3629892e4234675d36e565d23c14d082d58216215d9e1cb080c5cd71155`;
- chained: `1573d34295ed9065db9c2132be308d2655741f9668886c34d0cb71339b26ac8a`;
- month: `512c575f9464f955a27b02571eb63a6cc9fb1d5f4f90eba63378a2b7dbc04a01`.

## Program-composition discriminator

After removing only `legoesm_git_sha`, this round's discriminator JSON equals
the source round-7 JSON exactly. It measures:

| Comparison | Exact | Signed-zero-only | Numerical debt |
|---|---:|---:|---:|
| next stage-1 record versus prior restart | 63/63 | 0 | 0 |
| certified chained stage 3 versus restart | 27/70 | 0 | 43 |
| production state versus restart | 64/100 | 10 | 26 |
| production stage 1 versus recorded stage 2 | 33/130 | 20 | 77 |
| production boundary versus recorded stage 1 | 40/130 | 20 | 70 |

The exact 63-row result is narrow: it maps `u_before/v_before` to restart
`un/vn`, after the next step's steady forcing update. It does not compare the
restart's `uu_n/vv_n` arrays and is not evidence of identical capture instants.

The certified materialized composition first has numerical restart debt at
step 2 SSH: 189 words, maximum absolute residual
`1.3552527156068805e-20`. The forcing representation debt is 640 signed-zero
words on the V-stress row for each affected field occurrence. The one-ULP
control changes two words by at most `1.3234889800848443e-23`, so it binds.
The JSON SHA-256 is
`0c1de1024f9fbc93b5b957fed4b3cacaf23ac90ec7029a68031dfdb57d82d3a1`.

This preserves, rather than repairs or relaxes, the source lane's result: the
formal materialized per-step composition has last-bit prognostic debt, while
the operational compiled-day scan has the distinct exact behavior reported
by the source lane. The scan's exact annual endpoints remain an uncited claim
until the reproduction package commits and runs the yearly restart checker.

## Commands for the fidelity rows

```bash
export PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/home/dbalwada/legoESM/.venv/bin/python
LAD=scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_ladder.py
DG_STAGE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round1/oracle_swg_dg_r1/instrumented
DG_MONTH=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round1/oracle_swg_dg_r1_30d
EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdgsa_rounds/round1

"$PY" "$LAD" --arm stages-given --deck maddison --stage-root "$DG_STAGE"
"$PY" "$LAD" --arm stages-chained --deck maddison --stage-root "$DG_STAGE"
"$PY" "$LAD" --arm stages-given --deck maddison --stage-root "$DG_STAGE" --plant ulp
"$PY" "$LAD" --arm month --deck maddison --month-root "$DG_MONTH"
"$PY" "$LAD" --arm month --deck maddison --month-root "$DG_MONTH" --plant ulp
"$PY" scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_round7.py \
  --root "$DG_STAGE" --json "$EVIDENCE/per_step_restart_discriminator.json"
```

The shipped arms used the same command and the default shipped roots, with
`stages-given`, `stages-chained`, `independent`, `carried-state`, and `month`,
plus both `--plant ulp` variants. Every JSON and log named above is in the
round evidence directory.

## Test battery

The requested CPU/x64 selection collected 311 tests and finished with
**310 passed, 1 registered skip** in 1995.72 seconds. It includes the receipt
gate, tsunami compatibility card and ladder, full step-gradient matrix, every
unit/fidelity file matching the requested SWG patterns, the five driver tests,
the seven climate-instrument tests, and the five direct round-7 tests. The
skip is the matrix's pre-existing six-step CPU GYRE row, whose test body
records the LLVM section-memory limit. The log SHA-256 is
`e316ead202b998a90c4771268e9a837af969e1309dce71cd2abc60dec498bb1f`.

```bash
"$PY" -m pytest -q \
  tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py \
  tests/ocean/unit/test_nemo_tsunami_card.py \
  tests/ocean/fidelity/test_nemo_tsunami_ladder.py \
  tests/ocean/validation/test_step_gradient_matrix.py \
  tests/ocean/fidelity/test_nemo_swg_check_records.py \
  tests/ocean/fidelity/test_nemo_swg_dg_climate.py \
  tests/ocean/fidelity/test_nemo_swg_dg_round6.py \
  tests/ocean/fidelity/test_nemo_swg_dg_round7.py \
  tests/ocean/fidelity/test_nemo_swg_geometry_gate.py \
  tests/ocean/fidelity/test_nemo_swg_ladder.py \
  tests/ocean/fidelity/test_run_swg_double_gyre.py \
  tests/ocean/unit/test_nemo_swg_card.py \
  tests/ocean/unit/test_swg_swe_rk3_stage.py
```

Passing direct tests do not prove untested behavior. In particular, the five
round-7 tests do not execute the real `run()` path, real restart assembly, or
the one-ULP plant; the separate real-data discriminator and plant commands
above are the evidence for those paths.

## OPEN review findings and next round

The source lane's last independent verdict remains **HOLD**. This replay does
not hide or mark fixed any of its six findings:

1. “Carried U/V” was overbroad; the 63-row mapping is narrowed above.
2. The year-1/year-2/year-3 identity claim has no committed citable comparator
   artifact. The reproduction package must add and run it.
3. The independent outer-jitted composition discriminator from review is not
   committed. The reproduction package must commit a discriminator before
   using that result in a decision.
4. The decision space must include pinning or reshaping the formal per-step
   XLA composition, not only carrying debt or certifying the daily scan.
5. A forcing-only signed-zero exception is not machine-enforced; any accepted
   exception needs a field allowlist and a planted prognostic violation.
6. Round-7 test coverage was overstated; the narrower scope is recorded in the
   test section above.

The exact bar remains unchanged. The operator must still decide whether the
640 forcing-only signed zeros count as equal and which composition is the
certified contract. My recommendation is exact comparison for prognostic
fields, a machine-enforced forcing-only signed-zero exception, and openly
carrying the formal materialized-composition debt while the operational daily
scan and possible XLA pin/reshape path remain separately identified.

Round B must provide the portable NEMO build/run tools, portable legoESM
driver, yearly identity reports, rank-count control, movie, environment pin,
runbook, and tested disposition of the review findings. Until that work and
independent review are complete, this receipt stays **OPEN** and the branch is
not represented as PR-ready.
