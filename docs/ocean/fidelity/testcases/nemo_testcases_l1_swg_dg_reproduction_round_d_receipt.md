# NEMO SWG double-gyre reproduction package: Round D receipt

Date: 2026-10-13. Status: **LANDED**. Round D closes the six findings in the
second independent review of Round C. The implementation commit is
`4c252c4df7f629732b8d1e4e696dd3ac187fa41b`. No model physics, deck value,
acceptance bar, NEMO source, or external acquisition output changed. This
round ran no GPU, `makenemo`, or MPI command.

Work is in
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgdgsa-279125080`.
Round-D evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdgsa_rounds/round4`.
The deciding review log has SHA-256
`11fd1fd149a6689555239e101f2095ee9f6d4285c8c836554651f6f1a6c1bd3c`.

## Review disposition

| Review item | Disposition | Evidence | Reversion that makes the test fail |
|---|---|---|---|
| 1. Deck guard was not exhaustive | **FIXED.** The guard resolves every one of the 131 compiled-read inventory fields. It uses the resolved echo where present and the content-hashed `namelist_ref` then `namelist_cfg` precedence where the echo omits a group. It compares 16 additional case-specific echo values, giving 158 rows. The two-key run-control allowlist remains unchanged. The summary equality value is derived from the comparison rows. `ocean.output` does not print a machine-readable `nambdy` value on these runs; the report says `not_printed`. | Fixture tests pass physical `rn_tau`, unechoed controlling `ln_bdy`, missing-everywhere, and configuration-over-reference plants. Reference three-year, fresh smoke, and fresh three-year runs all print `SWG_DG_DECK_GUARD PASS compiled_reads=131/131 comparison_rows=158 allowed_differences=2`; report hashes are listed below. | Restore echo-only resolution or omit `nambdy.ln_bdy`: the unechoed-change and missing-field plants refuse. Stop comparing an echoed physical value: the `rn_tau` plant stops refusing. Reverse NEMO precedence: the override test fails. |
| 2. Python environment could not be created | **FIXED.** The guide creates conda-forge Python 3.13.0 build `h9ebbce0_101_cp313` before making the venv. The restricted requirements file now pins 40 packages, including `typing-extensions==4.15.0` and `certifi==2026.4.22`. | The environment test walks installed requirement metadata recursively and reports no required edge outside the pinned set. The guide-completeness test requires the creation command and both pins. | Delete either transitive pin: the closure test reports the parent-to-child missing edge. Restore the assumed `python3.13` instruction: the guide test cannot find the exact conda match specification. |
| 3. Exact conda lock omitted checksums | **FIXED.** The lock was regenerated with `conda list --explicit --md5`; all 122 package URLs carry 32-hex-digit MD5s. The URLs without hashes are byte-for-byte the reviewed reference set. | Lock SHA-256 `9aa5e72d56547edeaf27ad6514a58457d39940aa3eb09514b8526381e10c81f5`; unhashed-URL-list SHA-256 `deed23744c963e5c5787cd0731517a3aebd6475872a3c8938c3566eb0d80d56f`; committed syntax/content test. | Strip any MD5: the 122-line hash assertion fails. Change, add, or remove a package URL: the URL count or reference-list digest fails. |
| 4. Failure-path baseline was prose only | **FIXED.** A compact committed baseline contains all 30 per-step/per-field signatures for both outer-scan and materialized forms. The scorer compares two reports with that baseline and prints either `same as reference` or the first departing step and field. The guide gives complete CPU, GPU, materialized, baseline, stage-entry, and composition commands and reconciles the 189, 1,278, and 414,545 counts in one place. | Baseline SHA-256 `b5c2239e361d05d85a1d471633cf4386f884faada5ec891bafdf0df7268cbf9e`; actual reference-report comparison log SHA-256 `b3be1037083fae2436b8a87e5c041e61e36bd1b3a6da7ad0c69b14d5abdfd92a`; comparator test plants a changed step-4 V count and localizes it. | Delete the baseline or restore a prose-only comparison: the fixture test cannot load or compare it. Ignore a per-row signature: the planted step-4 V departure no longer localizes and the test fails. |
| 5. Test evidence boundary and counts were unclear | **FIXED.** The guide and this receipt name the exact 18-file battery, separate portable package guards from external campaign evidence, give the repository-only expectation, and bind every advertised count to a log. | Battery inventory and logs below. Round D: `369 passed, 1 skipped in 2290.25s`; log SHA-256 `e356391d7453b96aba10e6630a75eb62e9d5cf4e014d4872494cb77876a12bd4`. | Remove a file from the documented inventory, the `352 passed, 18 skipped` repository-only expectation, or the nine-step section set: the runbook-completeness test fails. |
| 6. Guide said eight steps and understated climate differences | **FIXED.** The guide has nine numbered steps. It states the measured maxima as kinetic energy `2.286230926795696e-11` and eddy kinetic energy `1.402270660699756e-11`, not “agrees to 1e-11.” | Runbook fixture test checks the exact ordered headings and the exact maxima sentence. | Restore eight headings or the old 1e-11 claim: the heading-order or forbidden-claim assertion fails. Remove the measured-maxima sentence: the exact-sentence assertion fails. |

## Exhaustive deck measurements

All three completed-run checks are **MEASURED** with the Round-D tool. Each
resolved 131 of 131 compiled reads, compared 158 rows, found only the two
registered output-control differences, and computed physical/numerical
equality as true.

| NEMO run | Result artifact SHA-256 | Command log SHA-256 |
|---|---|---|
| Original reference three-year run | `e5530e8a1d0d54001707b6c2a1de6dcc84bf93106273d02973b96364e04c0dd9` | `bcf8acdbb8f442bd52ec7f42c70586a1a6324d09662426c27674113dbad8bcd0` |
| Fresh operator 30-day smoke | `5fa9bc54fd1b9d28e3124dcb82f5818be7e502cb0b971370968072f4fffc4a17` | `bcf8acdbb8f442bd52ec7f42c70586a1a6324d09662426c27674113dbad8bcd0` |
| Fresh operator three-year run | `d5de92b71232822586de99947fb54310ba6b9b04921a6fa1b9d53e22cab47ad1` | `bcf8acdbb8f442bd52ec7f42c70586a1a6324d09662426c27674113dbad8bcd0` |

For all three, all 21 `nambdy` values came from the hashed `namelist_ref`;
`namelist_cfg` supplied no override. The committed configuration-over-reference
fixture proves the resolver applies an override when present.

## Test evidence boundary

The exact battery comprises these 18 files:

1. `tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py`
2. `tests/ocean/unit/test_nemo_tsunami_card.py`
3. `tests/ocean/fidelity/test_nemo_swg_check_records.py`
4. `tests/ocean/fidelity/test_nemo_swg_dg_climate.py`
5. `tests/ocean/fidelity/test_nemo_swg_dg_composition.py`
6. `tests/ocean/fidelity/test_nemo_swg_dg_restart_identity.py`
7. `tests/ocean/fidelity/test_nemo_swg_dg_round6.py`
8. `tests/ocean/fidelity/test_nemo_swg_dg_round7.py`
9. `tests/ocean/fidelity/test_nemo_swg_geometry_gate.py`
10. `tests/ocean/fidelity/test_nemo_swg_ladder.py`
11. `tests/ocean/fidelity/test_nemo_tsunami_ladder.py`
12. `tests/ocean/fidelity/test_swg_dg_reproduction_guards.py`
13. `tests/ocean/fidelity/test_swg_dg_reproduction_shell.py`
14. `tests/ocean/fidelity/test_swg_double_gyre_movie.py`
15. `tests/ocean/unit/test_nemo_swg_card.py`
16. `tests/ocean/unit/test_pv_flux_ene.py`
17. `tests/ocean/unit/test_swg_swe_rk3_stage.py`
18. `tests/ocean/validation/test_step_gradient_matrix.py`

The Round-C requested set collected 354 tests. Its split final evidence was
342 non-matrix passes in
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdgsa_rounds/round3/requested_cpu_battery_final_nonmatrix.log`
(SHA-256
`8d63a51d613d60234ad8e6d12d7a4b41ea690eb26eb373745a768298bd7bf7e7`),
plus 11 matrix passes and one registered skip retained from
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdgsa_rounds/round3/requested_cpu_battery.log`
(SHA-256
`d52b05bf28e86889885f93efb063b665e2be34763b63f4bf787d9c570222ccdf`):
353 passed and one skipped over the 354 tests. Both use the 18-file list
above.

The operator autopilot used the same list after twelve more tests were added:
366 collected, 365 passed, one skipped. Its log is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/autopilot_swg_dg_standalone/iter_202610131726_1/tests.log`
(SHA-256
`e294826b3692b094c44d0d946554cd157277c5ce2a3fac7c3a856f5468625f5d`),
and its command is in that iteration's `autopilot.sh`.

Round D added four tests. At clean implementation commit `4c252c4df`, the
same list collected 370 and ended exactly:

```text
================= 369 passed, 1 skipped in 2290.25s (0:38:10) ==================
```

The log is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdgsa_rounds/round4/full_lane_battery.log`,
with SHA-256
`e356391d7453b96aba10e6630a75eb62e9d5cf4e014d4872494cb77876a12bd4`.
The sole skip is the registered `gyre_nemo_card_6step` LLVM section-memory
case.

On a machine with only this repository, up to 18 cases visibly skip:

- four geometry cases when the external default mesh is absent;
- two double-gyre stage, three default-restart, five default-stage, and two
  default-month ladder cases when their admitted campaign roots are absent;
- one compiled-source inventory case when the preprocessed NEMO tree is
  absent; and
- the one unconditional registered six-step GYRE LLVM case.

Thus the documented repository-only result is 352 passed and 18 skipped out
of 370. The first 17 are external-evidence tests, not portable package guards.
All fixture-backed guard, shell, environment, baseline, and movie tests run.

## Fresh operator run state

The operator's read-only fresh CPU log now reports:

```text
STEP build exit=0
STEP case-identity exit=0
STEP rank-check exit=1
STEP nemo-smoke exit=0
STEP nemo-3yr exit=0
```

The known rank command failure occurred after both NEMO arms and was the
Round-C wrong-checkout import defect; its manual rescore remains
`all_bit_identical=True variables=21`, and the repaired wrapper has the decoy
subprocess coverage. The fresh three-year run completed step 1,892,160 with
`WALL_SECONDS=17404`; its `run.log` SHA-256 is
`145cabbb53e3223bd31599e728a1491add7a2f7e6a65a404cc027560c0f821fa`.
The CPU orchestration log SHA-256 is
`6bc52148d4e3deed0415c2402de3947b2a19d64314ac387d1f4e55d9dd3a6745`.

At the final audit, the separate operator-owned GPU log contained only its
queued manifest-wait line. This round neither starts nor modifies that job.
Its eventual acquisition is a continuation of the already queued Round-C
exercise, not new Round-D work.

## Carried exactness debts

The compact ladder baseline preserves, rather than relaxes, the reference
behavior: all 30 outer-scan rows exact; materialized first difference at
step-2 `ssh_m`, 189 words, maximum absolute difference
`1.3552527156068805e-20`. The separate formal stage-entry count remains 1,278
SSH words, the step-10 completed-state total remains 414,545 words, and the
640 signed-zero V-stress forcing words remain reported. No bar or exception
changed, and the existing user decision remains outside this round.

## PR description must say

- This is a stacked branch: the 58 replayed commits and the Round-A fidelity
  evidence remain as recorded; Round D changes only the reproduction package.
- The exact annual headline is end-of-day `ssh_m/u_m_s/v_m_s` versus
  `sshn[0]/un[0,0]/vn[0,0]` over T/U/V wet masks at days 365, 730, and 1095,
  not full-restart or dry-cell identity.
- The deck guard resolves 131 of 131 compiled-read fields and compares 158
  rows, including `nambdy` from hashed inputs where NEMO's echo omits it.
- The environment is reconstructable from a 122-URL MD5 conda lock, an exact
  conda-forge Python build, and a tested 40-package recursive Python closure.
- The failure path has a committed two-program baseline and a copy-paste GPU
  command. The 189, 1,278, and 414,545 counts are different comparison
  boundaries, not contradictory values.
- The nine-step guide reports climate maxima of
  `2.286230926795696e-11` for kinetic energy and
  `1.402270660699756e-11` for eddy kinetic energy.
- The final Round-D lane battery is 369 passed and one registered skip on the
  reference machine; a repository-only colleague should expect 352 passed
  and 18 visible skips because 17 tests require external campaign evidence.
