# GYRE identical-twin gradient receipt — 2026-09-21 pass 2

Status: **STOPPED — D48 CONFIRMED, D49/P5 CONFIRMED, P1 REFUTED.** The
mandatory direct adjoint gate still returns NaN for both parameter gradients,
so no optimizer update, recovery curve, identifiability scan, retune, GPU run,
or NEMO run was made.

The incoming GYRE lane tip was
`dda3f3257dad5a7f86e1177f934afc531e6567d9`. The five pass-1 commits from
`/tmp/gradtwin-2710410971` were fetched and cherry-picked before new work. The
clean measurement commit was `901cfe0d8752d29465949b00248d63cc46d3a9ec`
on `feat/gyre-gradient-twin-2`. Execution used CPU only, JAX x64, fp64/libm,
and `/home/dbalwada/legoESM/.venv/bin/python` with the requested `PYTHONPATH`.
Evidence is outside git under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradient_twin/pass2/`.

## D48 — reverse-safe TKE dissipation square root

The only new model-file change relative to the incoming lane is the guard at
the pass-1-localized post-`tke_avn` statement. The oracle statement is live in
the DINO override at
`cfgs/DINO/MY_SRC/zdftke.F90:832-837`: `zsqen = SQRT(en)` followed by
`dissl = zsqen / zmxld`. The legoESM guard uses the module's established
double-`where` idiom. It substitutes safe values for both operands on the
inactive `tke_curr <= 0` branch, then returns exactly zero there. No model or
configuration default moved.

The direct regression covers positive, exactly zero, smallest-positive
denormal, negative, and NaN TKE. For every input where the old expression is
finite, the new float64 primal has identical `uint64` bits. At zero TKE the
cotangents for both TKE and the dissipation-length divisor are finite `0.0`,
including the production-relevant `(TKE, divisor) = (0, 0)` case. With the
production guard reverted, the test failed (`1 failed in 2.06 s`); with the
final guard it passed (`1 passed in 1.84 s`). The retained logs are
`d48/unit_reverted_fail.log` and `d48/unit_guard_final_pass.log`.

### GYRE primal identity

Both arms were clean: before at `dda3f3257d`, after at `901cfe0d8`.

- The registered `--trajectory-only --max-step 10` ladder had 70 compared
  rows, zero differing rows, zero worsened cells, and maximum worsening
  `0 ULP`.
- All 210 arrays in the two `ladder.residuals.npz` files satisfy
  `np.array_equal`. Both files have SHA-256
  `0ca9f6a2e9af6350429cba95c5cc4c2b0a5a5bbc4f593633c8b5c3ac70b7b345`.
- Every `day001.npz` through `day030.npz` is byte-identical. The before and
  after members took 223.383011 s and 234.507696 s, respectively.

The exact records are `d48/ladder_comparison_final.json`,
`d48/exact_ladder_identity_final.json`, and
`d48/daily_snapshot_identity_final.json` in the evidence directory. D48's
GYRE primal-bit-identity verdict is **CONFIRMED**.

### DINO shared path

DINO and GYRE consume the same
`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py`. The finite old
branch is unchanged by construction and uses the same double-`where` idiom as
decision 42. The required DINO unit suite finished `128 passed, 9 warnings in
105.63 s`. The cheap canonical `fidelity_bar_gate.py` retained its expected
open-DEBT exit 1 and produced byte-identical stdout before/after, SHA-256
`e6a5ea00f4f91601a3e0079fe75c0bfec592024c0baf06e0e80abb4bc9bd736c`.
This is an unchanged gate result, not a claim that DINO's outstanding fidelity
debt is closed.

## D49 — differentiate the campaign program

The twin no longer uses `lax.scan`. It starts from the raw campaign state with
`bt_hist=None` and runs a Python loop. Each iteration preserves the campaign
order: construct forcing for one-based `kt`, perform the public-step shim's TKE
carry seed/cache prime, then call one checkpointed per-step JIT of the same
forward-Euler `_step_impl` branch with only `A_h` and `c_k` supplied as dynamic
operands. The differentiated loss is not wrapped in a whole-rollout JIT.

The structural control rejects a restored `lax.scan`, requires the raw cold
sentinel, and compares two parameterized calls against two literal public
`model.step` calls. T, S, u, v, and eta all had zero unequal bits.

P5 then ran from the clean measurement commit. The 60-step truth loop took
140.857772 s; the existing year-from-rest campaign member took 146.061538 s.
At day 10, all five saved fields were bit-identical:

| field | unequal / values | shared SHA-256 |
|---|---:|---|
| T | 0 / 21,120 | `86bd904a6ead8fe75ef41e51c9a7edc6d5cc8a8acb524dead83cb8f4cbc52e4b` |
| S | 0 / 21,120 | `2758c80fb6337889218d2c9cc7e676d7b415f44fc6d8d80e2106be92d46f5b7f` |
| u | 0 / 21,120 | `9e615ba14ff92e610508bf16c9940e3a76bb4c252743e8338ff6f77c706e3c5e` |
| v | 0 / 21,120 | `a20454932c113152e820cddb9547ecf1e4c2fbc74e1cf6778985e150e0c2871d` |
| ssh | 0 / 704 | `fd42bf60ef52c3fb434c22119bcad0e7e38467be862a4f73c69b6dd8daf8e007` |

That is zero unequal bits across 85,184 values, stronger than the preregistered
T-only P5. Peak RSS was 8,360.027 MiB. D49/P5 is **CONFIRMED**.

## Mandatory adjoint gate and P1

The four-test direct battery retained the independent reducer and planted
path-severing controls, confirmed both override routes, confirmed the two-step
primal identity above, and reached a finite positive start loss. The finalized
D48 scalar guard passed. The real two-step reverse path nevertheless returned
NaN for both selected leaves, so the required `assert_no_inert` gate failed:

```text
no-inert-parameters gate: non-finite gradient: ['A_h', 'c_k']
```

Finite differences were evaluated before the assertion:

| parameter | log step | reverse AD | central FD | AD/FD | verdict |
|---|---:|---:|---:|---:|---|
| `A_h` | `1e-3` | NaN | 0.0028727000523191826 | undefined | REFUTED |
| `A_h` | `1e-4` | NaN | 0.0028726975144499622 | undefined | REFUTED |
| `c_k` | `1e-3` | NaN | 0.044595334376873116 | undefined | REFUTED |
| `c_k` | `1e-4` | NaN | 0.04459535808389037 | undefined | REFUTED |

The cold reverse call took 1,765.160373 s and peaked at 39,391.355 MiB. XLA
reported 888.198345 s for its slow per-step transpose compilation; the
remaining 876.962028 s includes execution and any unreported compilation, so
it is not mislabeled as a clean steady-state gradient measurement. The full
battery ended `1 failed, 3 passed in 2118.05 s`. A bounded forward audit found
finite TKE, `tke_dissl`, `tke_avm`, and `tke_avt`; it rules out a forward
active-cell overflow but does not localize the next reverse-only singularity.

This direct test is a mandatory precondition. Therefore the registered
60-step P1 arms were not entered, all four registered P1 ratios remain
undefined, and P1 is **REFUTED**. The failing test remains unmarked: a later
repair must turn the same gate green.

## P1–P5 verdicts and recovery curves

| prediction | verdict | result |
|---|---|---|
| P1 reverse-mode FD | **REFUTED** | mandatory two-step gate: both AD numerators NaN; four finite FDs; registered 60-step ratios undefined |
| P2 `A_h` recovery | **UNMEASURED** | zero updates; P1 forbids optimization |
| P3 `c_k` recovery | **UNMEASURED** | zero updates; P1 forbids optimization |
| P4 whole-loss JIT/eager parity | **UNMEASURED** | P1 stopped the 60-step loss experiment |
| P5 campaign identity | **CONFIRMED** | 0 / 85,184 unequal day-10 values across T/S/u/v/ssh |

The `A_h`, `c_k`, and joint recovery curves are empty. The optimizer was not
invoked, and the 11-point scans were not run: those scans are registered only
for a recovery that stalls after a passing P1, not for an invalid adjoint. The
CPU 25-update fallback likewise cannot override P1. There is no defensible GPU
time estimate because no valid gradient or GPU timing datum exists; no GPU work
was started.

Machine-readable records are
`gyre_gradient_twin_2026-09-21_pass2_summary.json` and
`gyre_gradient_twin_2026-09-21_pass2_p5_summary.json`. No snapshot, residual
sidecar, or other multi-megabyte artifact is committed.

## OPEN

1. D48 makes the selected `dissl_new` operation reverse-safe but does not make
   the real two-step adjoint finite. The next first non-finite reverse operation
   must be localized before any additional physics edit or optimization.
2. After that repair, rerun the unchanged direct gate, then the registered
   60-step P1, P4, A_h, c_k, and joint arms in that order. Only a measured
   post-compile gradient may select the 25-update CPU fallback or support a GPU
   time estimate.
3. GitHub issue #1455 could not be read or updated: `gh` could not connect to
   `api.github.com` from this sandbox.
4. In-sandbox `codex review --commit 901cfe0d8` was attempted and failed to
   initialize its app-server client because the required path is read-only.
   This unit remains **UNREVIEWED** here; the operator's requested Claude
   review is required before citation or physics merge.
