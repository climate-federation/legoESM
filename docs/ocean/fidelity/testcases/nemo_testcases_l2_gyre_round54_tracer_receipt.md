# GYRE round-54 step-2 tracer receipt

Date: 2026-09-11. CPU/fp64/scalar-libm. The authoritative artifacts are
`round54/step2_tracer_decomposition_v3.json` and
`round54/kt2_zdf_score_v4.json`. Both are clean-tree stamped at
`5bb67a1b847b`. Their producers mechanically retract decomposition v1/v2's
peak/TKE-lifetime defects and ZDF-score v1/v2's TKE-lifetime defect.

## Decomposition and preregistered owner

The kt=2 NEMO entry T/S/u/v/ssh inputs are bit-exact. T RMS by RK stage is
`8.373925232808e-13`, `3.201810863771e-8`, then
`4.154394627918e-4 K`; S is `5.193066126771e-14`,
`2.386123839629e-9`, then `5.993918044440e-5`. This locates the birth in the
stage-3 block, where compiled NEMO executes qsr, ldf, and finally tra_zdf
(`stprk3_stg.f90:924-958`).

Temperature squared-error shares are levels 0/1/2 =
`43.8391/47.9376/8.22334%`; levels >=3 total `<1.9e-5%`. Surface/interior/
bottom shares are `43.8391/56.1609/~0%`. South/north shares are
`60.7153/39.2847%`; west/middle/east are `49.0518/38.4190/12.5292%`. The
largest error is `8.741316411903e-3 K` at `(j=4,i=30,k=1)`. The frozen
three-level two-tracer redistribution signature preregistered tra_zdf first.

## Operator scores

| Operator/arm | T RMS K | S RMS | Verdict |
|---|---:|---:|---|
| adv+sbc through stage 2 | `3.202e-8` | `2.386e-9` | combined upper bound, not owner |
| K33 | K33 max debt `1.458e-10 m2/s` | same K | too small; hmlp still non-bit |
| ZDF content substitution | `4.13514e-4` | `5.97438e-5` | removes 0.464/0.326% |
| NEMO effective-K substitution | `4.22651e-6` | `2.38836e-7` | removes 98.983/99.602% |
| all NEMO ZDF operands | `5.99654e-13` | `9.56273e-13` | confirms owner |
| zero effective K | `1.86922e-2` | `4.13783e-4` | ablation worsens, influence only |

Given NEMO operands, the tracer matrix assembly, content RHS, and ordered
sweep are each bit-exact; NEMO's recorded sweep output is also bit-exact to
the kt=3 entry. Compiled `zdf_phy` copies closure `avt_k` then calls EVD
(`zdfphy.f90:347-359`); EVD replaces at `MIN(rn2,rn2b)<=-1e-12`
(`zdfevd.f90:107-110`). Its 136 `avt=100` interfaces agree within
`9.95e-14 m2/s`; stable interfaces carry the `7.7635e-3 m2/s` max debt.
The earliest existing non-bit boundary is therefore `zdf_tke` output: en RMS
`3.779e-7`, avm RMS `8.069e-6`, avt RMS `2.471e-4 m2/s`. Compiled NEMO forms
the TKE matrix/RHS at `zdftke.f90:409-426`, sweeps at `:458-475`, then forms
avm/avt at `:682-694`. Existing records skip these boundaries, so no first
statement is yet nameable and Rule 12 blocks a fix and cross-card claims.

The WRITE-only acquisition records entry operands, matrix/RHS, sweep result,
mixing lengths, Prandtl ratio, and output coefficients. It adds no physical
statement and removes zero shipped lines. Run only:

```bash
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round54_tke_operands/run.sh
```

ASKED: decomposition, causal scoring, and the required stopping acquisition.
UNASKED: none. No NEMO run/build, production physics change, GPU use, or
configuration/state choice was made. Open: first non-bit TKE statement; then
Rule-12 GYRE/tanks/DINO/ORCA2 adjudication and days 1-30 remain blocked on it.
