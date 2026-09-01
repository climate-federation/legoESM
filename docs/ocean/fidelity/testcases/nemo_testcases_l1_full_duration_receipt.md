# NEMO testcase lane 1: full-duration statistical receipt

Status: **LOCK_EXCHANGE-zco classified; OVERFLOW-zps OUTSIDE because the
certified legoESM card does not reach the registered midpoint or full
duration.** This is a post-PR lane-1 extension. It does not revise the shipped
geometry/kt60 certification.

## Outcome

| case | required legoESM duration | fp64 / fp32 execution | statistical result |
|---|---:|---|---|
| LOCK_EXCHANGE-zco | 61,200 steps | COMPLETE / COMPLETE | 0 `INDISTINGUISHABLE-AT-FLOOR`, 6 `WITHIN-SCHEME-SPREAD`, 0 `OUTSIDE` |
| OVERFLOW-zps | 6,120 steps | first non-finite after step 2,877 / 2,879 | 0 `INDISTINGUISHABLE-AT-FLOOR`, 0 `WITHIN-SCHEME-SPREAD`, 6 `OUTSIDE` |

The OVERFLOW result is deliberately not softened to “incomplete” in the
metric ledger. The preregistration says a missing or non-finite registered
input is `OUTSIDE`; all six registered rows therefore carry that exact verdict
and null distances. No endpoint section, plume statistic, scheme-spread claim,
or fp32 floor is inferred from the two-step difference between its failure
times.

LOCK does reach the statistics class. Every registered distance is above its
fp32-vs-fp64 floor but below the NEMO FCT2-vs-FCT4 scheme spread. Thus all six
rows are `WITHIN-SCHEME-SPREAD`; none is claimed indistinguishable at the
roundoff floor.

## Preregistration and reconciliation chronology

The initial protocol was committed at
`76abced8bf994a6db348ef792e975dbf988def7f`, before launching either NEMO N4
arm or computing any legoESM metric. Commit
`5dd9a7ee64f6dcbfb5bcdc3b55c81fd47f56da63` made the already-known phase-1
endpoint excess explicit. Before computing any L64/L32 metric, a scorer dry
run against N2 alone then proved the proposed endpoint hard failure
self-refuting: OVERFLOW N2 ends at `20.00000000000307 C`, above the proposed
`sqrt(N)*eps` snap floor. Commit
`27e569b2093229c2d390658fca78bbc7f46fa488` reconciled the instrument with the
certified phase-1 result: the sqrt floor remains a loud `UNMEASURED`
classification, the `1e-6` relative gross-excursion hard failure remains, and
only histogram/census endpoint membership is snapped. No statistical verdict
or distance was changed post hoc.

The four arms contain no Frankenstein combination:

- N2 is each executed NEMO FCT2 namelist.
- N4 is the shipped NEMO FCT4 reference arm, with a fail-closed semantic diff
  allowing only `cn_exp` and `(nn_fct_h,nn_fct_v): (2,2)->(4,4)`.
- L64 and L32 are the identical certified legoESM scheme identity; only the
  explicit precision policy and JAX x64 state differ.

The reference and executed source citations for both FCT arms are frozen in
the preregistration: OVERFLOW shipped namelists at
`tests/OVERFLOW/EXPREF/namelist_zps_FCT{2,4}_flux_ubs_cfg:64-70`, LOCK at
`tests/LOCK_EXCHANGE/EXPREF/namelist_FCT{2,4}_flux_ubs_cfg:63-68`, and active
horizontal/vertical order dispatch at
`src/OCE/TRA/traadv_fct.F90:183-278`. Both arms retain the executed RK3
two-step predictor at `traadv_fct.F90:153-161`.

## Execution receipt

The compile-warmed 200-step CPU timing probe projected 2,080.75 s for LOCK and
462.40 s for OVERFLOW, both below the preregistered 3,600 s GPU threshold. All
production runs therefore used `JAX_PLATFORMS=cpu`; no GPU and no mixed-backend
segment was used. The dtype gate printed all state and vertical-coordinate
arrays as float64 under `PrecisionPolicy.fp64()` and float32 under
`PrecisionPolicy.fp32()` with JAX x64 disabled.

| arm | completion / wall time | saved artifact SHA256 | run-code commit |
|---|---|---|---|
| LOCK L64 | 61,200 / 737.153 s | states `d599d3e0e39de99ad050229c2ec62ff48b4332495913d2b92ee61daa50c15a0a` | `09840d1ca72e604c6445fb55f69a10f1a24e94e2` |
| LOCK L32 | 61,200 / 632.131 s | states `27ff889c775424143604dc80c1214862253a563d78d48093354921bec3d63f29` | `09840d1ca72e604c6445fb55f69a10f1a24e94e2` |
| OVERFLOW L64 | fails at 2,877 / 261.813 s | failure `4640a3bc79604a2e5abcae0245d8d1c60e3821771995f96d1cb3f97b891c51ab` | `914fec1a5684a567f6d98667e9931553b21f178f` |
| OVERFLOW L32 | fails at 2,879 / 175.836 s | failure `c940c1a4253271cc7d5c1294a3b50fad0519a049c2dd05a3e75aefd2ae0c729d` | `914fec1a5684a567f6d98667e9931553b21f178f` |

The OVERFLOW per-step guard reports fp64 T/S/u/v non-finite while SSH remains
finite at `t=28,770 s`; fp32 reports all five fields non-finite at
`t=28,790 s`. These are committed-harness results, not the earlier throwaway
localization probe.

Both NEMO FCT4 controls used the certified CPU binaries without `mpirun` and
completed: LOCK `time.step=61200`, 304.807 s timed stepping; OVERFLOW
`time.step=6120`, 371.823 s. Their final restart hashes are respectively
`e7250a62a9104f82dd213fd1bd76f3d3fe359fffcadf8d42810865a3dc8ce080`
and `9d6ddc316c1c752a4d6bc3c6163cf0e470c1c6143eaa2d5e064795c32b9f5b37`.

## LOCK_EXCHANGE-zco scores

Distances follow the frozen rule `D=distance(L64,N2)`,
`F=distance(L32,L64)`, `S=distance(N4,N2)`. Equality is inclusive.

| metric | D | F | S | exact verdict |
|---|---:|---:|---:|---|
| front position curve (km) | `2.1844058e-2` | `7.4034378e-4` | `3.2017657e-2` | `WITHIN-SCHEME-SPREAD` |
| front speed / Benjamin anchor | `7.7672136e-4` | `2.3985106e-5` | `1.1419219e-3` | `WITHIN-SCHEME-SPREAD` |
| reference PE relative curve | `9.3340701e-8` | `3.8314077e-8` | `3.2938219e-7` | `WITHIN-SCHEME-SPREAD` |
| T-variance fraction curve | `2.0786519e-4` | `1.3063430e-4` | `1.4489910e-3` | `WITHIN-SCHEME-SPREAD` |
| normalized wet T-centre L-inf | `3.3930025e-2` | `8.3641536e-3` | `1.1151325e-1` | `WITHIN-SCHEME-SPREAD` |
| normalized wet instantaneous-U L-inf | `4.2435992e-2` | `8.6567846e-3` | `8.4848863e-2` | `WITHIN-SCHEME-SPREAD` |

The analytic Benjamin anchor is `0.5330076485 m/s`. N2 and L64 fitted speeds
are `0.5149051627` and `0.5144911643 m/s` (`c/c_B=0.96603710` and
`0.96526038`). Final front positions are `64.0597733 km` and `64.0379292 km`.
Final N2/L64 relative RPE changes are `-8.9640999e-5` and `-8.9734340e-5`;
final variance fractions are `0.80557509` and `0.80536722`.

The deterministic/statistical handoff is on one registered series. At the end
of the certified before-entry ladder (`completed=59`), midpoint before-entry,
and final restart, T errors are `3.9549166e-6`, `2.6553015e-2`, and
`3.3930025e-2`; instantaneous-U errors are `7.4111368e-6`, `1.6754899e-2`,
and `4.2435992e-2`. Neither series saturates by the final sample: both increase
from midpoint to final. That deterministic growth does not override the
separate, preregistered scheme-spread verdict.

The machine reports are:

- LOCK: `b191261395a883b174437c3a44db6012a1508095a79284f3fb3ad5d6af9ea031`
  at `/data/abyssal/dbalwada/nemo-testcases-l1/full_statistical/reports/lock_exchange_zco.json`.
- OVERFLOW: `db9020b1f86b2708a769ec2883fc4a341d3925d6dc67b7b003ed4e67cc98f387`
  at `/data/abyssal/dbalwada/nemo-testcases-l1/full_statistical/reports/overflow_zps.json`.

## Figures and controls

legoESM section fields were **loaded from the saved CPU production states**;
they were not recomputed for plotting. The section comparator uses the scorer's
phase-3 loader, T-centre frame, halo stripping, and common wet mask. Each
difference panel has its own symmetric scale. The exact-zero initial LOCK
difference uses an explicit `1e-15 K` display scale and remains labeled zero.

| figure | SHA256 |
|---|---|
| `lock_exchange_zco_full_temperature_sections.png` | `ecba351e76a8502518057e701e6efb78357aa9605e446b4a0b72b319e9de0182` |
| `lock_exchange_zco_full_statistical_metrics.png` | `4c63c340d6575556944099dd7f63b034bb7f80a2dd9d6781766d153bde1f2882` |
| `overflow_zps_full_duration_failure.png` | `1e99327fbd0fd75fb5b5261f47243783c8dbbd53e813c101216fdc833306c764` |

Primary copies are under
`/data/abyssal/dbalwada/nemo-testcases-l1/figures_full/`; byte-identical copies
are under `/tmp/l1_figures_full/`. OVERFLOW has no three-time section or metric
curve figure because no finite legoESM midpoint/final state exists. The failure
figure states that absence directly instead of drawing unobserved fields.

The final lane testcase subset is 55/55 passing: 8 full-statistics tests,
18 phase-1 oracle-gate tests, 7 phase-2 tests, 3 EOS tests, 8 first-divergence
tests, 2 stage-sweep tests, and 9 trajectory-gate tests. In addition, the
planted wet T control moves
LOCK `temperature_linf` to `OUTSIDE`; the planted unregistered-metric control
exits nonzero on the inventory mismatch; the endpoint test preserves loud
`UNMEASURED` classification while the gross excursion hard-fails. The planted
OVERFLOW census arm cannot be evaluated after the primary candidate fails
before its registered midpoint; its reducer and CLI wiring remain directly
tested, but no claim is made that a missing production state exercised it.

## Open debt

- **CONFIRMED OUTSIDE:** OVERFLOW-zps full-duration stability. The next causal
  task starts at the first finite-to-non-finite transition near completed step
  2,877; this receipt does not authorize a stabilizer or a changed card.
- **WITHIN-SCHEME-SPREAD, not at-floor:** all six LOCK statistical metrics.
  More oracle times or an ensemble would be a new preregistered experiment,
  not a reinterpretation of these three samples.
- **UNAVAILABLE BY PRIMARY FAILURE:** OVERFLOW plume descent/front/final census
  and full-duration deterministic bridge. Their machine verdict is `OUTSIDE`,
  while the quantities themselves were not computed from non-finite states.
