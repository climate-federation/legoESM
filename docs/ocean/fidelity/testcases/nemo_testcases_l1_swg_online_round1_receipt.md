# SWG online learning — round 1 receipt

Date: 2026-10-13

Status: **MEASURED; HELD FOR ORACLE ACQUISITIONS AND INDEPENDENT REVIEW**.

Incoming tip: `8f8551c87`. Instrument commits: `23bc6d0cb`, `24df0b1b9`,
and `e12622c33`. Writable clone:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgonl-288230979`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round1/`.

## Deck and oracle preflight

`dg50` resolves from the same parameterized card as the fine deck with
`nn_SWG=2`, 50 km, `rn_Dt=400 s`, and 216 steps/day. Wind, gravity, 500 m
layer, 10 m2/s Laplacian viscosity, `1e-7 s-1` drag, beta, and f0 are
unchanged. Both decks have the same per-axis gravity-wave Courant number,
`sqrt(9.80665 * 500) * dt / dx = 0.56019`. This is the stability-preserving
scaling for the executed RK3 pressure-gradient stages
(`SWG_DG/BLD/ppsrc/nemo/stprk3.f90:140-171`,
`SWG_DG/BLD/ppsrc/nemo/stprk3.f90:196-227`, and
`SWG_DG/BLD/ppsrc/nemo/stprk3.f90:252-290`); NEMO independently defines the
advective velocity Courants as `abs(u)*dt/e1` and `abs(v)*dt/e2`
(`SWG_DG/BLD/ppsrc/nemo/diacfl.f90:74-97`).

The stage and 30-day no-run preflights pass at `e12622c33`. They bind the
`SWG_DG` reference binary SHA-256
`0be0d4b0846c34b3540105a54eaa6de225e03a9d8927cf5dcc8d85198b88bf09`,
resolve 10 stage steps and 6,480 month steps, and stop before `makenemo` or
`mpirun`. Their evidence hashes are respectively
`68d5e564dfc76352b4571d9c5b08b971eab93bd0f879cb3b3dc840d58cb4fc7a`
and `a80747ce2c9e20cee1525092dea52eaec9661238c0af1fd88d8a8f34c17ee733`.
The operator acquisitions remain OPEN; commands are in the OPEN section.

The established no-argument shipped card digest remains
`f9d46bfd08f23702`; the fine card digest is `b2f60cd83f7d0a00`; dg50 is
`a2d1a0ed9ff103a3`. The mounted shipped/fine ladder tests remain passing.

## Coarse-grained reference

The committed operator uses a 50 km Gaussian (`sigma=8` fine cells,
six-sigma support), mask normalization at walls, and exact overlap-area
averaging onto native T, U, and V footprints, followed by a float64 wet-area
mean residual correction. Its three direct tests pass: constants are
preserved, the periodic plane wave agrees with
`exp(-sigma^2*k^2/2)` within `2e-5`, every staggered wet mean is conserved to
`5e-16`, and the unnormalized-wall planted control differs by more than 0.1.

All 1,095 existing fine daily frames were converted twice: instantaneous
snapshots and daily means. Each archive has 1,095 finite 84 by 84 T/U/V files,
source and output file hashes, the exact source-manifest hash, both resolved
namelists, filter parameters, sampling convention, and commit. Snapshot
manifest SHA-256:
`63b37bc3ebe60fc63e8ef89b8dde31617e3786ab4ddea457711fbe725bb51680`;
daily-mean manifest SHA-256:
`b8013a3b90b3a9c14a2b23aa5503bf2aa3d587c2bb0623a4dd0a61f8d60f8e72`.

## Twenty-year no-closure baseline

The full 7,300-day dg50 run completed from rest at `e12622c33`, using a
one-day `lax.scan` on CPU under the fp64/libm policy. A 216-step profile
measured 1.418 ms/step for the scan and 11.610 ms/step for Python dispatch;
the registered scan/loop parity gate passes at `1e-12`. Baseline manifest
SHA-256:
`cdbdf5607a4185275ba794d63be152eb643a03579350e0bf8c2758dcb924cd6a`.

The final all-20-year health envelope is finite, minimum total layer thickness
497.67697 m, maximum `Cu=0.0209152`, and maximum `Cv=0.0146682`. The original
10 m2/s viscosity is therefore numerically usable. No grid-scaled-viscosity
decision is needed and no parameter was changed.

The matched days 1--1095 snapshot comparison completed. Report SHA-256:
`3dbbe21c14fbbc04af1b8d3416eccb60aa5230c8e15d1de578725a30a004bb29`.
Coarse-minus-filtered headline gaps are:

- basin KE mean: `+0.00688765 m2/s2` (coarse `0.0205002`, filtered
  `0.0136126`), RMS daily difference `0.00698002 m2/s2`;
- basin EKE mean: `+0.000586061 m2/s2`, RMS daily difference
  `0.00222271 m2/s2`;
- EKE-map bias/RMSE: `+0.000531575 / 0.00801336 m2/s2`;
- time-mean SSH map RMSE: `0.0756971 m`;
- mean jet core: 51.875 km north of filtered truth, with 162.884 km RMS
  displacement; mean eastward core speed bias `+0.169681 m/s`;
- mean KE-spectrum bias/RMSE over 40 common bins:
  `+688.765 / 810.719 m3/s2`.

This is a large, resolved no-closure target rather than a marginal correction.

## Fine predictability measurement

The exact day-1095 restart was restored with native T/U/V staggering. A
32-step fine CPU profile measured 20.7324 ms/step for the scan and
118.294 ms/step for Python dispatch; scan/loop maximum differences were
`7.33e-15` SSH, `1.33e-15` U, and `1.11e-15` V. The measured budget admits
`floor(3600 / (2 * 1728 * 0.0207324)) = 50` twin days. Both 50-day members
completed under fp64/libm: one unperturbed, one with seed-0 wet-T white SSH
noise of nominal amplitude `1e-6 m`.

The exact realized initial SSH RMS is `1.0013953e-6 m`; its e-fold threshold is
`2.7220746e-6 m`. The perturbation initially projects away: SSH RMS is
`9.74860e-8 m` at day 1, reaches its partial-curve minimum
`4.42259e-8 m` at day 18, and returns to `3.02926e-7 m` (0.3025 times the
initial RMS) at day 50. It never crosses the threshold. Therefore the fine
predictability time and saturation are **UNRESOLVED**, not assumed to be nine
days. Predictability JSON SHA-256:
`627558be9b8a7f50b2407f55d098e8d7a2aa3a653db3c780d76b258a0cf15846`.
A 120-day GPU twin remains an acquisition.

## Gates and review

The focused committed suite passes 49/49 for the driver, coarse-graining,
paired climate, predictability, dg50 ladder dispatch, and citation gate; after
the CPU acquisition additions the driver suite passes 10/10. The citation
gate passes all four design-note oracle citations, while the shifted `diacfl`
plant exits nonzero with `SYMBOL-NOT-AT-LINE`. Python compilation, shell
syntax, `git diff --check`, and Ruff on the new/locally clean Python files pass.

Independent adversarial review is not available in this environment: the
Claude CLI refuses the untrusted workspace and hangs before review, and GLM is
not installed. No review verdict is fabricated. This is a process HOLD under
the repository's two-reviewer rule.

## OPEN

- Operator-only dg50 stage acquisition:
  `cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgonl-288230979 && bash scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/run.sh --deck dg50 --run`
  -> `round1/oracle_swg_dg50_r1/`.
- Operator-only dg50 30-day acquisition:
  `cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgonl-288230979 && bash scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/run_30d.sh --deck dg50 --run`
  -> `round1/oracle_swg_dg50_r1_30d/`.
- When GPU 1 is free, run the 120-day control and perturbed commands printed in
  the final handoff, then the committed predictability analyzer. Until then
  the e-folding and saturation results remain unresolved.
- The operator's announced 7,300-day fine truth directory is still absent.
  Filtering and climate scoring currently cover the complete 1,095-day
  snapshot and daily-mean archives; extend both only after that source closes.
- Obtain two independent adversarial review verdicts and disposition every
  finding before this round is cited or shipped.
