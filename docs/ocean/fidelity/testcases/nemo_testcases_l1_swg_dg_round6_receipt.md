# NEMO testcase L1 SWG_DG round 6 receipt

**Date:** 2026-10-12  
**Status:** HELD — H1 and H3 are refuted; the frozen same-entry GPU replay for
H2 is acquisition-blocked because this sandbox exposes no CUDA device.  No
physics, configuration, production arithmetic, or fidelity bar changed.

## Frozen question and instrument control

Round 6 tests whether the certified CPU's `kt=2`, stage-1 last-bit debt is
owned by CPU fast math/FMA (H1), backend choice (H2), or NEMO rank count (H3).
The preregistration is
`nemo_testcases_l1_swg_dg_round6_preregistration.md`.  The first proposed
multi-output trace was rejected before campaign measurement: merely returning
its intermediates changed 1,600/1,936 tiny-grid SSH words by at most
`3.3881317890172014e-21` m.  The committed replacement replays the complete
production JIT and inserts one identity boundary per arm.  Its no-boundary
baseline reproduces ordinary production SSH word for word.  The one-ULP U
plant changes two SSH words and is binding.

The committed measurement is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_round6.py`.
Its admitted record hashes, resolved namelist, JAX/jaxlib versions, policy,
backend, flags, platform, inputs, raw arrays, and output hashes accompany every
replay.  It refuses overwrite, unequal inputs, incomplete rank coverage,
conflicting coordinate overlaps, and non-float word comparisons.

## H1 verdict — REFUTED

Three fresh processes replayed the same admitted Maddison `kt=2`, stage-1
entry under certified `PrecisionPolicy.fp64(transcendentals="libm")`:

1. the ordinary environment;
2. `XLA_FLAGS=--xla_cpu_enable_fast_math=false`;
3. `XLA_FLAGS=--xla_cpu_enable_fast_math=false
   --xla_cpu_max_isa=SSE4_2`.

All three retain exactly the round-5 debt: SSH has 1,278 unequal
wall-adjacent wet words and maximum absolute residual
`3.3881317890172014e-21` m.  The other non-exact fields are V (408,318 words,
`6.6174449004242214e-24` m/s), r3t (1,276 words), r3u (1,168), r3v (1,072),
and r3f (2,148); every maximum is at most
`1.3234889800848443e-23`.  More importantly, every one of the 13 production
output arrays and all nine statement-probe arrays are word-identical between
default and explicit-fast-math-false, and again between explicit-fast-math-
false and the SSE4.2 cap.  No boundary arm moves one word toward NEMO, so the
bisect names no owner statement.

This JAX 0.10.0 wheel records jaxlib commit
`a33ed614c58ee8a10d0b7536c50c2609c38500c1` and XLA revision
`b6f37ab7767f428fd6f993de5e211643d47d4deb`.  At that revision XLA's
[debug defaults](https://github.com/openxla/xla/blob/b6f37ab7767f428fd6f993de5e211643d47d4deb/xla/debug_options_flags.cc)
set CPU fast math false, and its
[flag schema](https://github.com/openxla/xla/blob/b6f37ab7767f428fd6f993de5e211643d47d4deb/xla/xla.proto)
defines `xla_cpu_max_isa` as a cap on generated instructions.  SSE4.2 admits
no FMA instruction.  NEMO's executed build uses gfortran `-O3`, loop unrolling,
and `-fno-tree-vectorize`, with no explicit FP-contraction option.  Therefore
the preregistered fast-math and FMA-instruction ownership criteria are both
refuted; this does not prove that every possible CPU reassociation is absent.

## H2 verdict — UNMEASURED, acquisition required

The committed launcher pins physical GPU 1 and the native-transcendental
policy.  It reached JAX initialization but refused with
`CUDA_ERROR_NO_DEVICE`; no GPU evidence file was created.  The operator's
already accepted three-year native-GPU integrations remain strong supporting
evidence—yearly instantaneous restarts are word-identical to NEMO through
1,892,160 steps—but they do not satisfy this round's frozen same-admitted-entry
test and are not substituted for it.

The exact operator command is:

```text
bash /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgdg-2423021149/scripts/run/run_swg_double_gyre_round6_gpu.sh
```

It will write `gpu_replay.npz`, `gpu_replay.json`, and
`cpu_gpu_trace_comparison.json` under the round-6 evidence root.  H2 remains
UNMEASURED until those products exist.  No backend or arithmetic owner is
claimed.

## H3 verdict — REFUTED

The completed NEMO rank check compares the 1-rank step-10 restart with all 32
rank files assembled by exact coordinate keys.  All 21 covered variables are
word-identical: 12 state/forcing arrays, five restart clock scalars, longitude,
latitude, level, and time coordinate.  All overlaps agree, all global points
are filled, and every rank file is hash-pinned in `nemo_rank_check.json`.
NEMO rank count therefore does not own the single-rank ladder debt.

## Why the shipped deck does not expose the debt

The shipped CPU replay remains exact.  The observational operand audit selects
each deck's first active west-wall cell: shipped `[1,1]` at 25 km and Maddison
`[2,2]` at 6.25 km, the index shift produced by the common 10 km lower mask
bound.  Both chains are nonzero.  Their west-neighbour flux and meridional
difference are zero in both decks; the live zonal chain is nonzero in both.
Thus this is **not** a term that vanishes on the shipped deck.

The admitted shipped/Maddison wall values are respectively:

- U: `-1.4032382322555297e-3` / `-9.745531883500077e-6` m/s;
- zonal flux difference: `-1.7540477903194122e4` /
  `-3.0454787135937742e1` m3/s;
- reciprocal T area: `1.6e-9` / `2.56e-8` m-2;
- transport divergence: `-2.8064764645110597e-5` /
  `-7.796425506800062e-7` m/s;
- stage timestep: `1200` / `16.666666666666668` s;
- SSH increment: `-3.3677717574132716e-2` /
  `-1.2994042511333437e-5` m.

The double-gyre increment is 0.000385835 of the shipped increment and crosses
a different final rounding boundary.  The direct exposure is therefore the
changed wall magnitude and association: `dt=50` rather than 3,600 s and the
6.25 km rather than 25 km metric enter directly; `g=9.80665` rather than
0.02, `ahm=10` rather than 500 m2/s, and the 4,000 km rather than 2,000 km
basin shape the admitted wall velocity.  Because those parameters change
together, this observational comparison does not attribute the magnitude to
one of them.  `deck_exposure.json` stores every operand's value, IEEE word,
and cross-deck magnitude ratio.

## Preserved bars, tests, and preflights

The exact bar is unchanged.  The shipped no-argument card digest remains
`f9d46bfd08f23702`, and its certified rows remain given-entry 338/338,
chained 338/338, independent 50/50, carried 29/29, and month 690/690.  The
SWG-focused final battery records **238/238 passed** in
`round6/direct_tests.log`; it includes the card, production stage, mounted
ladders, acquisition checks, climate instrument, driver, and all five direct
round-6 tests.  Focused Ruff and `git diff --check` pass.

All four no-run acquisition preflights pass in
`round6/acquisition_preflights.log`: shipped and Maddison stage/month, always
building the target from shipped `SWG` before overlaying the selected case.
The daily cadences remain 24 and 432 steps/day.  No `makenemo`, `mpirun`, or
NEMO engine command was run in this round.

The canonical evidence root is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round6/`.
The failed pre-JSON payload is retained explicitly as
`cpu_default_replay.failed1.npz` and is not cited as scientific evidence.

## Open and inherited items

- H2 needs the operator GPU replay above.  Until then this round is HELD and
  there is no arithmetic/backend owner for a decision.
- The inherited forcing signed-zero question remains open: compare forcing
  `+0.0` and `-0.0` numerically equal while keeping prognostic rows word-exact?
  The operator's pick remains yes for forcing-only rows; this round does not
  alter the bar.
- The unrelated inherited GYRE/LOCK_EXCHANGE/OVERFLOW digit pins remain stale.
  Expected values are `5850bbcfcd351c29` / `4ec82201f0fae7bf` /
  `7f1ec89b3e6578f5`; current values are `f02d2f42a570f413` /
  `d5dfdf83b3a1e0ab` / `83de0010e1170032`.  This lane does not re-pin them.

## Oracle citations

The executed SWG_DG stage sets the stage-1 timestep, computes pressure and the
momentum RHS, calls the SSH update, then advances U/V at
`SWG_DG/BLD/ppsrc/nemo/stprk3.f90:140-171`.  The horizontal-divergence
association is explicit at `SWG_DG/BLD/ppsrc/nemo/divhor.f90:134-140`.
The SSH routine calls that divergence, multiplies by live thickness, subtracts
the stage increment, masks, and exchanges the result at
`SWG_DG/BLD/ppsrc/nemo/sshwzv.f90:135-149`.
