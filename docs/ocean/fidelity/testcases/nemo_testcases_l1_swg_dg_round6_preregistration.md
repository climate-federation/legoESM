# SWG double-gyre round 6 preregistration

Date: 2026-10-12

This round discriminates the certified-CPU last-bit debt accepted in round 5.
It changes no physics, configuration, arithmetic, fidelity bar, or NEMO source.
The inputs are the admitted round-1 `kt=2`, stage-1 record and its stage-2
target.  The exact IEEE-word bar remains binding.

## Fixed provenance

The incoming legoESM tip is `83d965c7548fba34d3411f9b67ba38ca20d1832f`.
The installed JAX and jaxlib versions are both `0.10.0`; the wheel records JAX
commit `a33ed614c58ee8a10d0b7536c50c2609c38500c1`, which pins OpenXLA commit
`b6f37ab7767f428fd6f993de5e211643d47d4deb` in
`third_party/xla/revision.bzl:22`.

At that XLA pin, `xla/debug_options_flags.cc` sets
`xla_cpu_enable_fast_math=false`, while `xla/xla.proto` documents
`xla_cpu_max_isa` as the upper ISA emitted by CPU code generation.  That pin
has no separate user-facing CPU FMA-contraction flag.  Consequently the H1
matrix distinguishes:

1. the default CPU build;
2. explicit `--xla_cpu_enable_fast_math=false`; and
3. explicit fast-math false plus `--xla_cpu_max_isa=SSE4_2`, which removes FMA
   instructions from the eligible ISA rather than claiming to alter HLO
   association.

The executed NEMO build uses gfortran flags
`-O3 -funroll-all-loops -fno-tree-vectorize` and no explicit fp-contract flag,
as recorded in `SWG_DG/BLD/cfg/parsed_bld.cfg:41-42`.  NEMO evaluates the
stage-1 continuity statements at `divhor.f90:134-140` and the SSH update at
`sshwzv.f90:135-149`.  legoESM's production path is
`swe_rk3.py:246-267` through the JIT dispatch at
`ocean_model_latlon_cgrid.py:13258-13280`.

## Instrument and controls

A committed round-6 probe will run the same production stage and emit a
source-ordered continuity trace: live face thickness, metric-thickness
products, U/V transports, the two directional differences, their sum, area
scaling, live-thickness division, one-layer reintegration, stage-time product,
SSH subtraction, and mask product.  It must satisfy all of these controls:

- its final SSH array is word-identical to the ordinary production-stage SSH
  on each backend; otherwise the trace is REFUTED and cannot localise a line;
- a one-ULP change to a nonzero admitted wet U operand changes the production
  SSH output; otherwise the probe is non-discriminating;
- CPU and GPU traces carry identical names, shapes, dtypes, input hashes, card
  digest, and admitted-record hashes before a first differing statement may be
  reported;
- every replay prints the backend, x64 state, precision policy, `XLA_FLAGS`,
  JAX/jaxlib versions, and git commit.

The trace is diagnostic only.  It does not replace or modify the production
stage, and no arithmetic finding authorises a patch.

## H1 — CPU fast math or FMA eligibility

Run all three CPU arms above in fresh processes.  Each replays admitted
`kt=2`, stage 1 and scores all 13 output fields against admitted stage 2.

- **CONFIRM fast-math ownership** only if explicit fast-math false changes the
  first SSH debt relative to default and moves it toward the NEMO words.
- **CONFIRM FMA-instruction ownership** only if the SSE4.2 cap changes that
  debt relative to explicit fast-math false and moves it toward NEMO.
- **REFUTE each hypothesis** if its controlled arm leaves every output word
  and trace word unchanged.
- A changed residual that does not move toward NEMO is a backend sensitivity,
  not ownership.

The registered baseline is round 5: 1,278 unequal wall-adjacent wet SSH cells,
maximum absolute residual `3.3881317890172014e-21` m.  The first CPU/GPU trace
statement with unequal words owns the backend split only if both trace-final
arrays reproduce their ordinary production outputs exactly.

## H2 — backend discrimination

The committed operator launcher will pin physical GPU 1 and replay the same
admitted entry under `PrecisionPolicy.fp64(transcendentals="native")`.  It is
supporting, non-certified evidence; the certified CPU policy remains
`fp64(transcendentals="libm")`.

- **CONFIRM backend ownership** if GPU production replay is word-identical to
  all admitted dynamic target fields for which CPU has nonzero residuals, and
  the common statement trace identifies a first CPU/GPU difference.
- **REFUTE backend ownership** if GPU reproduces the CPU debt.
- Any mixed result is DEBT and must be reported field by field.

The operator command is:

```text
bash scripts/run/run_swg_double_gyre_round6_gpu.sh
```

It writes only under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round6/`.

## H3 — NEMO rank-count invariance

The committed probe will compare the completed 10-step instantaneous restart
from one rank with the 32 coordinate-assembled rank files under
`swg_rounds/maddison_deck/rankcheck_kt10/`.  It covers every restart state and
forcing array, every stored level, and the scalar clock values; overlapping
tiles must agree word for word.

- **CONFIRM rank-count ownership** if any prognostic word differs between the
  one-rank and 32-rank restarts at step 10.
- **REFUTE it** if all compared words are identical.
- Missing coordinates, conflicting overlaps, dtype changes, or omitted restart
  variables REFUTE the instrument rather than yielding a physics verdict.

## Why the shipped deck does not expose the debt

The same trace will be run for the shipped admitted `kt=2`, stage-1 row, which
is already word-exact.  The comparison will report only resolved values that
differ between the two decks and, at the first double-gyre debt cell, the
magnitudes and IEEE words of every continuity operand.  It will classify the
exposure as one of:

- a term exactly zero on the shipped deck but nonzero on the double gyre;
- the same nonzero term crossing a rounding boundary because of magnitude;
- or unresolved if neither is mechanically shown.

No parameter will be varied: the requested comparison is observational, not
an attribution by a multi-change A/B.

## Frozen outcomes and stop condition

H1, H2, and H3 will each receive CONFIRMED or REFUTED from the definitions
above.  The first source statement will be named only if the trace controls
pass.  The arithmetic and exact bar remain unchanged.  A found owner is
`DECISION_NEEDED`, never implementation permission.

The no-argument shipped card digest and certified ladders must remain
`f9d46bfd08f23702`, given-entry 338/338, chained 338/338, independent 50/50,
carried 29/29, and month 690/690.  The acquisition preflights remain unchanged.
The unrelated inherited GYRE/LOCK_EXCHANGE/OVERFLOW digit pins remain stale:
expected `5850bbcfcd351c29` / `4ec82201f0fae7bf` / `7f1ec89b3e6578f5`, current
`f02d2f42a570f413` / `d5dfdf83b3a1e0ab` / `83de0010e1170032`.

## Preregistered instrument amendment — rejected trace shape

Before any campaign replay, the direct tiny-grid control rejected the proposed
multi-output statement trace: asking XLA to materialise the intermediate arrays
changed `ssh_k` in 1,600 of 1,936 cells, with maximum absolute difference
`3.3881317890172014e-21` m.  Compiling each intermediate as a separate root did
not restore the production result.  These are instrument-control results, not
campaign measurements; no admitted record was measured by the rejected trace.

The replacement statement bisect preserves the full production stage and its
ordinary outputs.  Each arm inserts one identity source-rounding barrier after
exactly one continuity statement; a baseline arm inserts none.  The baseline
must reproduce the ordinary production stage word for word or the bisect is
invalid.  An inserted boundary can identify the first rounding-sensitive
statement only if that control passes and the arm moves the replay toward the
admitted NEMO words.  H1, H2, H3, their frozen verdicts, and the exact bar are
otherwise unchanged.
