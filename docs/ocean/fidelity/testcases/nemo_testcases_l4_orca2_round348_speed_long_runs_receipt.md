# ORCA2 round 348 — fast exact-libm and long-run arm receipt

Date: 2026-10-30. Base: `4d86c201351d`. Scope: final round-348 review fixes.

## Result

The measured profile was read with `python -m pstats` before implementation:
the 24-step run took 836.812 s cumulatively, with 514.707 s in the unary
Python scalar-libm loop, 11.689 s in its binary counterpart, and 184.813 s in
XLA backend compilation. These are profile observations, not new run results.

The three round-348 commits were rebased onto `4d86c201351d`; main's portable,
lazy `glibc_libm()` API remains the one loader for `libm.so.6`. The compiled
loop cache key contains the packaged C-source, GCC-version, and compile-flag
digests. The per-user cache and each keyed directory must be owner-only mode
0700; every shared object has a sidecar SHA-256 that is checked before load
(`packages/core/legoesm/core/transcendentals.py:36-214`). The C loops call the
double-precision scalar symbols once per element in ascending order; float32
also calls the double symbol and converts only the stored result
(`packages/core/legoesm/core/_scalar_libm.c:4-38`). The existing Python ctypes
loops remain only as the test oracle. The callbacks still use
`jax.pure_callback`, `vmap_method="broadcast_all"`, and the existing custom
JVPs (`packages/core/legoesm/core/transcendentals.py:224-449`).

The round-342 year driver accepts only `libm` or `native`, defaults to `libm`,
checks output emptiness and clean provenance before card construction, and
records the resolved policy plus loaded shared-object path, SHA-256, and GCC
version in the real manifest writer
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py:234-299` and `:302-560`). Its wrapper performs the same
preflight before creating launcher evidence.

The new arm comparator performs raw all-field snapshot comparison with signed
zeros distinct, reuses the round-342 area weights/loaders and the round-343 wet
3-D scorer, refuses identical roots, checks and prints both manifest modes, and
runs a plant through the production daily-temperature comparison before writing
daily SST/3-D-T RMS to JSON and PNG
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py:56-223` and `:249-361`). The existing round-342
NEMO metrics and round-345 Decision-188 gate were not changed.

## MODEL-CODE CHECKLIST M6

A repository grep for imports of `legoesm.core.transcendentals` found callers
only in core bulk flux and thermo, sea-ice dynamics/rheology, ocean bulk flux,
the NEMO recipe, shortwave penetration, lateral mixing, and vertical mixing.
All enter the eight public functions. Those functions have exactly two libm
callback sinks: unary `_scalar_libm` and binary `_scalar_libm_binary`; native
mode enters the corresponding JAX primitive
(`packages/core/legoesm/core/transcendentals.py:396-505`). Therefore the
million-value equality tests cover every function routed by every caller.

## Verification

- 55 transcendental tests passed, including at least 1,000,000 float32 and
  float64 values for each of exp, log, log10, atan, tanh, sin, cos, and pow;
  edge sets include subnormals, signed zero, infinities, NaNs, boundary ranges,
  and pow special cases.
- The disassembly test requires each f32/f64 loop to call its named scalar
  symbol through the PLT and rejects vector-math calls. Its substituted-symbol
  plant fails. A separate fast-function substitution plant fails bitwise.
- Seven round-348 comparator tests passed. Their byte-flip, signed-zero,
  area-weight, wet-mask, routing-hook, and production daily-temperature plants
  all fire, including explicit no-op-control failures.
- The new driver default/native round trip, wrapper pass-through, unknown-mode
  refusal, shell syntax, and focused lint checks passed.
- Direct core-wheel construction was not available in this sandbox because
  Hatchling was absent and network installation is disabled. The C source is
  inside the wheel-selected `legoesm` package tree; runtime compilation and
  loading passed in the tests.

After the implementation commit, the complete focused suite passed on the
clean tree, including the two round-342 provenance-stamp tests.

## Acquisition disposition

ACQUISITION_NEEDED. Re-run the preregistered default-flag 80-step bitwise
identity check on this final code. The exact CPU-only, one-at-a-time commands,
expected lines, signed-zero policy, and refusal conditions are preregistered in
`PREREG_nemo_testcases_l4_orca2_round348_speed_long_runs.md`. No NEMO or long
legoESM integration was run in this round.

## Review and choices

Independent reviews: Claude Sonnet — HOLD; Gemini Argon — HOLD. Every item was
dispositioned:

- **Sonnet + Argon — power dtype:** the libm callback now casts both operands to
  JAX's weak-type-aware result dtype before crossing the callback; native mode
  remains unchanged (`packages/core/legoesm/core/transcendentals.py:334-344`
  and `:476-481`).
- **Sonnet — routing hook:** the round-16 control now poisons the compiled loop
  result and counts routed elements; its no-op plant fails the test
  (`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round16_discriminator.py:353-407`).
- **Sonnet + Argon — cache:** compile flags now enter the cache key; the default
  is per-user; ownership, permissions, build/load failures, temporary cleanup,
  corrupt binaries, and SHA mismatches fail closed
  (`packages/core/legoesm/core/transcendentals.py:90-214`).
- **Argon — non-contiguous reference:** the Python oracle writes to a
  C-contiguous result and is tested with a transpose
  (`packages/core/legoesm/core/transcendentals.py:261-275`).
- **Sonnet + Argon — manifest/driver/wrapper:** the manifest records the
  resolved policy and compiled-library provenance; both launch layers refuse
  occupied output, and clean provenance is checked before model setup. The old
  stub round trip is replaced by the real resolver and writer
  (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py:234-323`).
- **Argon — comparator:** identical roots are refused; both manifests' CPU
  backend and mode are checked and printed; the planted control now traverses
  `compare_daily_temperature`
  (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py:155-223` and `:249-361`).

Operator choices explicitly preserved: compare both libm and native arms;
keep libm as the certified default; perform long acquisitions on CPU one at a
time; do not change the existing native-vs-NEMO metric card or Decision-188
bar. No unasked scientific choice or relaxed threshold was introduced.

## Landing gates

- Receipt citation gate: 22/22 passed. The round-348 receipt had
  zero failures and unmapped citations, the complete map audit passed, all
  self-test controls fired, and the shifted C-source citation plant failed as
  required.
- Named serial CPU suites: transcendental 55/55, lazy-libm 2/2, round-348
  comparator 7/7, and round-342 driver/provenance 14/14. No whole-tree or
  parallel test run was used.
