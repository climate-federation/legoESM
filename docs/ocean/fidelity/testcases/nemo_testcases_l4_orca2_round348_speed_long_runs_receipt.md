# ORCA2 round 348 — fast exact-libm and long-run arm receipt

Date: 2026-10-30. Base: `6b9723445516`. Scope: tasks 1–4 only.

## Result

The measured profile was read with `python -m pstats` before implementation:
the 24-step run took 836.812 s cumulatively, with 514.707 s in the unary
Python scalar-libm loop, 11.689 s in its binary counterpart, and 184.813 s in
XLA backend compilation. These are profile observations, not new run results.

The libm policy now loads a fail-closed compiled loop. Its build cache key is
the packaged C-source digest plus GCC-version digest, and the build command
uses `-O2 -fno-builtin -fno-tree-vectorize` without fast math
(`packages/core/legoesm/core/transcendentals.py:35-117`). The C loops call the
double-precision scalar symbols once per element in ascending order; float32
also calls the double symbol and converts only the stored result
(`packages/core/legoesm/core/_scalar_libm.c:4-38`). The existing Python ctypes
loops remain only as the test oracle. The callbacks still use
`jax.pure_callback`, `vmap_method="broadcast_all"`, and the existing custom
JVPs (`packages/core/legoesm/core/transcendentals.py:145-258`).

The round-342 year driver accepts only `libm` or `native`, defaults to `libm`,
sets the fp64 policy, confirms the resolved card, and records the mode in the
manifest (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py:229-263` and `:477-528`). Its wrapper passes the
same flag and rejects an unknown value.

The new arm comparator performs raw all-field snapshot comparison with signed
zeros distinct, reuses the round-342 area weights/loaders and the round-343 wet
3-D scorer, and writes daily SST/3-D-T RMS to JSON and PNG
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py:56-185` and `:229-290`). The existing round-342
NEMO metrics and round-345 Decision-188 gate were not changed.

## MODEL-CODE CHECKLIST M6

A repository grep for imports of `legoesm.core.transcendentals` found callers
only in core bulk flux and thermo, sea-ice dynamics/rheology, ocean bulk flux,
the NEMO recipe, shortwave penetration, lateral mixing, and vertical mixing.
All enter the eight public functions. Those functions have exactly two libm
callback sinks: unary `_scalar_libm` and binary `_scalar_libm_binary`; native
mode enters the corresponding JAX primitive
(`packages/core/legoesm/core/transcendentals.py:297-411`). Therefore the
million-value equality tests cover every function routed by every caller.

## Verification

- 41 transcendental tests passed, including at least 1,000,000 float32 and
  float64 values for each of exp, log, log10, atan, tanh, sin, cos, and pow;
  edge sets include subnormals, signed zero, infinities, NaNs, boundary ranges,
  and pow special cases.
- The disassembly test requires each f32/f64 loop to call its named scalar
  symbol through the PLT and rejects vector-math calls. Its substituted-symbol
  plant fails. A separate fast-function substitution plant fails bitwise.
- Five round-348 comparator tests passed. Their byte-flip, signed-zero,
  area-weight, and wet-mask plants all fire.
- The new driver default/native round trip, wrapper pass-through, unknown-mode
  refusal, shell syntax, and focused lint checks passed.
- Direct core-wheel construction was not available in this sandbox because
  Hatchling was absent and network installation is disabled. The C source is
  inside the wheel-selected `legoesm` package tree; runtime compilation and
  loading passed in the tests.

The two pre-existing round-342 tests that demand a clean provenance stamp were
deferred until after the implementation commit; all other focused tests passed
while the tree was dirty. The citation gate, default gate, and ratchet results
are recorded after the clean-tree run below.

## Acquisition disposition

ACQUISITION_NEEDED. The exact CPU-only, one-at-a-time commands, expected lines,
signed-zero policy, and refusal conditions are preregistered in
`PREREG_nemo_testcases_l4_orca2_round348_speed_long_runs.md`. No NEMO or long
legoESM integration was run in this round.

## Review and choices

independent review: operator (Claude)

Operator choices explicitly preserved: compare both libm and native arms;
keep libm as the certified default; perform long acquisitions on CPU one at a
time; do not change the existing native-vs-NEMO metric card or Decision-188
bar. No unasked scientific choice or relaxed threshold was introduced.

## Landing gates

- Receipt citation gate: PENDING CLEAN-TREE RUN.
- Default ORCA2 gate: PENDING CLEAN-TREE RUN.
- Ratchet: PENDING CLEAN-TREE RUN.
