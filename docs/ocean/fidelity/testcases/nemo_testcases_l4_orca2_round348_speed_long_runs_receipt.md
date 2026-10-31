# ORCA2 round 348 — fast exact-libm and long-run arm receipt

Date: 2026-10-30. Rebased base: `9c070141d0cf`. Scope: final round-348 review
fixes and post-catch-up landing gates.

## Result

The measured profile was read with `python -m pstats` before implementation:
the 24-step run took 836.812 s cumulatively, with 514.707 s in the unary
Python scalar-libm loop, 11.689 s in its binary counterpart, and 184.813 s in
XLA backend compilation. These are profile observations, not new run results.

The round-348 implementation was first rebased onto `4d86c201351d`; main's portable,
lazy `glibc_libm()` API remains the one loader for `libm.so.6`. The compiled
loop cache key contains the packaged C-source, GCC version, glibc version from
`gnu_get_libc_version`, and compile flags in one composite digest. The per-user
cache and each keyed directory must be owner-only mode
0700; every shared object has a sidecar SHA-256 that is checked before load
(`packages/core/legoesm/core/transcendentals.py:36-239`). The C loops call the
double-precision scalar symbols once per element in ascending order; float32
also calls the double symbol and converts only the stored result
(`packages/core/legoesm/core/_scalar_libm.c:4-38`). The existing Python ctypes
loops remain only as the test oracle. The callbacks still use
`jax.pure_callback`, `vmap_method="broadcast_all"`, and the existing custom
JVPs (`packages/core/legoesm/core/transcendentals.py:249-484`).

The round-342 year driver accepts only `libm` or `native`, defaults to `libm`,
checks output emptiness and clean provenance before card construction, and
records the resolved policy plus loaded shared-object path, SHA-256, GCC
version, and glibc version in the real manifest writer
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py:234-299` and `:302-560`). Its wrapper performs the same
preflight before creating launcher evidence.

The new arm comparator performs raw all-field snapshot comparison with signed
zeros distinct, reuses the round-342 area weights/loaders and the round-343 wet
3-D scorer, refuses identical roots and manifests, binds each manifest to its
run root, and checks and prints both modes. The certified pre-r348 round-342
reference may omit the mode because that lane unconditionally used libm; the
comparator prints this rule when it applies. A plant traverses the production
daily-temperature comparison before JSON and PNG are written
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py:56-223` and `:249-402`). The existing round-342
NEMO metrics and round-345 Decision-188 gate were not changed.

## Resume 3 rebase

The nine commits in the original `4d86c20135..a666faaef0` range were rebased
with `git rebase --onto refs/remotes/lane/landed 4d86c20135` onto the landed
Decision-189 catch-up at `9c070141d0cf`. Git replayed all nine commits and
reported `Successfully rebased and updated`; it stopped on no commit and wrote
no conflict markers.

Conflict-hunk inventory: **none**. There is therefore no file, ours/theirs
pair, or manual resolution to record. Every replayed diff was accepted by Git
without a content conflict. The post-rebase citation audit identified only the
line shift introduced later by the new `argparse` import and tests; the two
affected receipt citations and their `CITATION_MAP` keys were re-anchored from
lines 34–38 to 35–39 and from lines 179–349 to 180–448.

## MODEL-CODE CHECKLIST M6

A repository grep for imports of `legoesm.core.transcendentals` found callers
only in core bulk flux and thermo, sea-ice dynamics/rheology, ocean bulk flux,
the NEMO recipe, shortwave penetration, lateral mixing, and vertical mixing.
All enter the eight public functions. Those functions have exactly two libm
callback sinks: unary `_scalar_libm` and binary `_scalar_libm_binary`; native
mode enters the corresponding JAX primitive
(`packages/core/legoesm/core/transcendentals.py:431-546`). Therefore the
million-value equality tests cover every function routed by every caller.

## Verification

- 69 transcendental tests passed, including at least 1,000,000 float32 and
  float64 values for each of exp, log, log10, atan, tanh, sin, cos, and pow;
  edge sets include subnormals, signed zero, infinities, NaNs, boundary ranges,
  and pow special cases.
- The disassembly test requires each f32/f64 loop to call its named scalar
  symbol through the PLT and rejects vector-math calls. Its substituted-symbol
  plant fails. A separate fast-function substitution plant fails bitwise.
- Nine round-348 comparator tests passed. Their byte-flip, signed-zero,
  area-weight, wet-mask, routing-hook, and production daily-temperature plants
  all fire, including explicit no-op-control failures.
- Nineteen round-342 tests passed on the clean tree. The CLI reaches the native
  card override; removing its argument pass-through makes that test fail. The
  wrapper dirty-tree refusal, trailing-slash log placement, mode pass-through,
  unknown-mode refusal, shell syntax, and focused lint checks passed.
- The round-342 callable, CLI, and shell-wrapper defaults are behaviorally
  pinned to certified `libm` at all three definition sites
  (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py:302-312`;
  `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py:585-588`;
  `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_legoesm_run.sh:27`).
  Flipping each default independently to `native` makes its pin assertion fail
  (`tests/ocean/fidelity/test_orca2_round342_s4_year.py:265-360` and
  `tests/ocean/fidelity/test_orca2_round342_s4_year.py:710-727`).
- The comparator's legacy-field exception remains limited to the certified
  round-342 reference with its exact status, and never applies to the candidate
  (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py:282-300`).
  Wrong-round, wrong-status, and candidate-side certified-round-342 plants all
  fail; the candidate plant satisfies every legacy predicate except
  `index == 0`, so removing only that condition makes the test fail
  (`tests/ocean/fidelity/test_orca2_round348_transcendentals.py:312-356`).
- Direct core-wheel construction was not available in this sandbox because
  Hatchling was absent and network installation is disabled. The C source is
  inside the wheel-selected `legoesm` package tree; runtime compilation and
  loading passed in the tests.

After the implementation commit, the complete focused suite passed on the
clean tree, including the two round-342 provenance-stamp tests.

## Acquisition disposition

Acquisition A4 at `7820af6091` passed the preregistered default-flag 80-step
bitwise identity check: 11 snapshots were bitwise exact in 459.5 seconds. No
NEMO run is needed for this test-only follow-up.

## Review and choices

Initial independent reviews: Claude Sonnet — HOLD; Gemini Argon — HOLD. Their
first-round items were dispositioned:

- **Sonnet + Argon — power dtype:** the libm callback now casts both operands to
  JAX's weak-type-aware result dtype before crossing the callback; native mode
  remains unchanged (`packages/core/legoesm/core/transcendentals.py:367-379`
  and `:511-522`).
- **Sonnet — routing hook:** the round-16 control now poisons the compiled loop
  result and counts routed elements; its no-op plant fails the test
  (`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round16_discriminator.py:353-407`).
- **Sonnet + Argon — cache:** compile flags now enter the cache key; the default
  is per-user; ownership, permissions, build/load failures, temporary cleanup,
  corrupt binaries, and SHA mismatches fail closed
  (`packages/core/legoesm/core/transcendentals.py:91-239`).
- **Argon — non-contiguous reference:** the Python oracle writes to a
  C-contiguous result and is tested with a transpose
  (`packages/core/legoesm/core/transcendentals.py:286-300`).
- **Sonnet + Argon — manifest/driver/wrapper:** the manifest records the
  resolved policy and compiled-library provenance; both launch layers refuse
  occupied output, and clean provenance is checked before model setup. The old
  stub round trip is replaced by the real resolver and writer
  (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py:234-323`).
- **Argon — comparator:** identical roots are refused; both manifests' CPU
  backend and mode are checked and printed; the planted control now traverses
  `compare_daily_temperature`
  (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py:155-223` and `:249-402`).

Resume-2 review inputs were Claude Sonnet — LAND and Gemini Argon — HOLD. Every
new claim was measured on `4d86c20135` and `08bc33f0f8` before this follow-up:

- **Argon 1 — REAL:** the comparator did not exist at the base, and HEAD
  crashed on the real certified round-342 manifest because it lacks the later
  mode field. The fix applies the explicit certified-reference libm rule,
  prints it, refuses identical manifest paths, and requires each manifest to be
  directly inside its run root. The regression copies the real round-342
  manifest rather than constructing its reference
  (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py:249-312`; `tests/ocean/fidelity/test_orca2_round348_transcendentals.py:227-261` and `:276-308`).
- **Argon 2 — REAL:** the base raised for a Python-float base with a float32
  exponent, while HEAD made the primal work but regressed `grad` and `jvp` with
  a float64 tangent mismatch. Integer operands raised on both. Libm operands
  are now strongly cast to the weak-type-aware floating result before the
  custom JVP; integer libm operands are promoted, and the native branch is
  unchanged. Float32/float64, Python-scalar, integer, `grad`, and `jvp` cases
  are covered (`packages/core/legoesm/core/transcendentals.py:367-379` and
  `:474-522`; `tests/unit/test_transcendentals.py:147-219`).
- **Argon 3 — REAL, direct-helper scope:** HEAD's public callback path already
  pre-cast its operands, so the callback could not receive the reported mixed
  dtype through that path. Direct calls to both the compiled helper and its
  reference did promote float32-array plus Python scalar to float64 after
  broadcasting. Result type is now chosen before broadcasting, with direct
  tests for both helpers (`packages/core/legoesm/core/transcendentals.py:316-364`;
  `tests/unit/test_transcendentals.py:216-219`).
- **Both reviewers' test finding — REAL:** the old test called the resolver and
  manifest writer directly, leaked global policy, and source-grepped wrapper
  pass-through; it never exercised CLI parsing or the card override. The new
  fixture restores policy, the behavioral test traverses `main()` into the
  card override, its deleted-pass-through plant fails, and fake-process wrapper
  tests cover dirty-tree refusal and argument routing
  (`tests/ocean/fidelity/test_orca2_round342_s4_year.py:35-39` and `:180-448`).
- **Sonnet portability/provenance finding — REAL:** HEAD had removed the
  non-glibc module skip, a trailing slash placed `.log` inside the output
  directory, and glibc version was absent from both cache identity and manifest
  provenance. All three are restored and behaviorally covered
  (`tests/unit/test_transcendentals.py:20-25`; `packages/core/legoesm/core/transcendentals.py:91-239`;
  `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_legoesm_run.sh:58-108`).

Operator choices explicitly preserved: compare both libm and native arms;
keep libm as the certified default; perform long acquisitions on CPU one at a
time; do not change the existing native-vs-NEMO metric card or Decision-188
bar. No unasked scientific choice or relaxed threshold was introduced.

Resume-4 reviews of `7820af6091`: Claude Sonnet returned **LAND** and Gemini
Argon returned **HOLD**. Both identified the same two test gaps: the callable
default was unpinned, and the candidate legacy-field plant was rejected by its
round before exercising the candidate-only index guard. This test-only
follow-up closes both gaps; Argon re-review remains an operator landing step.

## Recorded follow-ups (unchanged in this resume)

- `--output-dir foo//` can still place the wrapper log incorrectly.
- The JAX path still flushes denormals.
- The libm `pow` JVP can produce NaN at base zero or a negative base.
- The comparator still does not require the two manifest modes to differ.

These four items are recorded only; this resume deliberately changes none of
them.

## Landing gates

- Per-receipt citation gate final line: `"status": "PASS"`; 33 citations,
  zero failures, zero unmapped citations, zero map-audit failures, and all nine
  self-test controls fired.
- Round-348 named files, each in its own fresh serial CPU process:
  - `tests/unit/test_transcendentals.py`: `69 passed`.
  - `tests/unit/test_transcendentals_lazy_libm.py`: `2 passed`.
  - `tests/ocean/fidelity/test_orca2_round348_transcendentals.py`:
    `9 passed`.
  - `tests/ocean/fidelity/test_orca2_round342_s4_year.py`:
    `19 passed`.
- The exact eight-file ORCA2 `land.sh` list passed 157/157, with every file in
  a fresh serial process. Quoted final lines:
  - receipt-citation suite: `22 passed in 22.38s`;
  - private-import ratchet: `6 passed in 8.96s`;
  - TKE terms: `76 passed in 45.21s`;
  - NEMO recipe: `25 passed in 364.21s (0:06:04)`;
  - real-freshwater closure: `16 passed in 6.74s`;
  - stage face-mask rank: `2 passed in 554.41s (0:09:14)`;
  - prognostic barotropic state: `8 passed in 64.02s (0:01:04)`;
  - ORCA2 parallel receipt gate: `2 passed in 0.46s`.
- Root and standing ratchets were also run as separate named files:
  - inline coefficients: `376 passed in 6.12s`;
  - dispatch hardening: `24 passed in 11.07s`;
  - strict-validation coverage: `73 passed in 5.88s`;
  - RK3-WS/MXL3: `12 passed, 1 warning in 118.17s (0:01:58)`;
  - parameter specs: `1 failed, 72 passed in 4.54s`; inherited failure
    `test_config_module_has_valid_param_spec[packages/ocean/legoesm/ocean/physics/shortwave_penetration.py]`
    names non-float fields `rgb_chl_profile`, `scheme`, and `water_type`;
  - hard-coded constants: `1 failed, 6254 passed, 2 skipped, 1 warning in
    49.04s`; inherited failure while parsing round-336
    `make_stub_records.py`, `IndentationError` at line 72;
  - saturation reimplementation: `1 failed, 6250 passed, 2 skipped, 1 warning
    in 48.25s`; the same inherited round-336 parse failure;
  - worktree-stamp census: `1 failed, 11 passed in 4.28s`; inherited
    `test_every_report_emitter_stamps_the_worktree` census of 19 older gate
    scripts;
  - SI3 scalar-math provenance: `1 failed, 4 passed in 0.86s`; inherited
    `test_full_v2_gate_and_plants` failure `A MY_SRC is not verbatim`.

No failing ratchet path is changed by the resume-4 test/receipt commits.
