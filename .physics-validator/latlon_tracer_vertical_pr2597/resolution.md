# T20 merge resolution

Merge `origin/main` 3f07d5b866d35994f82aead14760645f046589fb into
`fix/latlon-tracer-vertical-conservation` at 576824fa0e122225b51eb7a1e498cde18a3cec31.
Initial tracked/untracked status was clean. No rebase, force, stash, push or PR.

| File / conflict hunk | Main had | Branch had | Kept | Why |
| --- | --- | --- | --- | --- |
| packages/coupler/legoesm/driver/config.py / DycoreConfig tail | Appended spectral_sponge_nu_top_m2s=0.0 after vertical_form | Appended tracer_vertical_scheme=None, tracer_van_leer_width=1e-10, tracer_van_leer_flux_width=1e-6 after vertical_form | Main sponge field followed by all three branch fields, comments and defaults unchanged | Both features survive; preserve main positional prefix, append topic fields. Both parents' positional tails cannot coexist: branch tracer slots shift by one. VOTED A: Codex A, GLM A. No production scheme/default choice introduced. |
| scripts/run/run_amip.py / build_config_from_args keywords | Forward spectral sponge with None-to-0.0 conversion | Forward tracer selector and both widths | All four keywords, including main's conversion | Independent settings must both reach DycoreConfig. |
| tests/atmosphere/hydrostatic/unit/test_mpas_vert4_flux_form.py / test_new_fields_are_appended_last | Assert main MPAS 3-field tail and Dycore 4-field tail including sponge | Assert MPAS 5-field tail with widths and Dycore 6-field tail with selector/widths | Branch MPAS 5-field tail; combined Dycore 7-field tail: filter, log filter, vertical form, sponge, selector, two widths; updated docstring | Test actual merged append layout and retain checks for every field from both parents; cannot retain either obsolete Dycore tail assertion literally. Same A vote. |
| tests/unit/test_run_amip_cli.py / end-of-file tests | Launcher SSO defaults, two-cell product cutoff helper/test/control, ML pipeline deck refusal | Tracer CLI/serialization/validation round trips, explicit smoothing decks, production tracer borrow on | Both complete test blocks unchanged | Independent regression coverage; no tests or planted violation discarded. |

## Detail vote and independent reviews

- Choice A = main spectral positional slot first, append branch tracer tail.
  Choice B = branch tail first, move main spectral slot. Codex voted A;
  GLM voted A (`vote_glm.txt`, served model glm-5.3 despite wrapper default glm-5.2).
- GLM's statement that branch positional calls would necessarily raise TypeError
  is not adopted: either ordering can misbind a competing parent's positional
  tail. No claim of preserving both positional ABIs is made.
- Original hunks: `conflicts.patch`; exact code pasted for vote/review:
  `vote_prompt.txt`, `review_prompt.txt`. Final review receipts recorded below.
- No new numerical helper, test case or production option was invented.
  New implementation search is not applicable to this union of existing code.
- UNASKED choices: none. No mechanical gate exists for choice/review policy;
  explicit votes and independent review receipts provide the audit trail.

## Auto-merge semantic audit (source inspection)

- `packages/core/legoesm/grids/vertical.py`: main changes only the sigma
  unknown-scheme error wording since merge base 2305947aef. Branch smooth van
  Leer reconstruction, independent widths, conservative SB helper unchanged.
  Exact main change retained in `auto_merge_audit.patch`.
- `packages/coupler/legoesm/driver/component_factory.py`: main adds only spectral
  sponge forwarding in this interval; branch global lane resolver and lat-lon /
  MPAS width forwarding remain. No new factory lane requiring resolver changes.
- Main does not add calls of `vertical_advection_hybrid` or
  `vertical_advection_hybrid_van_leer`; the existing generic, spectral, lat-lon
  and MPAS sites are recorded in `main_hybrid_callers.txt`. Main's spectral
  change is post-step sponge logic, independent of tracer transport.
- `validate_strict`: both spectral finite/nonnegative, lane and energy-fixer
  guards and branch selector/membership/width guards survive. Other main guards
  (ML pipeline and land snow validation) survive. No guard bypass introduced.
- Four production YAMLs: main's v3 land IC paths survive in all four; main C24
  SSO product and spectral hyperdiffusion/top sponge selections survive. Branch
  MPAS van_leer and lat-lon sb, both positive widths, null overrides on duo /
  spectral, and inherited conservative tracer borrow remain unchanged.
- Main appended one DycoreConfig field; no existing DycoreConfig defaults or
  pre-vertical_form order changed. Merge places that field before branch tail.
  Config round trips and driver constructions use field names. Positional
  consumers of the branch-only tail need the merged layout (inherent ABI debt).

## Verification

Tests run only on PBS develop@desched1, account UCLB0017, CPU,
JAX_ENABLE_X64=1. One process per requested file to avoid compiler limits.
Baseline: detached origin/main worktree `/glade/derecho/scratch/pg2328/wt_main_t20`;
only failed node IDs rerun there, with baseline-first PYTHONPATH, no stash.
Exact summaries and dispositions follow after job completion.

Independent Codex reviewer `/root/merge_review`: PASS; all four resolved hunks
preserve parent intents; no lost additions, numerical changes, new callers or
lanes. Reviewer identified inherent positional tail collision; documented above.
Existing `tests/unit/test_config_positional_abi.py` additionally run on PBS:
it scans real constructors and `_make`, plants a violation, and checks restart
serialization stays name-keyed. No claim that raw historical tuple pickles work.

GLM final resolved-hunk review: PASS (`review_glm.txt`, served glm-5.3).
Both reviews complete before merge commit. GLM's hypothetical claim that every
positional misbind fails loudly is not relied on: the positional AST audit is
our mechanical gate. PBS positional ABI summary:
`6 passed, 1 warning in 75.27s (0:01:15)`.

Requested suite summaries (verbatim pytest summary content):

- `tests_atmosphere_hydrostatic_unit_test_mpas_vert4_flux_form.py`: `18 passed in 22.47s`.
- `tests_atmosphere_hydrostatic_unit_test_primitive_eq_latlon_cgrid.py`: `1 failed, 73 passed in 240.56s (0:04:00)`.
- `tests_test_dispatch_hardening.py`: `24 passed in 16.19s`.
- `tests_test_no_inline_physics_coeffs.py`: `405 passed in 12.28s`.
- `tests_test_param_specs.py`: `81 passed in 9.78s`.
- `tests_test_validate_strict_coverage.py`: `76 passed in 5.76s`.
- `tests_unit_test_component_factory.py`: `90 passed, 1 warning in 22.65s`.
- `tests_unit_test_diag_hybrid_tracer_smoothing.py`: `2 passed in 0.55s`.
- `tests_unit_test_params_reachability_audit.py`: `7 passed in 4.54s`.
- `tests_unit_test_run_amip_cli.py`: `386 passed, 2 skipped in 30.01s`.
- `tests_unit_test_vertical_advection_hybrid_van_leer.py`: `35 passed in 60.02s (0:01:00)`.

Aggregate requested suites: {'passed': 1197, 'failed': 1, 'skipped': 2, 'errors': 0}.
All 11 files completed; only one failed test.

Failure: `tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py::TestTracerTransport::test_physics_tracer_tendency_applied`. Both candidate and main assert expected q_v=1.0 but produce 0.5025125558252623. Main rerun summary: `1 failed in 10.59s` (PBS 8248170.desched1, CPU/x64). This is a confirmed pre-existing failure, not fixed in this merge.

The first automated baseline attempt incorrectly converted a pytest class name into a file path (collection exit 4); that attempt is NOT evidence. The correct full node ID was rerun on main in the separate PBS job above, reproducing the same assertion and value. No other candidate failures exist in the 11 XML reports.

Test receipts: `/glade/derecho/scratch/pg2328/tmp/t20_results/`, `/glade/derecho/scratch/pg2328/tmp/t20_main_latlon.log`, `/glade/derecho/scratch/pg2328/tmp/t20_abi.log`. PBS suite job 8248154.desched1, host dec2439; ABI job 8248156.desched1. All JAX tests ran on CPU with JAX_ENABLE_X64=1.

UNVERIFIED: production integrations were not run; no scientific fidelity claim made. No new dependency APIs used. Static reviews and targeted tests establish merge compatibility within the requested coverage.
