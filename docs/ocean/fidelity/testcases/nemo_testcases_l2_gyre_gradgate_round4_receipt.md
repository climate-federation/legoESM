# Certified GYRE reverse-mode gradient gate — round 4 receipt

Date: 2026-10-11

Status: **LANDED (one-step CPU agreement confirmed; resource debt retained)**.
The user-authorized active-cell denominator treatment removes the certified
GYRE card's one-step reverse NaN without changing any bit of its ten-step
certified trajectory. Reverse mode agrees with forward mode and central finite
differences at the unchanged bars, and the deliberately broken VJP still turns
the row red. The six-step CPU compiler row and every CPU/GPU resource row
remain explicitly unmeasured.

## Provenance and preregistration

The incoming lane tip was
`d678430f3e473286e6f15c41e4bbadc856aa57db`. Round 4 was preregistered before
the pre-fix controls at clean commit
`bad01873cbc2d2757a70cf48be47b9e10eb987ac`; the implementation under
measurement is clean commit
`31a31cadf30ae42463aa8c30e9255e94256de53a`. The committed preregistration is
`docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_gradgate_round4_prereg.md`;
its external copy has SHA-256
`8f7c4a31f47ffe57ec22e001afb55dd6de960916fcb14151cb60917de0f2ac0e`.

All numerical evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round4/`.
Agreement rows use CPU fp64/libm, the recorded GYRE day-0 state, production
`LatLonCGridOceanModel._step_jitted`, the frozen four-field normalized
after-state loss, and the registered random direction. The unchanged bars are
reverse/forward relative error at most `1e-9` and reverse/central-FD relative
error at most `0.05`, with central-FD step `1e-3`.

## CPU agreement and plant

| run | steps | reverse | forward | central FD | rev/fwd relative | rev/FD relative | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| pre-fix control | 1 | NaN | `-3.884979632703889e-06` | `-3.884977393501259e-06` | NaN | NaN | **REFUTED, as predicted** |
| active-cell denominator | 1 | `-3.884979632703882e-06` | `-3.884979632703889e-06` | `-3.884977393501259e-06` | `1.9622487750297417e-15` | `5.763743530264072e-07` | **CONFIRMED** |
| deliberately doubled VJP | 1 | `-7.769959265407763e-06` | `-3.884979632703889e-06` | `-3.884977393501259e-06` | `0.499999999999999` | `0.5000002881871766` | **REFUTED; exit 1** |
| active-cell denominator | 6 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | **CPU LLVM ABORT; CI SKIP** |

The one-step evidence files have SHA-256 respectively
`c4e2a38889e40afb7a53532149bc6b94eb8e0da1dcf70d9b42145a5ba3b50e50`,
`ec494224700b91ebc756f62a69be12ae501995e471b68218b1e197fd5fcc6934`,
and `41526b9427b9c57b84f7eadd0e55a0398b338c4fa3470860846dc0116729eb80`.
The plant preserves forward mode and finite differences while doubling only
the reverse derivative, so its required nonzero exit is a live control rather
than an unrelated failure.

The six-step parametrization skips before tracing. The exact retained compiler
failure from the isolated Round 2 run is:

```text
LLVM compilation error: Cannot allocate memory
contiguous_section_memory_manager.cc:101] allocateMappedMemory failed with error: Cannot allocate memory
Failed to satisfy suballocation request for 3791
LLVM ERROR: Unable to allocate section memory!
```

It is the known XLA-CPU large-function section-memory failure, not a numerical
gradient verdict. The production program's six-step reverse, forward, and FD
values are therefore not inferred from the one-step result.

## Authorized statement change and oracle citation

The implementation at
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1755-1801`
constructs `active_t` from the tracer mask, selects the real `e3t_kmm` on
active cells and `1.0` on inactive storage, and explicitly selects positive
zero outside the active domain after each contribution. Both horizontal and
vertical CEN2 divisions use that denominator. The direct test at
`tests/ocean/unit/test_nemo_ws_tracer_rk3.py:143-226` retains bitwise equality
to the existing literal computation on active cells and proves that a pullback
seeded only on an inactive zero-thickness cell returns finite, exact-zero
cotangents for every operand.

This matches the compiled NEMO program's execution domain and statements:

- `cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv.F90:295-298,370-380` disables FCT in
  RK3 stages 1 and 2 and dispatches those stages to `tra_adv_cen`.
- `cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv_cen.F90:137-149` performs the live
  `e3t(Kmm)` horizontal division inside NEMO's owned-cell loop.
- `cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv_cen.F90:203-216` performs the matching
  vertical divisions on the same owned-cell domain.

No wet-cell arithmetic, card configuration, timestep, objective, direction,
tolerance, clipping, `stop_gradient`, or stabilizer changed. The former NaN
was a transpose of inactive cropped storage, not an executed NEMO wet-cell
operation.

## Certified primal identity

The same CPU fp64/libm trajectory-only command was run before and after the
change through step 10. All 70 JSON objects carrying `name`, `status`, and
`absolute_max` compare exactly: 9 remain `AT-BAR`, 58 remain `DEBT`, and 3
remain `UNINFORMATIVE`. Existing fidelity debt was neither hidden nor
reclassified. The complete residual NPZ files are byte-identical and share
SHA-256
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
The pre/post JSON SHA-256 values are respectively
`ce5db19d0fc4d771d8fcd5e74076e2bd4438012405bef0d075b9d0b67fcaf510`
and `055c7489b6e6558f8f720975b23d7d31cb2b40d9e2ea239f91b662459c0274fa`;
their provenance differs, while their registered result rows do not.

The incoming tip also carried stale ORCA2-era resolved-card digest expectations
for GYRE, LOCK_EXCHANGE, and OVERFLOW. That pre-existing test failed before the
CEN2 edit even though `git diff d678430f3..31a31cad` changes no card builder or
configuration. Commit `49e8c1905996d865e85cc8136d7d5cd0f31f36b5` preserves the historical
pins and records the incoming/post-change complete-card digests:
`99b2dfce324b0f77`, `70746f41480350ec`, and `2c562be8a7d2e793`.
The digest test remains fail-closed.

## Memory, wall time, and GPU rows

| platform/policy | schedule | 1-step reverse peak/time | 6-step reverse peak/time | 360-step extrapolation |
|---|---|---|---|---|
| CPU fp64/libm | none | UNMEASURED | UNMEASURED | UNMEASURED |
| CPU fp64/libm | uniform | UNMEASURED | UNMEASURED | UNMEASURED |
| CPU fp64/libm | binomial | UNMEASURED | UNMEASURED | UNMEASURED |
| GPU 1 fp64/native | none | UNMEASURED | UNMEASURED | UNMEASURED |
| GPU 1 fp64/native | uniform | UNMEASURED | UNMEASURED | UNMEASURED |
| GPU 1 fp64/native | binomial | UNMEASURED | UNMEASURED | UNMEASURED |

No compiler allocation, process RSS, estimate, or one-step trajectory storage
is mislabeled as reverse-pass peak memory. Therefore this round still cannot
state a 60-day/360-step memory requirement. No GPU result JSON existed under
`gradgate_rounds/` when the round started. The committed operator runner remains
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`;
it must run on GPU 1 with explicit
`PrecisionPolicy.fp64(transcendentals="native")`, and every result remains
labeled **“GPU, XLA transcendentals — not the certified bit-exact policy.”**

## Focused tests

- Full CPU step-gradient matrix: **10 passed, 1 skipped in 1088.15 s**. All
  nine legacy variants and the promoted one-step GYRE CONFIRMED row pass; the
  six-step pre-compilation skip is the only skip. Pytest exits normally, so
  the former fatal LLVM abort no longer kills the battery.
- CEN2/RK3 and gradient-gate direct tests: **44 passed in 1799.40 s**. This
  includes the inactive-cell exact-zero VJP control and the production broken-
  VJP plant.
- Recipe, DINO, admitted testcase-card, ORCA2, VORTEX/VORTEX_VEC, and resolved
  card-hash checks: the first combined run had **40 passed, 3 pre-existing
  stale digest failures**; after the fail-closed re-pin, those three digest
  parametrizations passed. The clean combined rerun had **43 passed in
  334.58 s**.
- The repository has no `TSUNAMI-zco` builder, admitted record, receipt, or
  focused test at this branch tip. It was not synthesized or substituted; its
  requested card row remains OPEN until that card exists.
- Receipt citation gate: **PASS**, 274 citations; all seven violation plants
  and both clean controls fired as expected.

## Review status

The campaign's required two independent adversarial reviews remain
**UNREVIEWED**. The available Codex read-only reviewer command refused during
initialization with `Read-only file system (os error 30)`; no independent
review claim is made.

## OPEN

1. Run
   `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`
   from an operator shell on GPU 1. One- and six-step GPU agreement, peak
   device memory, and wall time remain unmeasured.
2. Measure CPU reverse peak memory/wall time for the three checkpoint
   schedules without triggering the six-step XLA-CPU large-function abort,
   then measure rather than assume the 360-step extrapolations.
3. Generalize the gate selector to VORTEX-zco and VORTEX_VEC-zco, and to
   TSUNAMI-zco after that certified card and its admitted recorded entry state
   exist. This is the deferred round-5 scope; no extra physics was mixed into
   this round.
4. Complete the two independent adversarial reviews before promoting these
   measurements beyond this receipt.

## Choice audit

- ASKED: the approved active-cell double-`where` denominator for both CEN2
  divisions; pre/post GYRE trajectory identity; unchanged one-step gradient
  bars and plant; a live CPU pytest battery; certified-card focused checks;
  citation gate; and this receipt.
- UNASKED: none. The stale digest pin repair restores an incoming red
  fail-closed test and makes no configuration or model change. No six-step or
  memory number is inferred, and no optional additional-card gradient program
  was started.
