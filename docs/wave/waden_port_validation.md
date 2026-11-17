# Wave dynamics validation and reproduction

## Scope and acceptance protocol

The numerical reference is WADEN commit
`46aadc7834ad850442a535a08f94ca173e4b22e8` (MIT). The wave package remains
independent of atmosphere/ocean/land/ice; coupling belongs to the coupler.
The validation protocol carries forward the original registered cases and bars.

The source-free cases use WW3 6.07 archived output, the same t=0 separable
reconstruction of the initial spectrum in the port and the WADEN reference,
physical depths, prescribed boundaries, timesteps and scoring domains. Learned
sources/transport and GSE are disabled explicitly in the resolved config; the
port selects no frequency-tail operator. Only the initial reference field is
used to construct the initial spectrum; future interior reference output is
used for scoring only. The bars are NeuralSWM's
`test/direct/test_real.py` gates, unchanged.

| Case | Coordinates | Active transport | SWH RMSE gate |
| --- | --- | --- | --- |
| tp1.1 | Spherical | Periodic x propagation | < 1e-4 m |
| tp1.2 | Spherical | Meridional propagation | < .01 m |
| tp1.4 | Cartesian | Propagation and depth refraction | < .008 m |
| tp2.1 | Cartesian | Two-dimensional propagation | < 2e-5 m |

The tp2.1 maximum absolute SWH difference must also be < .002 m. Every case
must have finite, nonnegative predicted spectra. Thresholds are inherited;
they are not adjusted to fit the port. Station spectral, directional, integral
and positivity-projection diagnostics are descriptive. Open/prescribed domains
exchange energy through boundaries, so their integral change is not itself a
conservation residual.

## Geometry update after the recorded validation

Wave-grid trigonometric fields now use unrounded sine and cosine, following
WADEN's removal of five-decimal geometry rounding. Spherical metrics and
transport directions therefore change from the version used for the archived
measurements below. Those manifests and rollout differences describe their
recorded revisions; they do not certify the updated geometry. Re-run the
external evidence lane before citing them for the updated implementation.

## Local PR preparation checks (2026-11-17)

Runtime revision: `057bb826d5fe36defc43108a0eed402e5e9befce`, after merging
upstream into `wave/06-validation-evidence`. The contribution author executed
the commands below and supplied the terminal summaries. This record is based
on those runs; the working tree was clean when the results were recorded.
Environment: Python 3.13.13, JAX/jaxlib 0.9.2, pytest 9.1.1,
build 1.6.1 and hatchling 1.33.0.

| Check | Backend / setting | Terminal result |
| --- | --- | --- |
| Wave unit suite | CUDA, GPU 0 (RTX 4080 SUPER), `JAX_ENABLE_X64=1` | 168 passed in 876.05 s |
| Coupling, restart, KPP and CLI suite | CPU, `JAX_ENABLE_X64=1` | 16 passed, 1 skipped in 249.54 s |
| Wave distribution build | Local Python, no build isolation | Wheel and sdist built successfully |

Commands, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -m pytest tests/wave/unit/ --tb=short -q

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
.venv/bin/python -m pytest \
  tests/wave/integration/ \
  tests/ocean/unit/test_kpp_langmuir.py \
  tests/unit/test_run_coupled_wave_cli.py \
  --tb=short -q

.venv/bin/python -m build --no-isolation packages/wave \
  --outdir /tmp/legoesm-wave-pr-dist
```

The build produced `legoesm_wave-1.0.1-py3-none-any.whl` and
`legoesm_wave-1.0.1.tar.gz`. The skipped test is the historical NeuralSWM
parity lane requiring external reference/weight artefacts. These are local
checks, separate from GitHub Actions results. They do not remeasure the
archived WW3/WADEN comparisons below; the contribution author explicitly
elected to defer that rerun for this preparation pass.

The startup-script merge conflict received two independent same-model
reviews. The named Claude/GLM cross-model pair review has not been completed;
the review arrangement remains an explicit maintainer decision.

## Recorded results (2026-11-12)

The four reproducible cases were rerun on the current legoESM stack at the
revision recorded in each manifest (`docs/wave/evidence/waden_20261112/`).
float32, CPU, `JAX_ENABLE_X64=1`; the initial reference field only.

| Case | SWH RMSE | SWH max abs | Gate | Result |
| --- | --- | --- | --- | --- |
| tp1.1 | 3.862e-06 m | 2.915e-04 m | < 1e-4 m | pass |
| tp1.2 | 9.080e-03 m | 1.962e-01 m | < .01 m | pass |
| tp1.4 | 2.149e-03 m | 8.959e-03 m | < .008 m | pass |
| tp2.1 | 9.756e-06 m | 1.046e-03 m | < 2e-5 m / .002 m | pass |

The direct WADEN public-rollout comparison is at float32 round-off over the
full trajectory: tp1.1 is bit-identical, tp1.2 differs by 1.1e-07 (SWH
relative L2) and 1.6e-07 (spectrum), tp1.4 by 1.8e-08 and 7.1e-08, tp2.1 by
2.2e-06 and 3.7e-08. These are descriptive; they carry no newly invented
pass tolerance. The float64 manifests are committed alongside the float32 ones
(`docs/wave/evidence/waden_20261112/fp64_tp1_2.json`, `fp64_tp1_4.json`,
`fp64_tp2_1.json`). They close tp1.2 and tp1.4 at 3e-16..9e-16 relative.
tp2.1 is 2.4e-08 (SWH) and 1.9e-14 (spectrum): the spectrum is at round-off,
and the SWH residual is the SWH functional's `1/(2 sqrt(m0))` sensitivity
amplifying cell-local spectral round-off (both sides use the same moments
routine), not a trajectory divergence. No new gate is introduced.

The committed manifests record the code revision, environment (CPU, x64,
default backend), resolved configuration with named fields, input SHA256s and
the full scoring output.

## Hermetic CI lane

The `Wave component` job in `.github/workflows/ci.yml` installs local
workspace members through the federation installer, builds the wave wheel and
sdist, runs the wave unit tests with x64 and with the default float32 mode,
and runs the real atmosphere/ocean/KPP/wave integration and checkpoint
continuation test plus the KPP Langmuir tests. These use small generated
inputs; external weights and WW3 archives are not required. The historical
NeuralSWM parity test skips explicitly without its external reference and
weight artefacts. WW3 comparisons belong to the separate lane below and are
not claimed as GitHub CI results.

## External WW3 evidence lane: data access

The reference files used for these measurements are author-supplied local
inputs; a public downloadable copy has not been verified. The tp1.1/tp1.4/
tp2.1 archives were read from the local WW3 tree; tp1.2 was read from the
NeuralSWM test archive at `~/NeuralSWM/test/data/ww3/ww3_tp1.2/input`.
Exact input SHA256s in the manifests identify the measured bytes. For review,
request the bundle `legoesm-wave-ww3-inputs-20261111.tar.gz` (SHA256
`4d25b9c5410dbe9a11e3acf0815b209196d66c8852fddf1935e66409bdb80a77`). Verify
that digest and extract it into an empty `ww3-inputs` directory; a usable
bundle must contain all four `ww3_tp*/input` directories. A public
redistribution location and data provenance/licensing record must be supplied
before describing this as independently accessible public evidence. The
`reference_revision` field in the case configs is the legacy archive identity
inherited from the source example (`b80fdce6...`); it resolves in no inspected
checkout and is recorded as a declared label, not as a verified commit.

From a checkout with the workspace installed:

```bash
python scripts/experiment/install_federation.py --all --extras dev
mkdir -p ww3-inputs
tar -xzf /path/to/legoesm-wave-ww3-inputs-20261111.tar.gz -C ww3-inputs
WW3_ROOT=ww3-inputs
for case in tp1_1 tp1_2 tp1_4 tp2_1; do
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m scripts.validate.validate_wave_ww3 \
    --config "scripts/validate/wave_fidelity/configs/${case}.json" \
    --data-root "${WW3_ROOT}" --output "/tmp/wave-ww3/${case}"
done
```

The runner records the imported source, commit/dirty state, input hashes,
resolved configuration, dtypes, backend, full-window scores and clipping.
Compare inputs by relative case/input filename and SHA256; absolute paths in
the committed manifests describe their execution host.

To compare with the WADEN public rollout, obtain a clean WADEN checkout at
`46aadc78` and add `--waden-root /path/to/waden` to each command. The runner
checks that revision, clean state and imported module path. Direct differences
are descriptive.

## Differentiability contract and limits

The production operator follows WADEN's piecewise differentiable TVD limiting,
positivity projection, signed transport and depth cutoff. It is compatible
with JAX tracing, JIT and differentiation within active branches. Switching
surfaces can have discontinuous/zero/convention-dependent derivatives; a
successful `jax.grad` call does not establish gradient quality there.

Dispersion has an implicit physical root derivative. A positive, nonuniform,
nonflat wet fixture tests the complete pure-physics step through environment
reconstruction using finite nonzero compiled gradients for spectrum and depth,
and forward/reverse finite-difference agreement away from switching surfaces
(order 1, eps 1e-6, rtol 1e-4, atol 1e-6). Fixed masks are held fixed. This
check gives no general guarantee at limiter switches or over long rollouts.

The port keeps WADEN's hard TVD/positivity switches as the default. The
contribution author explicitly chose this exception to RULE 5
(differentiability versus oracle fidelity) on 2026-11-12. Acceptance of this
scope under the upstream project's smooth-switch policy remains a maintainer
review decision. A globally smooth operator would change the numerical model
and needs a separate proposal and scientific validation.

## Coupling and release review

Tests cover the real atmospheric stress producer's sign, shore/no-inflow faces,
physical global edges, nonuniform frequency budgets, area-weighted GSE with
blocked transfers, GSE footprint refusal/subdivision, prescribed zero boundaries,
and odd-phase checkpoint continuation with identical next wave forcing.
Malformed shape/phase/config/weight metadata is rejected before ocean restore.
This is a short coupling/continuation check, not a controlled climate experiment.

Pretrained forecast compatibility, long coupled experiments, directional
Langmuir closure, and GPU/cross-version fidelity are separate work.
