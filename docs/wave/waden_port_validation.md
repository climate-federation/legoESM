# Wave dynamics validation and reproduction

## Scope and acceptance protocol

The numerical reference is WADEN commit
`46aadc7834ad850442a535a08f94ca173e4b22e8` (MIT). The wave package remains
independent of atmosphere/ocean/land/ice; coupling belongs to the coupler.
The validation protocol carries forward the original registered cases and bars.

The source-free cases use WW3 6.07 archived output, identical initial fields,
physical depths, prescribed boundaries, timesteps and scoring domains. Learned
sources/transport, GSE and frequency tails are disabled explicitly in the
resolved config. Only the initial reference field is used to construct the
initial spectrum; future interior reference output is used for scoring only.

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

## Recorded results (2026-11-12)

The three reproducible cases were rerun on the current legoESM stack at the
revision recorded in each manifest (`docs/wave/results/waden_20261112/`).
float32, CPU, `JAX_ENABLE_X64=1`; the initial reference field only.

| Case | SWH RMSE | SWH max abs | Gate | Result |
| --- | --- | --- | --- | --- |
| tp1.1 | 3.862e-06 m | 2.915e-04 m | < 1e-4 m | pass |
| tp1.4 | 2.149e-03 m | 8.959e-03 m | < .008 m | pass |
| tp2.1 | 9.756e-06 m | 1.046e-03 m | < 2e-5 m / .002 m | pass |

The direct WADEN public-rollout comparison is at float32 round-off over the
full trajectory: tp1.1 is bit-identical, tp1.4 differs by 1.8e-08 (SWH
relative L2) and 7.1e-08 (spectrum), tp2.1 by 2.2e-06 and 3.7e-08. These are
descriptive; they carry no newly invented pass tolerance.

tp1.2 is not reproduced locally: its WW3 archive is absent on the machine used
for this port. Its gate above is inherited and unverified in this run; do not
describe it as passing until its archived inputs are available.

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
inputs; a public downloadable copy has not been verified. Exact input SHA256s
in the manifests identify the measured bytes. For review, request the bundle
`legoesm-wave-ww3-inputs-20261111.tar.gz` (SHA256
`4d25b9c5410dbe9a11e3acf0815b209196d66c8852fddf1935e66409bdb80a77`). Verify
that digest and extract it into an empty `ww3-inputs` directory; it contains
the four `ww3_tp*/input` directories. A public redistribution location and
data provenance/licensing record must be supplied before describing this as
independently accessible public evidence.

From a checkout with the workspace installed:

```bash
python scripts/experiment/install_federation.py --all --extras dev
mkdir -p ww3-inputs
tar -xzf /path/to/legoesm-wave-ww3-inputs-20261111.tar.gz -C ww3-inputs
for case in tp1_1 tp1_4 tp2_1; do
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m scripts.validate.validate_wave_ww3 \
    --config "scripts/validate/wave_fidelity/configs/${case}.json" \
    --data-root ww3-inputs --output "/tmp/wave-ww3/${case}"
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

The port keeps WADEN's hard TVD/positivity switches as the default; the
differentiability-versus-oracle-fidelity trade-off is explicitly waived for
this dynamics core. A globally smooth operator would change the numerical
model and needs a separate proposal and scientific validation.

## Coupling and release review

Tests cover the real atmospheric stress producer's sign, shore/no-inflow faces,
physical global edges, nonuniform frequency budgets, area-weighted GSE with
blocked transfers, GSE footprint refusal/subdivision, prescribed zero boundaries,
and odd-phase checkpoint continuation with identical next wave forcing.
Malformed shape/phase/config/weight metadata is rejected before ocean restore.
This is a short coupling/continuation check, not a controlled climate experiment.

Pretrained forecast compatibility, long coupled experiments, directional
Langmuir closure, and GPU/cross-version fidelity are separate work.
