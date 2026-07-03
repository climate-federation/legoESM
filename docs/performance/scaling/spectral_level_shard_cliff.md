# Spectral grid — level-shard scaling cliff (theoretical limit)

**Verdict: the spherical-harmonic spectral dycore does NOT scale across devices
under level-axis sharding — it is single-device by design.** This is the spectral
grid's theoretical limit on the only horizontal-decomposition-free sharding
scheme the codebase exposes, and it is the reason `create_level_mesh` reports
`_valid_gpu_counts -> [1]` and `DeviceConfig.is_distributed = False` for spectral.
Measured here to replace the asserted "by-design cliff" with data.

## Measurement (job 8520392, CPU, x64, nlev=24, 120 timed steps)

`scripts/bench/probe_spectral_shard.py` times `SpectralPrimitiveEquationModel`
SSP-RK3 steps with the level axis (`P(None, "level")`) sharded across `N`
emulated CPU devices, vs the single-device `none` baseline.

| truncation | none N=1 | level N=1 | level N=2 | level N=4 | speedup @ N=4 |
|------------|---------:|----------:|----------:|----------:|--------------:|
| **T42**    |   5.0413 |    4.8553 |    5.0324 |    5.3475 | 1.06× (flat)  |
| **T85**    |   0.7369 |    0.7175 |    0.5764 |    0.5484 | **0.74× (anti-scaling)** |

(steps/s; higher is better.) Adding devices never helps: T42 is flat within
run-to-run noise (~±6 %), and the larger T85 problem actively **slows down** as
devices are added.

## Why it cannot scale (mechanism)

The spectral state is `(n_sh, nlev)`. Two facts make level sharding a dead end:

1. **The semi-implicit solve couples all vertical levels.** Each zonal wavenumber
   carries a dense `(nlev, nlev)` implicit operator (gravity-wave / hydrostatic
   coupling). Sharding levels across devices forces an **all-gather of the level
   axis** before every SI solve — pure communication with no parallel speedup,
   because the solve itself cannot be decomposed across the sharded axis.
2. **The SH transforms emit no per-shard collectives.** `sh_analysis_3d` /
   `sh_synthesis_3d` batch/chunk the level axis for *memory* only, so the
   transform is replicated work per shard, not parallel work.

The anti-scaling worsens with resolution because the all-gather volume grows with
`n_sh`: T85 has ~4× the wavenumbers of T42 (3741 vs 946), so the level
all-gather is ~4× larger while the SI solve stays un-parallelized — hence T85's
clean 0.74× cliff vs T42's flat curve.

## What scaling spectral would actually require

The standard scalable spectral-transform parallelization is the **transpose
method** (e.g. IFS): domain-decompose grid space by latitude for the FFT/physics,
then **all-to-all transpose** to decompose by wavenumber for the Legendre
transform and SI solve. That avoids the all-gather but replaces it with an
all-to-all every transform — the most communication-intensive collective. On this
cluster's Gloo/TCP fabric (no NVLink/IB GPU-direct) that is comm-bound and
anti-scales, consistent with the campaign's multi-node verdict
(`docs/performance/scaling/distance_to_limit_2026-06-13.md`). It is therefore out of scope as
a Ginsburg throughput win; the single-device verdict stands as the practical
limit here.

## Precision note

Spectral is **float64-only by physics**: the transforms use `complex128` FFTs and
the precision policy forces `spectral_transform` to f64 storage/compute/accumulate
(`core/precision.py::_ATMOSPHERE_OVERRIDES`). There is no f32 spectral path, so
the "f32 vs f64" scaling axis is N/A for this grid — unlike the finite-volume
cube / lat-lon / icosahedral grids, which carry both precisions.

## Reproduce

```
sbatch scripts/cluster/scaling_ginsburg/spectral_cliff_probe.sbatch   # -> CSV
sbatch scripts/cluster/scaling_ginsburg/spectral_cliff_plot.sbatch <probe_job_id>
# -> docs/performance/scaling/spectral_level_shard_cliff.png
```

Plotter: `scripts/plot/plot_spectral_level_shard.py` (tested,
`tests/plot/test_spectral_level_shard_plot.py`). Correctness of the level-shard
*numerics* (independent of throughput) is gated by
`tests/parallel/test_spectral_level_shard.py` (single-device == level-sharded to
1e-12 at T21/T42, 2 and 4 devices).
