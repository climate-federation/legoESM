# AeroBulk oracle for air-sea bulk-flux algorithms

Status: LIVE (2026-07-03). Gated by `tests/unit/test_aerobulk_oracle.py`.

## What and why

The turbulent air-sea flux algorithms (`core/bulk_flux.py::compute_most_fluxes`
schemes `coare3`/`large_yeager`, and the OMIP NEMO port
`ocean/bulk_flux_omip.py::air_sea_fluxes(algo="ncar")`) are validated against
**AeroBulk** (Brodeau et al. 2017, <https://github.com/brodeau/aerobulk>) — the
lineage ancestor of NEMO's `sbcblk`.  This upgrades the previous coverage:

| Scheme | Before | Now |
|---|---|---|
| core `coare3` | self-consistency only | flux-level vs `coare3p0` |
| core `large_yeager` | neutral LY09 Eq. 6 formula (5%) | flux-level vs `ncar`, incl. stability + 2/10 m heights |
| OMIP `ncar` | NumPy transcription mirror of NEMO 5.0.1 | + independent executable lineage check |

Two genuine formulation bugs were found and fixed at introduction
(commit `d2ced865`): the LY09 stability shift cancelled at z = 10 m
(ψ(z/L) − ψ(10/L) instead of ψ(z/L) on neutral reference coefficients), and
`coare3` lacked the wind-dependent Charnock ramp (high-wind drag ~15% low)
and the COARE ψ functions.

## Oracle trust chain

1. aerobulk-python (v0.4.0, conda-forge) is a binding to the actual AeroBulk
   Fortran, not a reimplementation.
2. Its `ncar` output was cross-checked against the *independent* NumPy NEMO
   5.0.1 mirror in `tests/ocean/unit/test_ncar_bulk.py` (already proven
   against the JAX OMIP port at rtol 1e-12): stress/evap agree to ≤7e-4,
   latent ≤4e-3, sensible ≤2.2e-2 (preprocessing-lineage level, i.e.
   aerobulk's θ/ρ conversion vs NEMO 5.0.1's Exner `pres_temp`).  A wrapper
   bug (units/argument order/signs) would show O(1) errors.

## Frozen baseline

- `tests/unit/baselines/aerobulk_noskin_v1.npz` (+ provenance `.json`):
  1024 seeded random + 8 structured marine points; noskin; `ncar` +
  `coare3p0`; (zt, zu) = (10, 10) and (2, 10); niter = 5 (NEMO / kernel
  default).  205 KiB — the tracked-tiny-numeric-baseline carve-out.
- Regenerate (only when intentionally revising the oracle; bump `_v1`):
  `mamba create -n aerobulk -c conda-forge python=3.11 aerobulk-python`
  then run `scripts/data/generate_aerobulk_reference.py` in that env.
  The parity tests never import aerobulk.

## Gate design (see test docstring for full detail)

Three tiers per output: **median** over all points (tight — the LY09 ψ bug
moved the stress median 100×), **p95** and **max** over the well-posed
regime.  Excluded from p95/max only (still in the median):

- **LY09 Stanton-branch bistability**: the discontinuous 18/32.7×10⁻³
  Stanton switch has no self-consistent branch where the Obukhov-length
  numerator nearly cancels (warm-but-dry air); NEMO/aerobulk's own 5-step
  iteration limit-cycles there and its answer is an iteration-parity
  artifact.  Detected via the two-branch sign/near-cancellation
  discriminator in the test fixture.
- **Calm winds** (< 1–3 m/s) and extreme stability: wind-floor
  (`Ub = max(W, 0.5)`), `UN10 = max(0.25, …)` collapse, C_d floors and
  ζ-clip conventions differ; absolute flux differences stay O(10 W/m²).

Known, accepted convention residuals (NOT masked, inside tolerances):
constant `c_pd` vs aerobulk's moist `cp_air(q)` in sensible (~1%); the
parity test passes SST-dependent `L_latent` explicitly, production
`constants.L_v` differs by up to ~3% at warm SST.

## Scheme-native defaults (AeroBulk parity, adopted 2026-07-03)

`compute_most_fluxes(gustiness_w_zi=None)` resolves per scheme: `coare3` →
600 m (gustiness is part of the COARE 3.0 algorithm; aerobulk `zi0=600`,
`Beta0=1.25`), others → off (LY09/NEMO ncar has no gustiness).  Explicit
`0.0` disables.  Scheme-native calm floors on the bulk wind: `coare3`
0.2 m/s, `large_yeager` 0.5 m/s (the reference implementations' values).
Stress is normalized by the bulk wind (τ = ρ u*² u/U_eff ≡ ρ C_d U_b u,
AeroBulk semantics; byte-identical when gustiness is off and no floor
binds).  The `None` sentinel is threaded through `SurfaceLayerConfig`,
`SimpleOceanConfig`, and the run_amip/run_coupled `--gustiness-zi` flags
(unset = scheme-native, `0` = force off).

## Open follow-ups

- Sea-ice bulk dispatch (`ice/sea_ice.py::_bulk_flux_dispatch`) has no
  oracle; AeroBulk has an ice branch (Andreas 2015 lineage) — candidate v2.
- `coare3p6`/`ecmwf` columns are free to add to the baseline if those
  schemes are ever implemented.
- ~~`return_2m` diagnostic still uses Businger-Dyer ψ_h for `coare3`~~
  RESOLVED: the diagnostic ψ_h is scheme-matched to the main loop
  (`psi_h_coare(zeta_d, stability_scheme)` for `coare3`, including the
  selectable stable branch — see `bulk_flux.compute_most_fluxes`).
