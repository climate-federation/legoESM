# Fast Spectral-Bin Microphysics (FSBM-2) — faithful port spec

**Oracle**: WRF `phys/module_mp_fast_sbm.F` (FSBM-2, Hebrew University Cloud
Model; Khain et al. 2004 JAS 61:2963; Shpund et al. 2019 JGR 124:9800). Local
copy: `/tmp/wrf_sbm/module_mp_fast_sbm.F` (9053 lines, not committed).

**Goal**: switchable `MicrophysicsConfig.scheme="fast_sbm"` Eulerian bin
scheme, differentiable end-to-end (`jax.grad`), reusing repo helpers
(`legoesm.thermo`, `constants`, `microphysics.output.sedimentation_tendency`,
SDM kernels where physics overlaps).

## Oracle structure (what we are porting)

Four size distributions on mass-doubling bins (`m_{k+1}=2m_k`):

| Distribution | Bins | Notes |
|---|---|---|
| Liquid drops (cloud+rain) | NKR=33 | single liquid spectrum; KRDROP=15 (~50 um) cloud/rain split for diagnostics |
| Snow/ice crystals | 33 | with rime fraction `RF3R` |
| Graupel **or** hail | 33 | `hail_opt=1` default (hail fall speeds) |
| Aerosol (CCN) | NKR_aerosol=43 | 3-lognormal init (nucleation/accumulation/coarse) |

Key oracle constants: `COL=0.23105=ln2/3` (log-radius increment), quadrature
width `dm=3·COL·x=ln2·x`, moments `QC=(1/ρ)Σ3·COL·f·x²`, `QNC=(1/ρ)Σ3·COL·f·x`
(FAST_SBM lines 4993–4996). CGS internally; our port is SI with CGS only in
oracle-diff tests.

Process inventory (oracle subroutine → port module → status):

| Process | Oracle | Port | Status |
|---|---|---|---|
| Bin grid + moments | parameter block, QC/QNC diags | `fast_sbm/grid.py` | **done (iter 1)** |
| Collision kernels (in-code, not files) | `Kernals_KS` (l. 6238) | `fast_sbm/kernels.py` | todo |
| Collision-coalescence (Bott flux) | `coll_xxx_lwf` + `courant_bott_KS` | `fast_sbm/collision.py` | todo |
| Diffusional growth (cond/evap dep/sub) | `JERRATE_KS`→`JERTIMESC_KS`→`JERSUPSAT_KS`→`JERDFUN_KS`/`JERNEWF_KS` | `fast_sbm/diffusional_growth.py` | todo |
| Drop nucleation (CCN activation) | `JERNUCL01_KS`, `WATER_NUCLEATION`, `LogNormal_modes_Aerosol` | `fast_sbm/nucleation.py` | todo |
| Freezing/melting | `FREEZ`, melting block in FAST_SBM | `fast_sbm/ice_phase.py` | todo |
| Breakup (collisional + spontaneous) | `coll_breakup_KS`, `Spont_Rain_BreakUp` | `fast_sbm/breakup.py` | todo |
| Sedimentation per bin | fall-speed tables `VR1..VR5` + advection in FAST_SBM | `fast_sbm/sedimentation.py` (reuse `output.sedimentation_tendency`) | todo |
| Column driver + scheme wiring | `FAST_SBM` subroutine | `fast_sbm/column.py` + `integration.py` dispatch | todo |

**Lookup-table strategy**: WRF reads tables (`capacity33.asc`, masses,
terminal velocities, kernels `YW*`, breakup `PKIJ/QKJ`) from data files NOT in
the WRF git repo. Faithfulness plan: (a) grid masses derived analytically
(doubling from 2 um — reproduces documented landmarks), (b) collision kernels
via in-code `Kernals_KS` path, (c) terminal velocities from published
formulations already in `sdm/kernels.py` where identical, else ported, (d)
capacities analytic (sphere/oblate per category). Any residual table needed →
small committed `.npy` with provenance, like RRTMGP.

**Differentiability strategy**: oracle's remap (`JERNEWF_KS`) and Bott scheme
use integer bin shifts + positivity clamps — port keeps the *algorithm* but
implements branches as `jnp.where` with smooth-safe formulations; hard
`wrf_error_fatal` guards become finite clamps documented per-site. No
`lax.cond` on traced data for bin loops — vectorized over bins.

**Physics contracts**: every tendency-producing module ships
`__physics_contract__` + acceptance test before the body (CLAUDE.md
guardrails). Helpers (`grid.py`) live in `test_physics_contracts.py`
EXCLUDED.

## Iteration log

### Iter 1 (2026-06-11)
- Fetched oracle (9053 l). Mapped subsystem: dispatch in
  `microphysics/integration.py` (hardened ValueError), SDM = subpackage
  precedent, `MicrophysicsOutput` union container.
- Extracted oracle conventions: NKR=33/43, COL=ln2/3, dm=3·COL·x, f in
  cm⁻³g⁻¹ (→ SI m⁻³kg⁻¹), QC/QNC quadrature, KRDROP=15.
- **`fast_sbm/grid.py`**: `mass_doubling_grid` (bitwise-exact doubling),
  `radius_from_mass`, `bin_mass_widths` (=ln2·m exact), `number_density`,
  `mass_density`, `discretize_lognormal` (exact CDF-difference number
  projection; promoted `sdm.init.lognormal_cdf` to public for reuse).
- Tests `tests/atmosphere/microphysics/unit/test_fast_sbm_grid.py`: oracle
  landmarks (2 um, bin15≈50 um, top≈3.25 mm), quadrature == oracle formula,
  lognormal number exact + mass vs analytic 3rd moment (midpoint bias
  +ln²2/24 = +2.0% bracketed explicitly), truncation bounds, `jax.grad`
  correctness (linear-moment gradient == m·dm), jit + float32, ValueError
  guard.
- Tracer-state converters `f_from_bin_mixing_ratios`/`bin_mixing_ratios_from_f`
  (oracle carries per-bin mixing ratios ``chem_new``; ``f = q ρ_air/(m dm)``
  = its ``ρ/(3 COL x²)``) + round-trip/QC-sum test. (Surfaced by the codex
  pass before it hung; review re-run scheduled with iter 2.)
- Codex adversarial review attempt 1 hung after ~40 min (no log progress);
  cancelled, findings up to hang folded in; full review re-runs at iter 2.
