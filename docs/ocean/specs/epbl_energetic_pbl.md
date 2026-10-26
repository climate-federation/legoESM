# ePBL — Energetic Planetary Boundary Layer vertical mixing (MOM6 port)

## What it is
A JAX port of MOM6's `MOM_energetic_PBL.F90` (Reichl & Hallberg 2018, *Ocean
Modelling* 132, 112-129), selectable as a vertical-mixing closure alongside
`kpp` and `catke`:

```python
from legoesm.ocean.physics.vertical_mixing.config import (
    EPBLConfig, VerticalMixingConfig)
vmix = VerticalMixingConfig(scheme="epbl", epbl=EPBLConfig(dt_s=1200.0))
```

ePBL sets the boundary-layer diffusivity from an energy budget instead of a
prescribed shape. Each timestep it

1. converts the wind stress into mechanical TKE
   (`mstar * rho_0 * u_star^3 * dt`) and the surface buoyancy forcing into
   convectively released potential energy;
2. walks down the column, proposing `Kd = vonKar * vstar * mixlen` at every
   interface, pricing the potential-energy change of mixing implicitly with it,
   and paying for that change from the remaining TKE. Where the energy runs out
   mixing stops; that depth is the diagnosed boundary-layer depth;
3. iterates on the boundary-layer depth, because the mixing-length shape
   depends on it (`use_mld_iteration`).

It can never mix more than the forcing supplies.

## Files
| File | Content |
|---|---|
| `packages/ocean/legoesm/ocean/physics/vertical_mixing/epbl.py` | scheme: mstar schemes, column solver, `epbl_vertical_mixing`, `epbl_diagnostics`, `surface_tke_forcing`, `validate_epbl_config`, `__physics_contract__` |
| `packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py` | `EPBLConfig` (+ its `__param_spec__`), `"epbl"` in `VALID_VERTICAL_MIXING_SCHEMES` |
| `packages/ocean/legoesm/ocean/physics/vertical_mixing/integration.py` | `_make_epbl` factory (builds dSV/dT, dSV/dS, surface buoyancy flux, top-layer TKE forcing) |
| `packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py` | `epbl` branch of `compute_vertical_K_profiles` (3-D implicit path) |
| `packages/ocean/legoesm/ocean/scm.py` | ocean SCM honours `k_after_surface_flux` (see below) |
| `tests/ocean/unit/test_epbl.py` | unit tests (kernels, AD, jit/eager parity, dispatch guards, SCM ordering option) |

## Configuration
`EPBLConfig` is field-for-field the MOM6 `MOM_energetic_PBL` parameter block,
**with MOM6's own defaults except `vonkar`**: legoESM uses 0.4 (the repo's
core von Karman value) where MOM6 uses 0.41. The MOM6 name of each field is
in its comment.
Main groups:

| Group | Fields (MOM6 name) |
|---|---|
| timestep | `dt_s` (no MOM6 equivalent — the timestep the energy budget is priced over; the OMIP driver and the SCM set/check it against their dt, the lat-lon C-grid model ignores it and passes its step dt; see below) |
| rotation / decay | `omega_frac` (ML_OMEGA_FRAC), `tke_decay` (TKE_DECAY), `ekman_scale_coef` |
| energy partition | `nstar` (NSTAR), `mke_to_tke_effic`, `wstar_ustar_coef` (WSTAR_USTAR_COEF) |
| mstar | `mstar_scheme` = `constant` / `om4` / `reichl_h18` (EPBL_MSTAR_SCHEME), `fixed_mstar` (MSTAR), `mstar_cap`, `mstar_coef`, `c_ek`, `mstar_convect_coef` (MSTAR_CONV_ADJ), `rh18_mstar_cn1..3`, `rh18_mstar_cs1..2` |
| MLD iteration | `use_mld_iteration`, `mld_bisection`, `mld_iter_bug`, `max_mld_its`, `mld_tol` (EPBL_MLD_TOLERANCE), `mixlen_exponent`, `translay_scale`, `min_mix_len`, `vonkar` |
| velocity scale | `wt_scheme` = `cube_root_tke` / `reichl_h18` (EPBL_VEL_SCALE_SCHEME), `vstar_scale_fac`, `vstar_surf_fac` |
| equation-discovery options (Sane et al. 2026) | `eqdisc_shape`, `eqdisc_v0`, `eqdisc_v0h` and their caps |
| backgrounds | `K_bg` (KD), `A_bg` (KV) |
| operator-split ordering | `k_after_surface_flux` (see below) |

Requirements and guards (`validate_epbl_config`, all raise on violation):
- **implicit only** — the host must run `implicit_vertical_mixing=True`; ePBL
  returns `K_v`/`A_v` for the backward-Euler solve and has no explicit form;
- `dt_s > 0` (`validate_epbl_config`; checked on every path, including the
  lat-lon C-grid model, which does not otherwise read `dt_s`). Where `dt_s`
  is read it must equal the model timestep (ePBL is an energy-per-timestep
  closure), which the drivers enforce, not `validate_epbl_config`: the OMIP driver sets it from `--dt`, refuses to run
  ePBL without an explicit `--dt`, and raises if it differs from the run
  timestep; the ocean single-column model sets it to its own `dt` when
  `dt_s` is left at the default and raises when a non-default `dt_s`
  disagrees with that `dt`. The lat-lon C-grid model does not read `dt_s`:
  it passes its step dt to ePBL at step time, the same dt CATKE gets on that
  lane (the momentum dt: `2*dt` on leapfrog steps, `dt` on the Euler start),
  and refuses ePBL with `dt_mom_ratio != 1`; the lat-lon tendency probe does
  the same (momentum dt, and refuses a tracer dt that differs from it). Any
  other caller of the physics factory must set `dt_s` itself (nothing checks
  it there);
- unknown `mstar_scheme` / `wt_scheme`, and `eqdisc_v0` together with
  `eqdisc_v0h`, raise.

### JAX-specific notes
- Every state-dependent branch is a `jnp.where`; MOM6's two iterations are
  fixed-trip-count `lax.fori_loop`s that freeze once converged, so the scheme
  is reverse-differentiable. Consequence: `max_mld_its` / `max_kd_its` are
  executed in full every step — size them from the tolerance (MOM6's
  `EPBL_MLD_MAX_ITS = 500` costs nothing in Fortran but would here).
- Guarded `sqrt`/`pow`/cube-root keep gradients finite at zero forcing.
- Cost: the column descent is sequential in depth. In a global 1-degree,
  40-level run on one A100 ePBL took ~0.24 s/step vs ~0.03 s/step for KPP.

### Not ported (deliberate)
Non-Boussinesq arithmetic; `DIRECT_EPBL_MIXING_CALC`; Langmuir turbulence
(`EPBL_LT`, `USE_LA_LI2016`); the bottom boundary layer (`ePBL_BBL_column`);
pre-2024 `EPBL_ANSWER_DATE` variants; a sub-surface `TKE_forced` profile from
penetrating shortwave (the TKE forcing is applied to the top layer only).

## Operator-split ordering flag: `k_after_surface_flux`
legoESM and MOM6 order the column step differently:

- **legoESM (default, same as KPP):** the diffusivity is diagnosed from the
  step-entry state, then the surface flux is applied and the column is mixed.
- **MOM6:** the surface flux is applied first (`applyBoundaryFluxesInOut`),
  then ePBL diagnoses the diffusivity on the post-flux state.

Both orderings converge to the same answer as `dt -> 0`; at finite `dt` they
differ under strong surface heating or cooling. `EPBLConfig.k_after_surface_flux`
selects the ordering:

| Value | Ordering | Use |
|---|---|---|
| `False` (default) | legoESM / KPP order | normal runs |
| `True` | MOM6 order (K re-diagnosed on the post-flux state; one extra physics evaluation per step) | **comparing the legoESM single-column model against the MOM6 single-column model** |

The flag is honoured by the ocean single-column model (`OceanColumnModel`)
only. Every other driver raises `NotImplementedError` if it is set, so it can
never be silently ignored.

```python
cfg = EPBLConfig(dt_s=1200.0, k_after_surface_flux=True)   # MOM6 ordering
```

## Single-column test case (legoESM SCM vs MOM6 SCM)
Validation of the port against MOM6 itself, with matched constants
(`G=9.80616`, `RHO_0=1025`, `C_P=3994`, `OMEGA=7.292e-5`) and the Wright EOS.

**Case**

| Setting | Value |
|---|---|
| Column | 50 m deep, 1 m layers (50 levels) |
| Latitude | 40 N (f = 9.3744e-5 1/s) |
| Wind stress | `tau_x` = 0.2 N/m^2, `tau_y` = 0 |
| Surface heat flux | -200, 0, +200 W/m^2 (non-penetrating, no freshwater) |
| Initial state | T linear 20 -> 15 degC over 50 m (MOM6 `TS_CONFIG="linear"`), S = 35 psu, at rest |
| Timestep | 1200 s (also run at 300 s) |
| Physics | ePBL only (no convection scheme, no shortwave penetration), small backgrounds `K_bg=1e-5`, `A_bg=1e-4` m^2/s |
| ePBL options | `mstar_scheme="reichl_h18"`, `wt_scheme="reichl_h18"`, `nstar=0.06`, `omega_frac=0.001`, `tke_decay=0.001`, `mstar_convect_coef=0.66`, `rh18_mstar_cn3=-6.0`, `wstar_ustar_coef=15`, `fixed_mstar=0`, `mixlen_exponent=1.0`, `translay_scale=0.01`, `vstar_scale_fac=0.5477`, `vstar_surf_fac=1.8258`, `use_mld_iteration=True`, `mld_bisection=True`, `mld_tol=0.1`, `eqdisc_shape=True`, `eqdisc_v0h=True` |
| Compared | temperature profile after 8 hours |

MOM6 reference: `ocean_only` single column (`BUOY_CONFIG="const"`,
`WIND_CONFIG="const"`), ePBL as the only mixing, the same parameters as above
in `MOM_override`.

legoESM side:

```python
from legoesm.ocean.scm import OceanColumnModel
from legoesm.ocean.scm_forcing import OceanSCMForcing
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.vertical import create_ocean_z_star

epbl = EPBLConfig(dt_s=1200.0, k_after_surface_flux=True, ...)   # options above
phys = OceanPhysicsConfig(
    vertical_mixing=VerticalMixingConfig(scheme="epbl", epbl=epbl),
    convection=OceanConvectionConfig(scheme="none"),
    shortwave_penetration=None)
zc = create_ocean_z_star(n_levels=50, H_max=50.0, dz_surface=1.0, dz_deep=1.0)
forcing = OceanSCMForcing(f_c=f_40N, tau_x=lambda t: 0.2, tau_y=lambda t: 0.0,
                          q_net=lambda t: Q)
scm = OceanColumnModel.create(
    nlev=50, dt=1200.0, T_profile=T0, S_profile=35.0, physics_config=phys,
    z_coord=zc, latitude_deg=40.0, forcing=forcing,
    implicit_vertical_mixing=True,
    K_v_background=0.0, A_v_background=0.0)   # ePBL carries its own K_bg/A_bg
final, history = scm.run(nsteps=24, save_every=1)
```

**Result** — maximum |T_legoESM - T_MOM6| over the column at hour 8 [degC]:

| Heat flux | dt | legoESM order (default) | MOM6 order (`k_after_surface_flux=True`) |
|---|---|---|---|
| -200 W/m^2 | 1200 s | 0.124 | 2.5e-4 |
| -200 W/m^2 | 300 s | 0.050 | 2.6e-5 |
| 0 | 1200 s | 1.1e-7 | 1.1e-7 |
| 0 | 300 s | 5.2e-8 | 5.2e-8 |
| +200 W/m^2 | 1200 s | 0.404 | 1.1e-3 |
| +200 W/m^2 | 300 s | 0.176 | 1.3e-4 |

- With MOM6's ordering the port reproduces MOM6 to ~1e-3 degC or better in all
  cases; the neutral case (where ordering is irrelevant) matches to ~1e-7 degC.
- With legoESM's default ordering the gap is a splitting error, not a scheme
  error: it shrinks as `dt` decreases (by ~2.3-2.5x from 1200 s to 300 s; the
  maximum sits at the mixed-layer base, which moves in whole-layer jumps on the
  1 m grid, so convergence is below first order).

## 3-D use (OMIP driver)
`scripts/run/run_omip.py` accepts `--vertical-mixing-scheme epbl`; it requires
`--implicit-vertical-mixing` and an explicit `--dt`, which becomes
`EPBLConfig.dt_s`. Example (1-degree lat-lon, JRA55-do forcing):

```bash
python scripts/run/run_omip.py --grid latlon --resolution 180x360 --nlev 40 \
  --dt 1800 --vertical-mixing-scheme epbl --implicit-vertical-mixing \
  --forcing-mode jra55_do_tropical --jra55-cache <cache.zarr> \
  --woa-init --woa-t <T.nc> --woa-s <S.nc> --monthly-mld-sst ...
```

Related driver options added with this work: `--woa-t-var`, `--woa-s-var`,
`--woa-month` (read e.g. WOA13 monthly `ptemp_an`/`s_an`), and
`--monthly-mld-sst` (monthly-mean SST and de Boyer Montegut MLD written to
`<output>/<grid>/<res>/monthly_mld_sst.nc`, lat-lon JRA55-do runs).

## Tests
```bash
JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/ocean/unit/test_epbl.py
JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/unit/test_run_omip_cli.py
```
On HPC CPU nodes, compile with single-threaded XLA
(`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`)
to avoid out-of-memory during LLVM compilation.

## Credits
ePBL was created by **Dr. Brandon G. Reichl** and **Dr. Robert Hallberg**
(NOAA GFDL) as the `MOM_energetic_PBL` module of MOM6; the legoESM scheme is a
JAX port of that code, modified for legoESM:
<https://github.com/mom-ocean/MOM6/blob/main/src/parameterizations/vertical/MOM_energetic_PBL.F90>.
The machine-learned diffusivity options (`eqdisc_*`) are due to
**Dr. Aakash Sane** (Sane et al. 2023, 2026).

## References
- Reichl, B. G., & Hallberg, R. (2018). A simplified energetics based
  planetary boundary layer (ePBL) approach for ocean climate simulations.
  *Ocean Modelling*, 132, 112-129. https://doi.org/10.1016/j.ocemod.2018.10.004
- Reichl, B. G., & Li, Q. (2019). A parameterization with a constrained
  potential energy conversion rate of vertical mixing due to Langmuir
  turbulence. *Journal of Physical Oceanography*, 49(11), 2935-2959.
  https://doi.org/10.1175/JPO-D-18-0258.1
- Sane, A., Reichl, B. G., Adcroft, A., & Zanna, L. (2023). Parameterizing
  vertical mixing coefficients in the ocean surface boundary layer using
  neural networks. *Journal of Advances in Modeling Earth Systems*, 15(10),
  e2023MS003890. https://doi.org/10.1029/2023MS003890
- Sane, A., Reichl, B. G., Adcroft, A., & Zanna, L. (2026). Machine learned
  equations for vertical mixing coefficients in the ocean surface boundary
  layer. *Geophysical Research Letters*, 53(10), e2026GL122106.
  https://doi.org/10.1029/2026GL122106
