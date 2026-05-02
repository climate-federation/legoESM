# Turbulence inventory

Scope: `src/legoesm/atmosphere/physics/turbulence/`. 9 schemes + integration. ~3000 LOC total.

## Schemes

| Scheme | File | LOC | Carry | Type | Inputs |
|--------|------|-----|-------|------|--------|
| smagorinsky | smagorinsky.py | 146 | none | constant Km | u,v,T,q_v,p,z,T_sfc,q_sfc,ρ |
| louis | louis.py | 190 | none | local Ri-based | + |
| holtslag_boville | holtslag_boville.py | 212 | none | nonlocal K-profile | + |
| ysu | ysu.py | 213 | none | YSU-style | + |
| tke | tke.py | 200 | tke | prognostic Mellor-Yamada 2.5 | + tke |
| clubb_lite | clubb_lite.py | 295 | tke | unified 2nd-order | + tke |
| edmf | edmf.py | 304 | tke | eddy-diffusivity + mass flux | + tke |

## Shared

* `surface_layer.py` (103 LOC): `compute_surface_fluxes` — bulk aero, supports `coare3`, `large_yeager`, or constant Cd/Ch.
* `vertical_diffusion.py` (144 LOC): `implicit_vertical_diffusion` — Thomas tridiagonal via `jax.lax.scan`.
* `pbl_height.py` (237 LOC): `diagnose_pbl_height` (sigmoid-weighted) and `diagnose_pbl_height_interp` (interpolated).

## Output (`TurbulenceOutput`)

* `du_dt`, `dv_dt` [m/s²], `dT_dt` [K/s], `dq_v_dt` [kg/kg/s] — all (ncol, nlev).
* `Km`, `Kh` [m²/s] — eddy diffusivities at full levels (diagnostic).
* `shflx` [W/m² ↑], `lhflx` [W/m² ↑], `ustar` [m/s], `h_pbl` [m].

## Convention

* `theta_v = T * (p_ref/p)^kappa * (1 + 0.61 * q_v)` — virtual potential temp.
* Surface: index `[:, -1]`.
* Stress sign: `tau_x = -ρ Cd |V| u` (negative for u>0, friction).
* `surface_flux` argument to `implicit_vertical_diffusion`: positive upward.
