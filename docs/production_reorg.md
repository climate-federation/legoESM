# Production folder reorg — runbook

**Status:** planned, not started. **Sequencing: AFTER the production ship.**
Reorg is pure churn with real regression risk (749 import sites) and zero
functional payoff — do not stack it on the ship. Execute on a gated branch,
one PR per package, full test suite + import-lint gate per PR.

Decisions locked (2026-07-13):

1. **Keep the federation namespace.** `packages/*/legoesm/<subpkg>/` is the
   uv-workspace + PEP420 namespace carve (8 installable `legoesm-*` wheels
   sharing one `import legoesm.*` namespace; `members=["packages/*"]`, no
   `__init__.py` at the `legoesm/` level). It is **not** redundant. The reorg
   happens *inside* it. **Rule 0: never touch the `packages/<m>/legoesm/`
   layer.**
2. **dynamics/ → `gcm/les/crm/shared/`**, physical `git mv` + codemod all
   import sites to deep paths.
3. **Forcing: atmosphere only.** `ocean/forcing/` (OMIP) and `land/forcing/`
   (LMIP) are already component-owned and correct — leave them.

---

## Part A — `atmosphere/dynamics/` (55 files → 4 buckets)

Target: `packages/atmosphere/legoesm/atmosphere/dynamics/{gcm,les,crm,shared,neural}/`

> ⚠️ `global` is a Python keyword — **cannot** be a module name
> (`import ...dynamics.global` is a SyntaxError). Use **`gcm`**.

### gcm/ — global dycores (lat-lon / cubed-sphere / spectral / MPAS)
```
compressible_euler.py              compressible_euler_cdgrid.py
compressible_euler_latlon_cgrid.py compressible_euler_mpas.py
primitive_eq_cdgrid.py             primitive_eq_latlon_cgrid.py
primitive_eq_mpas.py               semi_implicit_cdgrid.py
shallow_water_fv3_cdgrid.py        shallow_water_latlon_cgrid.py
shallow_water_mpas.py              shallow_water_nesting.py
spectral_pe.py                     spectral_sw.py
spectral_nh.py                     sharded_atm_latlon_step.py
tiled_step_adapter.py              tracer_transport_latlon.py
tracer_transport_mpas.py           _fv3_lin_pgf.py
dcmip2025_ic.py
```

### les/ — large-eddy (plane / pseudo-incompressible)
```
spectral_les_plane.py    spectral_les_moist.py    spectral_plane.py
column_les.py            column_les_diagnosis.py
les_closure_diagnosis.py les_regime.py            les_vertical_mapping.py
tke_sgs_plane.py
pseudo_incompressible_plane.py       pseudo_incompressible_plane_mpi.py
pseudo_incompressible_poisson.py     pseudo_incompressible_poisson_mpi.py
compressible_euler_plane.py          compressible_euler_plane_halo.py
plane_fd_advection.py   plane_operators.py        plane_operators_halo.py
```

### crm/ — cloud-resolving (RCE / SAM)
```
rce_diagnostics.py  rce_mpi.py  rce_surface_flux.py
sam_case_setup.py   moist_mass_fixer.py
```

### shared/ — grid-agnostic numerics
```
tracer_transport.py  flux_form_tracer_transport.py  tracer_positivity.py
cfl_diagnostic.py    mean_wind_filter.py
```

### neural/ — learned dycores (surrogates)
```
sfno_pe.py  sfno_sw.py  ucast_pe.py
```

### Rulings on ambiguous files
- **plane substrate** (`plane_operators*`, `plane_fd_advection`,
  `compressible_euler_plane*`): consumed by both LES and CRM. Assigned to
  **les/** (primary consumer); CRM imports from there. If cross-import gets
  ugly during execution, promote to `shared/plane/` — one small extra bucket,
  decide then, not now.
- **`moist_mass_fixer`**: docstring "plane NH CRM" → **crm/**.
- **`sfno/ucast`**: learned dycores → **neural/** (stay in dynamics; do NOT
  move to `packages/ml` — that's a cross-package move + reopens the namespace
  question for no gain).
- **forcing files currently in dynamics/** (`plane_large_scale_forcing.py`,
  `column_large_scale_extract.py`): these are forcing, not dynamics → move to
  `atmosphere/forcing/` in Part B, not into a dynamics bucket.

### Codemod (deterministic — module names are unique tokens)
1. `git mv <file> <bucket>/` for each of the 55.
2. Add `__init__.py` to each new bucket dir (`gcm` etc. are **regular**
   subpackages — `legoesm/atmosphere/` has `__init__.py`, so this level is not
   PEP420).
3. Rewrite every import site from a `module → bucket` table:
   `dynamics.spectral_les_plane` → `dynamics.les.spectral_les_plane`, etc.
   Names are unique, so a token-anchored `sed`/`libcst` pass over all `*.py`
   is safe. ~749 sites, fully mechanical.
4. Fix intra-dynamics relative imports between moved files.

### Gate (must pass before merge)
- `grep -rn 'dynamics\.\(spectral_les_plane\|compressible_euler\|...\)' --include=*.py`
  returns **zero** old flat paths (assert no straggler).
- `python -c "import legoesm.atmosphere.dynamics.{gcm,les,crm,shared,neural}"` smoke import.
- Full `pytest` suite green.

---

## Part B — atmosphere forcing (scattered → `atmosphere/forcing/`)

Only atmosphere is messy. Ocean/land forcing stay put.

Target: `packages/atmosphere/legoesm/atmosphere/forcing/{idealized,scm}/`

| from | file | → |
|---|---|---|
| `atmosphere/` (top) | `kessler_forcing.py`, `large_scale_forcing.py`, `column_forcing.py`, `held_suarez.py` | `forcing/idealized/` |
| `atmosphere/` (top) | `scm.py`, `scm_forcing.py`, `dephy_scm.py` | `forcing/scm/` |
| `atmosphere/` (top) | `sam_case_forcing.py` | `forcing/` (pairs with `dynamics/crm/sam_case_setup`) |
| `dynamics/` | `plane_large_scale_forcing.py`, `column_large_scale_extract.py` | `forcing/` |

Leave in place (already correct):
- `ocean/forcing/` = **OMIP** (core2, jra55_do, woa, sss_restoring, …)
- `land/forcing/` = **LMIP** (cru_jra, solar)
- `tools/forcing/` = **AMIP** realistic hub (amip, amip_config, jra55_do,
  external). Optional: move to `atmosphere/forcing/amip/` for symmetry — but
  it's cross-cutting (drives coupler too), so `tools/` is defensible. Deferred.

Note: `atmosphere/idealized/` already holds idealized **initial conditions**
(rce, radiative_equilibrium, rcemip_ic, small_planet, topography). Keep the
IC-vs-forcing distinction: ICs stay in `idealized/`, forcing goes to
`forcing/idealized/`.

Same codemod + gate discipline as Part A (forcing has ~97 import sites).

---

## Part C — loose data at repo root
`forcing_amip/*.nc` (10 boundary-data files: sst_sic, ghg, ozone, solar,
volcanic, aerosol) sit loose at repo root. Move under a data path
(e.g. `data/forcing_amip/` or `configs/forcing_amip/`) and update the loader
path constant. Data-only, low risk, do last.

---

## PR breakdown (one package = one PR, gated)
1. `dynamics/` reorg + codemod (biggest; the loud mess).
2. atmosphere `forcing/` consolidation + pull forcing out of dynamics.
3. root `forcing_amip/` data relocation.

Each PR: mechanical move only, no logic changes, full suite + import-lint
green, codex adversarial review before merge (repo standard).
