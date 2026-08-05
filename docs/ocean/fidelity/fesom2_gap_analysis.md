# legoESM vs FESOM2 / FESOM2-JAX — configuration match and gap analysis

Two AWI papers motivate this:

* **arXiv:2606.11356** — *An Ocean Model Ported by a Large Language Model:
  Experience and Lessons from FESOM2 (Fortran to C to C++/Kokkos)*
  (Koldunov, Cheedela, Danilov, Sidorenko, Beyer, Jung).
* **arXiv:2608.01546** — *FESOM2-JAX v1.0: a differentiable shadow of the
  ocean–sea-ice model FESOM2, cast onto GPUs* (Koldunov, Danilov, Cheedela,
  Sidorenko, Beyer, Scholz, Kuznetsov, Streffing, A. Koldunov, Pantiukhin,
  Loza, Jung). Repo: `github.com/koldunovn/fesom_jax`; CORE2 data package
  Zenodo `10.5281/zenodo.21324319`.

FESOM2-JAX is the direct analogue of legoESM: the same language, the same
`lax.scan`-over-a-pure-step design, the same `shard_map` multi-GPU model, the
same end-to-end differentiability goal — reached by porting an existing
Fortran model kernel-by-kernel rather than by writing new physics. Its
validation target is therefore the natural yardstick for legoESM's ocean.

Everything below that is stated as measured was measured on Levante in this
worktree; everything inferred is labelled PLAUSIBLE.

---

## 1. The reference simulation, and where it lives on Levante

The hindcast both papers validate against is FESOM2 on the **CORE2 mesh**
forced by **JRA55-do**, cold-started from the **PHC3.0 winter climatology**.
All of its inputs are on this machine, and FESOM2's own namelists point at
them literally (`/work/ab0995/a270301/fesom2/config/` — a FESOM2 checkout in
the AWI project `ab0995`, which also holds Dmitry Sidorenko's account
(`a270029`), an author on both papers):

| Input | Path on Levante |
|---|---|
| Mesh | `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/` |
| Forcing | `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/{uas,vas,tas,huss,rsds,rlds,prra,prsn,psl,friver}.<year>.nc` |
| Initial condition | `/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc` |
| SSS restoring target | `.../JRA55-do-v1.4.0/PHC2_salx.nc` (monthly) |
| Chlorophyll | `/pool/data/AWICM/FESOM2/FORCING/Sweeney/Sweeney_2005.nc` |

Mesh, read from the files: **126 858 surface vertices, 244 659 triangles, 47
z-levels** (interfaces 0, 5, 10, 20, 30 … 5650, 6000, 6250 m — `core2_zaxis.txt`).
Paper 2's Table 2 quotes the same counts and Δt = 1800 s; the namelist on disk
uses `step_per_day = 36` (Δt = 2400 s).

Configuration below is read from those namelists, **not** from the papers'
prose. Be careful with the distinction: that checkout is a FESOM2 CORE2
configuration pointing at the same inputs, but it is **not verified to be the
papers' exact run**, and it differs from paper 2 in at least two places
flagged in the table (`step_per_day = 36` vs the paper's Δt = 1800 s;
`whichEVP = 0` = standard EVP vs the paper's mEVP). Where they disagree the
paper is the authority for what was published and the namelist is the
authority for what a FESOM2 CORE2 run on this machine actually does.

| Component | FESOM2 CORE2 setting |
|---|---|
| Vertical coordinate | `which_ALE = 'zstar'` |
| Free surface | semi-implicit, preconditioned CG, warm-started |
| Time stepping | Adams–Bashforth 2 (`AB_order = 2`), implicit vertical diffusion |
| Tracer advection | `nml_tracer_list = 'MFCT','QR4C','FCT'` = (horizontal, vertical, limiter). **MFCT**, not MUSCL — `oce_adv_tra_driver.F90:345` dispatches them to different routines, and the paper's prose says "MUSCL–FCT" while the namelist on disk selects MFCT. Both `adv_tra_hor_mfct` and `adv_tra_ver_qr4c` take `num_ord` = "fraction of fourth-order contribution", set by `tra_adv_pv = 1.0` (full 4th order) |
| Momentum viscosity | biharmonic, flow-aware |
| Vertical mixing | `mix_scheme = 'KPP'`, `Ricr = 0.3`, `concv = 1.6`, `A_ver = 1e-4` (paper 2's hindcast used cvmix-TKE, `ck = 0.1`, `c_eps = 0.7`) |
| Eddy parameterisation | GM after Ferrari et al. (2010) (`Fer_GM = .true.`, streamfunction BVP) + Redi; `K_GM_max = 1000`, `Redi_Kmin = 100`; ODM95 slope tapering (`S_cr = 2e-3`); resolution scaling; `exp(-z/500 m)` vertical downscaling |
| Bottom drag | quadratic, `C_d = 0.0025` |
| Bulk formulae | NCAR / Large & Yeager, **all three reference heights 10 m** (`ncar_bulk_z_wind = z_tair = z_shum = 10.0`) |
| Sea-ice dynamics | paper 2: mEVP, `alpha = beta = 250`, 120 iterations. **The namelist on disk sets `whichEVP = 0` = STANDARD EVP** (`MOD_ICE.F90:213`: `0=standart; 1=mEVP; 2=aEVP`); its `alpha_evp`/`beta_evp = 250` are inert at that setting. `evp_rheol_steps = 120`, `P* = 30000`, `e = 2`, `C = 20`, `Delta_min = 1e-11` |
| Sea-ice thermo | zero-layer, 7 thickness classes, `S_ice = 4`, albedos 0.81/0.77/0.70, emissivity 0.97 |
| IC | PHC3.0 winter, `t_insitu = .true.` (in-situ → potential temperature at initialisation) |

**No CORE2-mesh FESOM2 output is on this machine.** The only FESOM output
found (`/work/ab0995/a270301/monthly_martinas_run/fesom/`) is on the **DARS**
mesh (`ncells = 3 160 340`), a different configuration. So the FESOM-side
numbers quoted for comparison are the papers' **published** ones, not
locally recomputed:

* SST RMS between the JAX and Fortran runs **0.004 °C**; SSS **0.002** —
  against 0.61 °C / 0.38 versus observations.
* Volume-mean temperature drifts 3.636 → 3.619 °C over six decades
  (−1.7×10⁻² °C); the two codes differ by +8.8×10⁻⁵ °C.
* Seasonal ice-area extremes agree to ≤0.07×10¹² m².

That sets the bar for what "the same model" means in these papers: differences
**two orders of magnitude below** the model–observation distance.

---

## 2. What was built here to match the forcing and boundary conditions

Three pieces, all reusing existing legoESM machinery rather than adding a
parallel ingest path:

1. `scripts/data/stage_jra55_do_fesom_pool.py` — merges DKRZ's per-variable
   JRA55-do NetCDFs (the very files FESOM2 reads) into the single store
   `build_jra55_cache` requires. It resolves three real mismatches, each
   documented in the module:
   * the **flux variables are stamped at interval midpoints** (01:30, 04:30 …)
     while the state variables are stamped at interval starts (00:00, 03:00 …).
     `_build_noleap_record_index` maps a record to a slot by `round(hour/3)`,
     which aliases midpoint stamps onto colliding slots, so the flux axis is
     rebased by −1.5 h. This is the same reading FESOM2 uses
     (`nm_nc_tmid = 0`).
   * `friver` is **daily and on the 0.25° river grid**, not 3-hourly on the
     0.5625° atmospheric grid. It is conservatively regridded via
     `legoesm.grids.conservative_regrid` and zero-order-held to 3-hourly.
     The two grids have different longitude origins, so the source is
     wrap-padded before the overlap weights are computed; without that the
     destination's seam column came back half weighted (caught by
     `require_full_coverage=True`, not by inspection). The daily hold is
     binned against the interval **starts**, not the 12:00 stamps: binning on
     the stamps ran every day noon-to-noon, putting each runoff transition
     12 h late and giving the first day 12 three-hourly slots and the last
     only 4. Timing error, not a budget one — the annual mean moved by
     1.5×10⁻⁵ relative.
   * cell edges come from the files' **CF `lat_bnds`/`lon_bnds`**, not from
     `cell_edges_1d`'s uniform-spacing inference — the JRA55-do atmospheric
     latitudes are Gaussian (spacing varies 0.5569–0.5616° over the 320
     rows). Runoff conservation across the staging regrid: global integral
     9.35175889×10⁸ → 9.35175889×10⁸ kg/s, **−1.9×10⁻¹⁴ relative**. With the
     uniform-edge inference instead it was 3.8×10⁻³. Conserving the total is
     necessary but **not sufficient** — most of it lands where the ocean
     model will throw it away; see §3.5.
     (`build_jra55_cache` still makes the uniform assumption for its own
     regrid — a pre-existing approximation, unchanged here.)
2. `scripts/data/build_phc3_ic_for_omip.py` — PHC3.0 winter → the WOA18 layout
   `init_ocean_from_woa` consumes, converting **in-situ → potential
   temperature** and re-levelling onto `WOA_DEPTHS`.
3. `scripts/cluster/fesom2_comparison/stage_jra55_1958.sbatch` — runs both
   stages under SLURM with `JAX_ENABLE_X64=1`.

---

## 3. Gaps found

Ordered by how much they move numbers, not by how hard they are to fix.

### 3.1 CONFIRMED (fixed) — JRA55-do reference height was declared wrong

`run_omip.py:1670-1675` **previously built**
`CouplerConfig(z_ref=10.0, z_t_atm=2.0, z_q_atm=2.0)` for the
`jra55_do_tropical` lane (now 10.0/10.0 — see the fix note below). The JRA55-do v1.4.0 files carry an explicit
`height = 10.0 m` coordinate on `tas`, `huss` **and** `uas`, and FESOM2's
`namelist.forcing` sets all three heights to 10 m for exactly this dataset.
(The CF `comment` on `tas` says "usually, 2 meter" — that is boilerplate from
the CMOR table, contradicted by the file's own `height` coordinate.)

Measured with `scripts/tmp/_probe_jra55_reference_height.py` — identical
winds, T, q, SLP, SST, roughness, scheme, iteration count and stability
functions on both sides, the ONLY change being `z_t`/`z_q`; 8 three-hourly
records of 1 Jan 1958, area-weighted over PHC3 ocean points:

| Flux | at 10 m (correct) | at 2 m (as coded) | difference |
|---|---|---|---|
| Sensible | 27.48 W/m² | 29.61 W/m² | **+2.13 W/m² (+7.8 %)** |
| Latent | 105.55 W/m² | 117.08 W/m² | **+11.53 W/m² (+10.9 %)** |
| Wind stress | 0.0914 Pa | 0.0917 Pa | +0.0003 Pa (+0.3 %) |

**Scope of this number, stated precisely:** it is a ONE-DAY difference at a
FIXED (PHC3 winter climatology) SST, not an annual mean and not a spun-up
model bias. What the probe establishes is that the mis-declared height moves
the turbulent fluxes by ~10 % *at these conditions*; whether the annual,
evolving-SST bias is larger or smaller is not measured here, because SST
feedback damps a latent-heat perturbation. Even so, a ~10 % turbulent-flux
error dwarfs the whole FESOM2-JAX-vs-Fortran discrepancy the papers report,
and the fix is one line.

**FIXED here**: `run_omip.py` now sets `z_t_atm = z_q_atm = 10.0`, and the
test that asserted `2.0` (`tests/unit/test_run_omip_jra55_dispatch.py:181`)
was updated — it had been pinning the defect as correct. The stale "set to
2.0 for OMIP" prose in `core/bulk_flux.py` and `coupler/config.py` was
corrected too. Corroboration from the repo's own invariants:
`validate_air_sea_consistency` *raises* when `CouplerConfig.z_t_atm` differs
from the atmosphere's `z_ref` (10 m) — the coupled path already assumes the
value this lane was contradicting.

The probe lives under `scripts/tmp/` and is **gitignored** (`_probe_*.py`), so
it will not survive a fresh checkout. The method above is complete enough to
rebuild it: call `compute_most_fluxes` twice on one JRA55-do record with
everything fixed except `z_t`/`z_q`, and area-weight the difference over ocean
points with Gaussian-latitude weights from the file's own `lat_bnds`.

### 3.2 CONFIRMED — no in-situ → potential temperature conversion (now added)

Every legoESM EOS and the prognostic tracer are potential temperature, but the
climatologies the IC path reads (WOA, PHC) archive **in-situ** temperature and
nothing converted them. FESOM2 does convert (`t_insitu = .true.` →
`gen_ic3d.F90::insitu2pot` → `ptheta`/`atg`).

Added `adiabatic_temperature_gradient` and `potential_temperature` to
`legoesm.ocean.eos` (Bryden 1973 / Fofonoff 1977 — the `ATG`/`THETA` pair of
Fofonoff & Millard 1983, the same reference as the module's existing UNESCO-80
EOS), pinned in `tests/ocean/unit/test_eos_potential_temperature.py` against
both published check values (`ATG = 3.255976e-4 °C/dbar`,
`THETA = 36.89073 °C`) **and** three constructions that are algorithmically
independent of the module's own — a monomial expansion of ATG (catches a
misplaced parenthesis), a 4000-step classical RK4 integration (shares none of
the four-stage Gill weights), and the `dθ/dp = −ATG` identity — plus a round
trip, a non-zero `p_ref` case, jit/vmap parity, float32, and gradient
finiteness. **32 pass.**

Worth recording, because it looks like a discrepancy and is not: the
converged integration gives `36.8906933` at the check state while the module
gives `36.8907265`. The published `36.89073` carries the one-step scheme's
own ~3.3×10⁻⁵ °C truncation error — the module is right and the *published
value* is the approximate one.

Measured effect on the PHC3 IC: **max cooling 0.645 °C at 4000 m**, zero at
the surface. Initialising without it starts the abyss that much too warm.

Two ordering details that are easy to get wrong and both change numbers:

* **Interpolate first, convert second.** The adiabatic correction is a
  nonlinear function of depth, so converting at the source levels and
  interpolating the result afterwards is a *different* operation from
  interpolating the in-situ field and converting at the target level. On the
  real PHC3 file the two orders differ by up to ~1 °C (RMS 0.11 °C) in the
  deepest levels. FESOM2 does the latter (`gen_ic3d.F90` interpolates, then
  `insitu2pot` evaluates `ptheta` at the model depth `abs(Z(nz))`), and so
  does this script.
* **Cap the conversion pressure at each column's deepest valid source
  level.** Below the data the re-levelling holds the deepest valid value, so
  an uncapped conversion would adiabatically correct a shelf column's warm
  bottom value as if it sat at 5500 m — a fabricated 1.4 °C cooling in cells
  that are below the seafloor anyway. FESOM2 has the same restriction:
  `insitu2pot` loops wet levels only (`do nz = nzmin, nzmax-1`).

Related, and NOT yet fixed: `scripts/data/build_woce_ic_latlon.py` documents
that it passes TEOS-10 **conservative** temperature as potential temperature
(~0.05–0.2 °C). Different conversion, same class of defect.

### 3.3 CONFIRMED (fixed) — `init_ocean_from_woa` ignored the file's depth axis

`init_woa.py:644` (`woa_depths = WOA_DEPTHS[:T_woa.shape[-1]]`) takes the
source depth axis as `WOA_DEPTHS[:n_depth]`
regardless of what the file says. Feed it PHC3's 33 levels (0 … 5500 m) and
its deep values are read as if they sat in the top few hundred metres — a
silently wrong ocean, not an error. The repo already works around this
(`build_woce_ic_latlon.py` re-levels onto `WOA_DEPTHS` first, and says why),
and the PHC3 script here does the same.

**FIXED at the root**: `load_woa18` now returns the file's own depth
coordinate (`depth`/`z`/`lev`/`level`/`deptht`, made positive, validated
strictly ascending and length-matched to the field), and
`init_ocean_from_woa` uses it, falling back to `WOA_DEPTHS` only when the file
genuinely has none. The regression test is shown to FAIL with the fix
disabled: a two-level 0 m / 3000 m source read against `WOA_DEPTHS` returns
0.000 °C at 1400 m where the correct answer is 10.667 °C.

### 3.4 CONFIRMED (fixed) — the OMIP-2 lane's sea ice had no rheology at all

`--jra55-sea-ice` does not select a sea-ice model; it hard-codes one.
`run_omip.py:1765-1767` builds `SeaIceConfig(stability_scheme=...)` and
nothing else, and that default is `dynamics = "none"`, `n_categories = 1`
(`ice/config.py:353,367`) — a thermodynamic slab with diagnostic free drift.
The comment there says so outright: *"Slab ice: dynamics='none', n_cat=1"*.
There is no CLI flag in `run_omip.py` to change it.

FESOM2 runs a full rheology (`whichEVP = 0`, standard EVP, 120 subcycles,
`P* = 30000`, `e = 2`). legoESM's `evp_stress_update` / `mevp_stress_update`
exist and are tested, but they are **unreachable from this driver**, so any
statement that legoESM "has mEVP" is a statement about the library, not about
this run. Ice drift, ridging, and the ice-ocean stress are therefore not
comparable to FESOM's, and neither is anything downstream of them (Arctic
freshwater, deep-water formation).

**FIXED**: `--ice-dynamics {none,free_drift,evp,mevp}`, `--ice-categories`,
and per-parameter overrides (`--ice-n-evp`, `--ice-p-star`, `--ice-e-yield`,
`--ice-c-strength`, `--ice-delta-min`, `--ice-alpha-mevp`,
`--ice-beta-mevp`). Selecting a rheology also switches the ice STATE to
`DynamicSeaIceState` — a `SeaIceState` has no velocity or stress fields for
the solver to advance — and threads the grid through to the tile, because the
strain rates need the metrics and `grid=None` would have given zero
deformation and a rheology that silently did nothing. Defaults reproduce the
old slab bit-for-bit. Rheology on a grid with no strain-rate branch (tripole)
raises rather than returning zero deformation.

FESOM's CORE2 values are therefore reachable directly:
`--ice-dynamics evp --ice-n-evp 120 --ice-p-star 30000 --ice-e-yield 2.0
--ice-c-strength 20.0 --ice-delta-min 1e-11`.

### 3.5 CONFIRMED (fixed) — river runoff reached the model unrouted

**RETRACTED, in full, before it was acted on:** the first version of this
section claimed the staged runoff total was a "full-cell proxy" ~3.4× too
large, on the reasoning that `friver` carries
`cell_measures: area: areacello` / `cell_methods: area: mean where sea` (it
does) and that an integral over full cell areas therefore overstates the
discharge. **That inference is wrong**, and a literature check kills it: the
full-cell integral is 9.35×10⁸ kg/s against an observed global river
discharge of ~1.2×10⁹ kg/s (≈37–40 ×10³ km³/yr), while the
"ocean-area-normalised" 2.72×10⁸ kg/s would be ~4× *below* observation. At
0.25° a river-mouth cell is essentially all ocean, so `areacello ≈ A_cell`
there and the full-cell integral is the physical discharge. The error was
using `sftof` — a **0.5625° atmospheric-grid** mask — as a stand-in for the
**0.25° river-grid** ocean area. Different grids, different masks.

What survives, and it is the part that matters:

| quantity (1 Jan 1958) | conservative regrid | nearest-cell, NO spreading |
|---|---|---|
| discharge-weighted mean `sftof` | 0.2905 | 0.2758 |
| share on cells with `sftof == 0` | **70.95 %** | **72.42 %** |

The two agree, which **refutes the obvious alternative explanation**: this is
not my 0.25°→0.5625° regrid smearing coastal discharge onto land. Mapping
each source cell straight onto the atmospheric cell containing its centre —
no spreading whatsoever — gives the same answer. JRA55-do genuinely places
discharge at 0.25° river-mouth cells whose 0.5625° host cell has zero ocean
fraction, which is exactly what you would expect when the river grid resolves
a coastline the atmospheric grid does not.

The discharge is also extremely concentrated: 14 541 of 1 036 800 source
cells are non-zero, and the **top 100 cells carry 68 %** of the global total.
So this is not a diffuse coastal haze that a mask will mostly catch — two
thirds of the world's river input rides on ~100 cells that each either land in
a wet cell or are lost entirely.

**The gap is routing, not normalisation.** legoESM's OMIP-2 lane passes
`friver` through and the ocean model masks dry cells, which *discards* that
share rather than relocating it. FESOM2 ships exactly the missing piece:
`use_runoff_mapper` / `runoff_radius = 500 km` in `namelist.forcing`.

**FIXED**, and the loss is now measured on the mask that actually matters.
`legoesm.ocean.forcing.runoff_mapper` builds the dry-to-wet routing plan once
on the host (a KD-tree over unit-sphere chords, so the dateline and the poles
need no special casing) and applies it on device as one scatter-add — jit-safe
and differentiable in the runoff field. Runoff is a flux DENSITY, so a donor's
value is scaled by `A_src / A_dst` on the way: `sum(F·A)` is conserved to
1e-12 in the unit tests, on a grid whose cell areas vary 100-fold.

Exposed as `--runoff-routing {none,nearest,spread}` and `--runoff-radius-km`
(default 500, FESOM2's `runoff_radius`). `none` is the default so existing
runs stay bit-identical.

Measured with **legoESM's own 1° ETOPO land mask** and the **time mean over
all 2920 records of the 1958 cache**
(`scripts/tmp/_probe_runoff_loss_model_grid.py`) — the model-grid number the
`sftof` proxy above could not give. Annual-mean global discharge in the cache
is 1.354×10⁹ kg/s — **~15 % above** the 1.18×10⁹ kg/s (1.18 Sv) observational
estimate of Dai & Trenberth (2002), which is the right order and further
confirmation that the retraction above is correct, but is NOT "agreement" and
should not be quoted as such. (The 1 Jan value, 9.35×10⁸, is a
northern-winter minimum, not the annual figure — which is why a single
January record is the wrong reference for the routing budget.)

| | delivered to the ocean | unrouted |
|---|---|---|
| no routing (the old behaviour) | **29.96 %** | — (70.04 % discarded) |
| `nearest` @ 500 km | **98.64 %** | 1.36 % |
| `nearest` @ 1000 km | **99.87 %** | 0.13 % |

`spread` delivers the same totals (it must — both conserve) over many more
recipient cells: 404 256 donor-recipient pairs at 500 km versus 9 759 for
`nearest`, which is the point of it, since a large river landing in one 1°
cell otherwise digs a salinity crater.

Scope of the mask: this is `init_ocean_bathymetry`'s output under the run's
own `BathymetryConfig`. The driver then applies equatorial smoothing and
optional passage widening, which adjust depths and can open or close a
handful of cells, so the percentages are the mask at that stage rather than
cell-exact for the integration.

The residual 1.36 % at 500 km is discharge with no wet cell in range at all;
the builder reports it as `unrouted_fraction` and the driver prints the whole
budget at setup — computed from the cache's TIME MEAN, not one record, because
the loss is seasonal. Raise the radius if that matters for a given basin.

Scope, stated so it is not over-read. First, **the mask is a proxy**: the only
`sftof` in the pool is **v1.6.0**, applied to **v1.4.0** `friver`. Their grids
and CF bounds are identical, but nothing here proves the two versions share a
land–sea definition, so the exact percentages are proxy numbers — the
qualitative conclusion (discharge sits at river-mouth cells a coarser ocean
mask calls land) does not depend on the version, the percentages do. Second,
`sftof` is JRA55-do's own atmospheric
mask, not legoESM's ETOPO-derived 1° land mask, so the *fraction lost at the
model's own grid* is not this number — a 1° cell swallows more coastline than
a 0.5625° one, so the true loss is smaller. The measurement establishes that
routing is required, not how much is lost.

### 3.7 CONFIRMED (open) — the matched cold start dies on step 1

The matched run **does not integrate**. It fails on the FIRST timestep, and
the cause is the initialisation, not the forcing.

`--woa-init` installs the full 3-D PHC3 density field while keeping **zero
velocity and zero eta** (`run_omip.py`, "Keep zero velocity, zero eta — let
the model adjust"). The entire baroclinic pressure gradient is therefore
unopposed on step 1.

Measured offline from the IC (1° lat-lon, 47 CORE2 levels, dt = 2400 s), at
the cell where the model's own `j_maxu`/`i_maxu` diagnostic puts the maximum
(lat 81.5 N, lon 96):

| PGF component | implied `du` in one 2400 s step |
|---|---|
| full | 41.87 m/s |
| barotropic (depth-mean) | 22.30 m/s |
| **baroclinic (deviation)** | **22.21 m/s** |

The model reports `max_speed = 21.95 m/s` at step 1, from `max_speed = 0.0`
at step 0. That matches the **baroclinic-only** prediction to 1.2 %, which
says the barotropic solver (`implicit_cn`) *is* absorbing the depth-mean part
correctly — what survives is the unbalanced baroclinic shear, and that alone
is supercritical (dx = 16 km at 81.5 N gives CFL = 3.2).

Everything downstream is advective wreckage, not a separate defect: `T`
reaches 149 °C and `S` reaches −204 PSU, concentrated in coastal (33 % of
coastal columns vs 0.67 % interior) and shallow columns (median 1180 m vs
3828 m global).

**Refuted**, each by a one-variable arm at `--diag-every 1` or an offline
measurement — do not re-chase these:

* explicit vertical mixing — implicit buys exactly one step (1 → 2);
* the conservation fixer — disabled, still step 1;
* runoff routing — amplification 1.0×, `dS`/step 0.17 PSU;
* thin bottom cells — no column below 1 m (3.267 m vs 3.244 m clean);
* barotropic CFL — the solver is already `implicit_cn`;
* polar-meridian CFL — the Fourier polar filter is already on;
* corrupt IC salinity — the 1e-4 PSU cells are the real Amazon plume, and
  FESOM2 reads the same file.

The existing mitigation does not apply: `--T-ramp-days` ramps the **wind
stress** only (`tau × min(1, t/T_ramp)`), not the internal PGF.

Two things follow. First, no timestep rescues an unbalanced start here —
killing a 22 m/s kick needs dt ≈ 30 s. A stable matched cold start requires
initialising `u`/`v` in thermal-wind balance with the density field (the
repo has `coriolis_f_safe` for the equatorial `f → 0` floor, but its
thermal-wind helpers assume a linear EOS and are not drop-in), or abandoning
the matched IC for rest + restoring — which changes the experiment.

Second, **why FESOM2 survives the same `phc3.0_winter.nc` cold start is NOT
established.** No measurement here supports any explanation, and none should
be asserted until one exists.

Method note: the failure step is only resolvable at `--diag-every 1`.
`run_omip` sets `block_size = max(1, diag_every)` and runs the blowup check
only at block boundaries, so five earlier arms at `--diag-every 36` all
reported "step 36" — the first check, not the first failure. Arms run at
coarse `DIAG` prove only "this setting alone does not fix it".

Also found while diagnosing: `--north-cap-lat` defaults to **90.0** while its
help text says "(default 80)", so a global lat-lon ocean runs with an **open
North Pole**; the FESOM-match sbatch never passed the flag at all. Both are
now explicit via its `NCAP` knob.

### 3.6 Numerics differences that are real but second-order

| Item | FESOM2 | legoESM | Assessment |
|---|---|---|---|
| Tracer advection | MFCT horizontal + QR4C vertical, both at `num_ord = tra_adv_pv = 1.0` (full 4th-order contribution), FCT-limited | `fct_tracer_advection` with `high_order ∈ {ppm, centred2}` (NEMO `traadv_fct`) | Both are FCT-limited high-order, so the limiter family matches. The high-order flux differs (PPM / centred-2 vs 4th-order MFCT / QR4C). FESOM also *offers* a PPM vertical option (`adv_tra_vert_ppm`), just not in this namelist. PLAUSIBLE: matters for thermocline sharpness on a 47-level grid, not for the mean state — not measured here. |
| GM closure | Ferrari et al. (2010) streamfunction **BVP**: `oce_fer_gm.F90::fer_solve_Gamma` assembles and Thomas-solves a tridiagonal system per column for `fer_gamma` from `bvfreq`, `fer_c`, `fer_K` — verified in the source, not inferred from `Fer_GM = .true.` | **diagnostic** skew-flux `ψ = κ_GM·S` with DM95 tapering and Visbeck-adaptive κ; no per-column solve anywhere in `physics/lateral_mixing/` | **Largest structural physics gap.** The BVP form makes ψ a *solution* that satisfies the surface/bottom boundary conditions; the diagnostic form leans on slope tapering to keep ψ well behaved in the mixed layer instead. NOTE the two "Ferrari" schemes are different papers and must not be conflated: legoESM's `surface_complement` (default `True`, `GMRediConfig`) is Ferrari **et al. (2008)** — mixed-layer horizontal diffusion — and IS present; it is Ferrari **et al. (2010)** that is missing. |
| Vertical mixing | KPP (namelist) / cvmix-TKE with `ck = 0.1`, `c_eps = 0.7` (paper-2 hindcast) | `{none, constant, richardson, tke, catke, kpp}`; `tke` is Gaspar (1990) / Burchard (2002) — the same closure family CVMix-TKE implements — with `c_k` and `c_eps` as `__param_spec__` tunables whose bounds (0.033–0.3, 0.231–2.1) contain the paper's values | Both of FESOM's choices have a counterpart, and paper 2's coefficients are settable without code changes. Not verified line-by-line against CVMix. |
| Free surface | semi-implicit, preconditioned CG | `solve_helmholtz_freesurface`, PCG with a `custom_vjp` adjoint | Structurally equivalent. |
| Time stepping | AB2 (ε = 0.1) | AB2 outer integrator (`VALID_AB2_SCOPE`) | Equivalent. |
| Vertical coordinate | ALE z* | `OceanZStarCoordinate` (+ partial cells) | Equivalent. |
| Sea-ice dynamics | standard EVP (`whichEVP = 0`), 120 subcycles; paper 2 says mEVP α = β = 250 | `evp_stress_update` / `mevp_stress_update` exist in `ice/rheology.py`, but the OMIP-2 driver hard-codes `dynamics = "none"` | **See §3.4 — this run has NO ice rheology.** |
| Sea-ice thermo | zero-layer conduction; `iclasses = 7` | zero-layer conduction `F_cond = k(T_f − T_i)/(h + h_min)`; prognostic ITD, CICE bounds for 1/3/5/7 categories (default 1) | Conduction equivalent in kind. **The "7 classes" are NOT the same object** and must not be matched by setting `n_categories=7`: FESOM's `iclasses` is a *diagnostic sub-grid* thickness distribution used only to average the conductive flux over `thact = (2k−1)·h/iclasses` within ONE prognostic thickness (`ice_thermo_oce.F90:492-507`), whereas legoESM's ITD carries a separate prognostic state per category with remapping between them. legoESM's default `n_categories = 1` is the closer analogue of FESOM's prognostic state; the sub-grid flux averaging has no legoESM counterpart. |
| Shortwave penetration | Sweeney (2005) monthly chlorophyll climatology | Jerlov water types **and** Morel–Maritorena RGB with chlorophyll | legoESM is the more capable of the two here; the *input* differs. |
| Runoff source | ambiguous in the checkout: the active `namelist.forcing` sets `runoff_data_source = 'JRA55'` with `nm_runoff_file = '<add path>'` (**unset**), while `namelist.forcing.JRA` sets `'CORE2'` + `CORE2_runoff.nc` — a single 180×360 `Foxx_o_roff` climatology. FESOM also ships `use_runoff_mapper` / `runoff_radius = 500 km` for coastal spreading | JRA55-do `friver`, the OMIP-2 protocol runoff, with no mapper | **A DELIBERATE, NON-MATCHING CHOICE, not a match.** Global totals differ by ~38 %: `CORE2_runoff.nc` 1.290×10⁹ kg/s, `friver` 0.935×10⁹ kg/s. Matching FESOM needs the production run's actual runoff namelist, which is not in the checkout. Compounded by §3.5 — neither source is routed to wet cells. |
| SLP | `l_mslp = .false.` — FESOM does **not** use it | legoESM's cache carries `psl` and uses it for moist-air density and saturation humidity | Small, but it is a genuine forcing-protocol difference. |
| Bottom drag | quadratic, `C_d = 0.0025` | the bathy lat-lon config already sets `bottom_drag_r = 2.5e-3` (`run_omip.py:1161`) | Matched by default. **Trap:** `--bottom-drag-cd0` (default `1.0e-3`) looks like the knob but is read only when `--bottom-drag-scheme != legacy` (`_apply_drag_iwm_overrides`, `run_omip.py:241`), so passing it on the default lane is a silent no-op. |

---

## 4. The grid question: what unstructured support would be needed

**legoESM cannot run FESOM's mesh today, and the obstacle is the
discretisation, not the mesh format.**

FESOM2 is a **cell-vertex finite-volume** scheme on an arbitrary triangulation:
scalars (T, S, SSH) live at mesh **vertices**, on the median-dual control
volumes; the **full horizontal velocity vector** lives at **triangle
centroids**. Read from the source, not from the papers —
`oce_setup_step.F90:645-648` allocates
`dynamics%uv(2, nl-1, elem_size)` (a two-component vector per ELEMENT) beside
a diagnostic `dynamics%uvnode(2, nl-1, node_size)`.

legoESM's only unstructured lane is MPAS-style **TRiSK** on a spherical
centroidal Voronoi tessellation (`ocean_pe_mpas.py`, Ringler et al. 2010):
scalars at Voronoi **cell centres**, a single **normal velocity component** at
cell **edges**, vorticity at the dual triangle vertices.

These are duals of one another in *connectivity* and completely different in
*prognostic variable placement*. `VoronoiMesh` already carries the arrays a
triangular scheme needs — `cellsOnVertex`, `verticesOnCell`, `areaTriangle`,
`kiteAreasOnVertex`, `dvEdge`, `dcEdge` — so the mesh container is not the
problem. What is missing, in dependency order:

1. **A FESOM mesh reader.** `load_mpas_mesh` reads an MPAS NetCDF; FESOM ships
   ASCII (`nod2d.out`, `elem2d.out`, `aux3d.out`, `nlvls.out`, `elvls.out`,
   `edges.out`, `edge_tri.out`). Straightforward, and it is the only piece
   that is purely mechanical.
2. **Arbitrary (non-centroidal) triangulations.** `create_voronoi_mesh` builds
   an SCVT from an icosahedron plus Lloyd relaxation; FESOM meshes are
   externally generated, variable-resolution Delaunay triangulations that are
   *not* centroidal. `_build_mesh_from_generators` does accept arbitrary
   generator points, so a Voronoi mesh dual to FESOM's node set is reachable —
   but it would be a *different* mesh from FESOM's, with different control
   volumes, so it cannot be used for a term-by-term comparison. PLAUSIBLE:
   it is good enough for an "equivalent-resolution" comparison and bad for a
   fidelity oracle.
3. **A cell-vertex FV operator set.** Gradient/divergence on median-dual
   control volumes, the P1 basis on triangles, the vertex↔centroid transfers,
   the Coriolis treatment for a centroid-collocated vector, and the FESOM
   pressure-gradient reconstruction. This is the real work — a second
   unstructured dycore, not an adaptation of the TRiSK one, because TRiSK's
   entire skeleton (edge-normal velocity, tangential reconstruction from kite
   areas, potential-vorticity flux) has no counterpart in a centroid-collocated
   vector scheme.
4. **Partitioning and halo exchange for a vertex/centroid pair.** legoESM's
   MPI halo machinery is edge/cell-indexed. FESOM ships METIS partitions
   (`dist_<N>/`) that could be reused, but the halo definition differs.

A cheaper intermediate that answers most scientific questions: keep legoESM on
its own grid and add **FESOM output regridding** — `conservative_regrid_unstructured`
already does MPAS↔MPAS, and the same overlap machinery extends to a
triangulation. That makes FESOM a *diagnostic* oracle without making it a
*numerical* one.

**Recommendation.** Do not build a FESOM cell-vertex dycore to chase this
paper. The two papers' contribution is a *porting methodology* and a
*differentiable FESOM*, not a new discretisation; legoESM already has a
differentiable ocean and an unstructured lane. The high-value items were the
forcing-protocol fixes (§3.1–3.3), the runoff routing (§3.5) and the missing
sea-ice rheology (§3.4) — **all now done**. What remains, in order:

1. **A regridding-based comparison.** Extend
   `conservative_regrid_unstructured` (already MPAS↔MPAS) to a triangulation
   so FESOM output can be scored against legoESM output on a common grid.
   This is the cheapest thing that turns "we ingest the same forcing" into
   "here is the skill difference", and it unblocks everything below.
2. **The Ferrari (2010) GM streamfunction BVP** (§3.6). Deliberately NOT
   implemented here, and the reason is not effort: it is a new closure, not a
   defect fix, and there is no FESOM CORE2 reference on this machine to
   validate it against. Shipping an unvalidated eddy closure would trade a
   documented difference for an undocumented one. Do item 1 first, then this
   becomes verifiable.
3. **A FESOM cell-vertex dycore** — see §4. Still not recommended.

---

## 5. Status of the matched run

* Forcing cache built from the FESOM2 pool files: see §2.
* IC built from the same PHC3.0 winter file, with the temperature conversion.
* The run uses `run_omip.py --grid latlon --forcing-mode jra55_do_tropical`,
  which is legoESM's OMIP-2 lane.

**This is not a replication of the papers' 62-year hindcast**, and a
mean-state comparison against their numbers would be a confound: different
grid (1° regular lat-lon vs a 126 858-vertex triangulation), different
bathymetry, different eddy closure, no ice rheology (§3.4), runoff that is
that arrives unrouted (§3.5), a different runoff source
(§3.6), different run length.

**The forcing is NOT byte-identical to what FESOM2 sees, and should not be
described that way.** It comes from the same source files, but five
transformations change delivered values:

1. flux time stamps rebased by −1.5 h onto the state axis (§2.1) — an
   *implementation choice*, forced by the cache builder's slot mapping, not
   by the grid;
2. conservative regrid from the Gaussian 0.5625° grid to 1° regular lat-lon,
   plus the separate river-grid regrid and the daily→3-hourly hold (§2.2–2.3)
   — unavoidable on a different grid;
3. float32 storage in the staged cache — an *implementation choice*, and the
   cheapest of the five to remove;
4. runoff from `friver` rather than whatever the production FESOM run used
   (§3.6) — a ~38 % difference in the global total; an *implementation
   choice* only in the weak sense that the alternative is knowable, but the
   production run's runoff namelist is not in the checkout, so it is
   currently **not recoverable**;
5. runoff routing (§3.5) — was a *missing capability* and is now implemented;
   with `--runoff-routing nearest` at 500 km, 98.6 % of the discharge reaches
   the ocean instead of 30.0 %. The residual 1.4 % has no wet cell in range
   and is printed at setup, not hidden.

What the run establishes is that legoESM ingests **the same source data** —
the thing that surfaced §3.1.

It does **not** establish that legoESM integrates stably under it. **It does
not: the matched configuration fails on step 1** (§3.7), from the
initialisation rather than the forcing. Until that is resolved there is no
integration to compare, controlled or otherwise. An earlier revision of this
section claimed stable integration; that claim was wrong and is retracted.
