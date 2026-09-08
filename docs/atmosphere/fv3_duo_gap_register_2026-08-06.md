# FV3 duo-grid gap register — 2026-08-06

Answers one question: **do we still have gaps for a faithful FV3 duo grid?**

Oracle = the Zenodo 8327578 `symmetryclean` Fortran tree.

> **CORRECTION 2026-08-13.** The original text of this section said two copies
> exist and are byte-identical, and told the reader to cite the
> `fv3_recon/duo_model/` copy. Both statements are wrong, and they corrupted two
> line numbers in this document (see the barrier citations below). There are
> **three** copies, and `dyn_core.F90` differs between them:
>
> | path | `dyn_core.F90` | status |
> |---|---|---|
> | `fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean/` | md5 `e5a5fab9…`, 3128 lines | **THE oracle** — the tree the verbatim extracts and `full_step_oracle_parity.py` cite |
> | `Code/FV3/duogrid_symmetryclean/.../` | md5 `e5a5fab9…`, 3128 lines | identical to the pinned tree; partial (no 3-D outer layer) |
> | `fv3_recon/duo_model/atmos_cubed_sphere-symmetryclean/` | md5 `0a5df09a…`, **3230 lines** | **INSTRUMENTED working copy, +102 lines** — do not cite |
>
> `sw_core.F90` IS byte-identical across all three (md5 `9e30c61d…`), so every
> `sw_core` citation in this document stands. `dyn_core.F90` is not, so
> `dyn_core` citations had to be re-derived. **Cite the pinned tree.**

The pinned tree is the full tree: it carries `fv_dynamics`, `fv_mapz`,
`nh_core`, `nh_utils`, `fv_arrays`, `test_cases` as well as the SW core.

The oracle also ships **reference solutions**, in
`Code/FV3/duogrid_zenodo/extracted/Code and simulations files/` — 43 run
directories, each with `rundir/atmos_daily.nc` and, critically, a
`rundir/logfile.000000.out` carrying the **resolved** namelist echo:

| case | what it is | resolutions | duo + plain? |
|---|---|---|---|
| `sw.case2` | Williamson-2 | C48/C96/C192/C384/C768 | yes, hord 5/6/8/10, α 0 and 45 |
| `sw.case6` | Rossby-Haurwitz | C48 | yes |
| `sw.case8` | colliding modons | C48 | yes, α 0 and 45 |
| `nh.case-13` | **DCMIP-2016 J&W baroclinic wave** (`test_cases.F90:77`) | C48/C96/C192/C384/C768 | yes |

Nothing in this repository read `nh.case-13` before 2026-08-06.

---

## The answer

**One structural gap accounts for every measurement below.** The faithful duo
port exists and is bit-exact certified — `d_sw1…d_sw6_duo`, `d2a2c_vect_duo`,
`divergence_corner_duo`, `ext_scalar`/`ext_vector` — but it is **pure NumPy and
LEAF-ONLY**: its only importers are `tests/grids/test_fv3_native_*.py` and
`scripts/validate/fv3_native/*.py`, it is in no factory, and it is neither
jittable nor differentiable. The lane that actually runs is a separate JAX
cd-grid solver whose `_d_sw_native` (`packages/core/legoesm/core/fv3_sw_core.py:3359`)
is **monolithic**.

The certified duo grid is a test fixture, not the model.

## Measured against the oracle

Both layers, matched grids, matched days.

### Shallow water — Williamson-2, α=0, day 5

The exact solution has `v ≡ 0`, so `max|v|` **is** the cube imprint. Oracle and
matrix output are both on the same 181×360 canvas.

| arm | max\|v\| | rms(v) |
|---|---:|---:|
| ORACLE C48 **duo** hord8 | 0.0291 | 0.0126 |
| ORACLE C48 plain hord8 | 0.2400 | 0.0431 |
| ORACLE C48 **duo** hord6 | 0.0236 | 0.0111 |
| ORACLE C48 plain hord6 | 1.3835 | 0.1068 |
| ours cubed_sphere C36 | **0.5407** | 0.0686 |
| ours icosahedral ico5 | 0.3576 | 0.0231 |
| ours latlon 72×144 | 0.0001 | 0.0000 |
| ours spectral T21 | 0.0000 | 0.0000 |

The duo grid buys **8.2×** at hord8 and **59×** at hord6 — inside the oracle,
one variable. Our production cube sits between the oracle's two *non-duo* runs;
scaling C36→C48 at second order ((48/36)² = 1.78) gives ≈0.30, essentially the
oracle's **plain hord8** value. **Our production cube performs like a non-duo
FV3.**

### 3-D — J&W baroclinic wave, area-weighted rms′ of `p_s` (hPa)

Alignment-free metric (invariant under the 0.5..359.5 vs −179.5..179.5
convention difference and under a uniform offset). Instrument:
`scripts/validate/fv3_native/jw_duo_oracle_compare.py`.

| arm | d1 | d3 | d5 | d7 | d9 |
|---|---:|---:|---:|---:|---:|
| oracle **duo** | 0.0295 | 0.0560 | 0.1634 | 0.6174 | 2.380 |
| oracle plain | 0.0344 | 0.0705 | 0.2156 | 0.8041 | 3.117 |
| ours latlon sigma | 0.0331 | 0.0585 | 0.1286 | 0.4308 | 1.658 |
| ours cube hybrid | 1.976 | 3.673 | 5.237 | 6.370 | 7.009 |
| ours cube sigma | **10.04** | 14.64 | 15.81 | 16.01 | 15.66 |

The oracle and our lat-lon grow **80×** over eight days — that is the baroclinic
instability. Our cube starts at 10 hPa and grows **1.6×**, flat from day 4: a
static grid-locked pressure pattern the physical wave never climbs out of, and
**6.7× larger than the oracle's fully developed day-9 wave**.

Holding the vertical coordinate at sigma and changing only the grid, lat-lon is
1.1× the duo oracle and the cube is **340×**. The matrix scores that case PASS
on mass drift and finiteness.

## The mechanism, from the oracle

The duo grid's core is **inter-panel flux averaging inserted between the `d_sw`
stages**: each panel computes its own seam flux, then `mpp_get_boundary` +
`0.5*(mine + neighbour)` makes it single-valued. Exactly two barriers are active
in one acoustic step:

- `dyn_core.F90:932` — `CGRID_NE`, after `d_sw1` (call `:891`, k-loop ends
  `:908`), on `allflux_x/y` for `iq=1` delp, `iq=4` temp, `iq>4` tracers;
- `dyn_core.F90:1049` — `BGRID_NE`, on the B-grid corner velocities
  `ubb`/`vbbtemp`.

Three more are **commented out** (`:1102`, `:1206`, `:1254`). There is **no flux
average anywhere in the `c_sw` stage** (`c_sw` at `:529`, k-loop ends `:536`,
next barrier `:932`).

Consequences for any port:

1. **The oracle has no monolithic `d_sw`.** It is six public routines
   (`sw_core.F90:74`) and the k-loop is split into four separate `do k=1,npz`
   nests (`dyn_core.F90:744, 914, 1066, 1218`) precisely so the barriers have
   somewhere to go. A monolithic `d_sw` cannot express them.
2. **`duogrid=.true.` forces `bounded_domain=.true.`** (`fv_arrays.F90:1512`), so
   every `bounded_domain` branch in `sw_core`/`tp_core`/`a2b_edge` flips under
   duo — the single most likely silent-divergence source.
3. …except **three call sites hardcode `bounded_domain=.false.`**, overriding it:
   `d2a2c_vect` (`sw_core.F90:151`), `ytp_v` (`:1316`), `xtp_u` (`:1374`).
   Mixed-mode by construction, with the original calls commented out above each.
4. **All four `fill_corner_region` call sites in `sw_core.F90` are commented
   out** (`:1747, 1755, 1763, 1764`). Adding duo corner fills to `d_sw5` moves
   *away* from the oracle.

## Prognostic stagger — persistent-D is the faithful configuration

`fv_arrays.F90:917-930`, verbatim: *"D-grid prognostatic variables: u, v, and
delp"* and *"The C grid component is 'diagnostic' in that it is predicted every
time step from the D grid variables."* Allocations confirm the edge layouts
(`:1163-1168`). `c_sw` takes D and produces diagnostic C/A
(`dyn_core.F90:524-535`); the outer-step exit `cubed_to_latlon` writes only
`ua,va` (`fv_dynamics.F90:798-799`); physics increments A-grid and maps back via
`update_dwinds_phys` (`fv_update_phys.F90:702-761`). The state never leaves D.

Our default does the opposite at every macro-step boundary: cc → D on entry
(`primitive_eq_cdgrid.py:2019-2059`), D → cc on exit.
`LEGOESM_HS_PERSISTENT_D=1` (`run_atmosphere_test_matrix.py:4307-4324`) is
therefore **the faithful setting, and default-off is a deviation** — a
conclusion from the oracle's state declaration alone, independent of the
Held-Suarez measurement (13.0 → 38.6 m/s). Two independent readings agree.

Its blast radius is exactly one read site inside `run_held_suarez`, so
`--test held_suarez --grid cubed_sphere` covers everything it can change; the
"needs the full matrix" caveat in `cube_structural_gaps_vs_fv3.md` was
overstated. Note it is between-macro-steps only — the physics adapter still
round-trips per RK stage (`primitive_eq_cdgrid.py:1613`/`:1045`), so the
measured gain is a lower bound.

## Missing: the Lagrangian vertical coordinate

First-class in FV3, not an optional refinement: `fv_dynamics.F90:568-674` calls
`Lagrangian_to_Eulerian` every outer split when `npz>4`. Our PE uses fixed
sigma/hybrid Eulerian vertical advection (`primitive_eq_cdgrid.py:774-868`); our
NH uses fixed z-star (`operators_3d.py:353-400`). A faithful transcription needs
`Lagrangian_to_Eulerian` 62-1080, `pkez` 1218-1273, `map_scalar` 1361-1453,
`map1_ppm` 1456-1547, `mapn_tracer` 1550-1663, `map1_q2` 1666-1756, `remap_2d`
1760-1852, `scalar_profile` 1855-2262, `cs_profile` 2265-2690, `cs_limiters`
2693-2768, `ppm_profile` 2772-3029, `ppm_limiters` 3032-3113. The dormant
`core/_future/vertical_remap.py` is a generic PPM, not this chain.

## Open NH failures

**TC2** — a finite-time singularity at a **resolved** 14.6 Δx scale, so
grid-scale dissipation is not the cure. Ranked: (1) terrain-following interface
geometry — the oracle enforces a *moving* surface `ws = (zs - gz_bottom)/dt`
(`nh_utils.F90:173-189`, `:293-309`) while our kernel imposes rigid `w=0` lids
(`compressible_euler.py:625-651`); (2) the Riemann treatment
(`nh_core.F90:138-202`); (3) the sponge — oracle `Ray_fast` damps upper-level
wind and **adds the removed dp-weighted momentum back** through `k=1:k_rf`
(`fv_dynamics.F90:902-938`), preserving the column mean, and never touches `pt`
(*"no special damping of potential temperature"*, `dyn_core.F90:833-866`); ours
is a pure multiplication with no restoration
(`compressible_euler_cdgrid.py:1592-1623`) and also damps `theta_prime`
(`:1010-1046`).

**TC3** — the initializer is the defect: `test_case_3.py:101-115` builds a
latitude-independent geographic east wind, singular at the poles, and sets
`rho_prime_data = 0` with no cyclostrophic correction. A `cos(lat)` arm passes,
which proves the **cause** — it is not automatically the specified profile. Fix
by transcribing the published DCMIP-2025 initializer, not by inventing a polar
regularization.

**baroclinic / latlon / hybrid** — new failure, blows up at step 1200
(`metric=nan`, `max|v|=1.2e25`, mass drift 2.5e+07), preceded by the warning that
these hybrid levels carry negative layer mass below 663.9 hPa.

## Retractions

- **"native duo stepper 0.0167 m/s vs production 0.54, ~23-30×".** That run
  applied `d_ext=0.02`; every duo deck **resolves** `D_EXT = 0.0`
  (`logfile.000000.out:406` and the other three). The filter acts on exactly the
  divergent seam mode the metric scores. Fixed 2026-08-06; any W2 duo figure
  from before that commit is not comparable to one after it.
- **"FV3's sponge relaxes to u00/v00 and never touches pt."** The
  never-touches-`pt` half holds; the relaxation description does not — see TC2
  (3) above.
- **"g083 is hung — TotalCPU=00:00:00 after 23 min."** `sacct TotalCPU` reads
  `00:00:00` for every job on this cluster, running or completed; the field is
  unpopulated and carries no information. A duo W2 run at C48 legitimately takes
  ~40 min per simulated day (C24 historical: 23-27 min for all 5 days). The
  codex exec rejections on g083 were real; the hang was not established.

## Instruments added

- `scripts/validate/fv3_native/jw_duo_oracle_compare.py` — the 3-D anchor,
  23 tests, each paired with a synthetic violation.
- `run_duo_stepper_w2.py --d-ext` defaulting to the deck value, plus
  `d_ext`/`git_sha`/`dt`/`n` in the npz (this was the one duo runner storing no
  commit).

## Not closable as a patch

Wiring the certified duo core into the JAX production lane, and the Lagrangian
remap. Both are scoped ports, listed here so they are not mistaken for tuning.
