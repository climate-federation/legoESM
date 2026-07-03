# Forced Ocean Driver (Tropical OMIP Item 4) — Revised Scope

**Status**: scoping (revised)
**Date**: 2026-05-02
**Parent docs**: `tropical_omip_plan.md` Item 4; `bulk_flux_ly09_audit.md`;
`jra55do_pipeline_plan.md`.

The original parent plan budgeted Item 4 ("ForcedOceanDriver") at **1.5
weeks**. After auditing the existing driver infrastructure, that estimate
collapses to **~4 days** because most of what I had assumed needed to be
built **already exists**.

## 1. What already exists (recon summary)

`scripts/run/run_omip.py` — **783 LOC of working OMIP-style driver** for
*all four ocean grids* (cubed-sphere, lat-lon, MPAS, spectral). It already
provides:

| Piece | Where in `run_omip.py` | Status |
|---|---|---|
| CLI scaffolding (grid, days, dt, resolution, restart) | `parse_args:62`, `run_omip_single:639` | use as-is |
| WOA18 init dispatch | `--woa-t/--woa-s` flags wired to init | use as-is |
| Per-grid setup (config + state + model) | `_create_setup:159`, `_init_rest_state:260` | use as-is |
| Surface-forcing config builder | `_build_surface_forcing:289` | extend |
| **Time loop** with check-finite + diagnostics | `_run_omip_loop:483` | extend |
| Haney SST/SSS restoring | `_apply_restoring:314` | reuse + extend |
| Scalar extraction (mean T, S, KE, total mass) | `_extract_scalars:393` | reuse |
| Output (CSV/JSON/PNG) | `_save_output:574` | reuse |

Plus, the supporting library pieces also exist:

| Piece | File | API |
|---|---|---|
| WOA18 loader + 3D interp | `src/legoesm/ocean/init_woa.py` | `load_woa18(...)`, `init_ocean_from_woa(grid, z_coord, T_path, S_path)` |
| Rest-state init | `src/legoesm/ocean/init_latlon_cgrid.py` | `rest_state_latlon_cgrid_ocean(...)`, `replace_land_mask(...)` |
| Sponge layer | `src/legoesm/ocean/sponge.py` | `SpongeForcing` namedtuple, `compute_sponge_gamma_latlon(grid, lat_south, lat_north, width_deg, timescale_days)` |
| Restoring surface forcing | `src/legoesm/ocean/physics/surface_forcing/restoring.py` | `restoring_surface_forcing(T, S, grid, cfg)` |
| Bulk-flux pipeline | `coupler/coupler.py:ocean_tile_response()` | LY09-fixed, takes `AtmToSurface` |
| Freshwater pipeline | `ocean/freshwater.py:freshwater_from_coupler()` | builds `FreshwaterForcing` |
| Step API | `ocean_model_latlon_cgrid.py:step()` | accepts `freshwater=`, `surface_forcing=`, `sponge=` |

## 2. What's missing for tropical OMIP

The driver needs three things `run_omip.py` does not currently do:

### Gap A — JRA55-do glue layer

Bridges the `JRA55Slice` produced by Item 2's loader to the `AtmToSurface`
and `FreshwaterForcing` structs the existing pipeline already consumes.
Lives in the driver, not in the bulk-flux or freshwater code (both of which
are correct as-is).

```python
def jra55_to_atm_surface(slice: JRA55Slice, lat, lon, day) -> AtmToSurface:
    """Build the 2D AtmToSurface struct from a single time slice."""
    rho_a = slice.psl / (constants.R_d * slice.tas
                         * (1.0 + 0.61 * slice.huss))
    return AtmToSurface(
        u_lowest=slice.uas, v_lowest=slice.vas,
        T_lowest=slice.tas, q_lowest=slice.huss,
        p_lowest=slice.psl, p_surface=slice.psl,
        rho_lowest=rho_a,
        sw_down=slice.rsds, lw_down=slice.rlds,
        precip_total=slice.prra, precip_snow=slice.prsn,
        cos_zenith=cos_solar_zenith(lat, lon, day),
        co2_ppmv=jnp.array(400.0),
        has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0),
    )

def jra55_to_freshwater(
    slice: JRA55Slice, lhflx: jnp.ndarray,
) -> FreshwaterForcing:
    """Build FreshwaterForcing from forcing reads + computed evap."""
    evap = lhflx / constants.L_v   # kg/m²/s
    return FreshwaterForcing(
        precip=slice.prra + slice.prsn,
        evap=evap,
        runoff=slice.friver_redistributed,    # already coastal-redistributed
        ice_fw=jnp.zeros_like(slice.prra),    # zero in tropical OMIP
    )
```

**Effort**: ½ day. Lives in `scripts/run/run_omip.py` or a thin
`src/legoesm/ocean/driver_glue.py` if it grows.

### Gap B — Replace "restoring-only" forcing with "bulk-flux + sponge + SSS-restoring" mode

`run_omip.py` currently runs a Haney restoring driver: surface T relaxes to
WOA, no momentum forcing, no freshwater. For tropical OMIP we need a new
forcing mode where:

1. Bulk fluxes computed from JRA55-do via `ocean_tile_response()`
   (LY09-compliant after Item 1 fixes).
2. Freshwater applied via `model.step(freshwater=...)`.
3. SSS restoring applied weakly globally (~5×10⁻⁷ m/s piston velocity)
   via `restoring_surface_forcing(...)` using only the salinity term.
4. Sponge layer at 60°N/60°S applied via `model.step(sponge=...)` —
   relaxes T, S to WOA monthly clim with τ ramp from 30 d (55°) to 5 d
   (60°), and zero-velocity in the last 2 cells.
5. T capped at `T_freeze_ocean` inside the sponge zone (the freeze-T cap
   that stands in for sea ice).

The new dispatch hangs off a `--forcing-mode` CLI flag:

```python
p.add_argument(
    "--forcing-mode",
    choices=["restoring", "jra55_do_tropical"],
    default="restoring",
    help="Surface forcing source (default: restoring).",
)
```

Inside `_run_omip_loop`, if `forcing_mode == "jra55_do_tropical"`:

```python
# Per-step:
slice = load_jra55_slice(forcing_cache, day)
atm_to_sfc = jra55_to_atm_surface(slice, lat, lon, day)
tile_resp = ocean_tile_response(
    atm_to_sfc, state.T[..., 0], state.u[..., 0], state.v[..., 0],
    coupler_cfg,
)
fw = jra55_to_freshwater(slice, tile_resp.lhflx)

# Build OceanSurfaceForcing from tile_resp (existing path; just
# pass the τ, Q_net produced by the LY09 bulk flux instead of
# Haney restoring).
surface_forcing = OceanSurfaceForcing(
    sw_down=atm_to_sfc.sw_down,
    q_net=tile_resp.q_net_into_ocean,   # already net of LW up
    tau_x=tile_resp.tau_x,
    tau_y=tile_resp.tau_y,
)
# T_freeze cap inside sponge mask:
surface_forcing = _apply_freeze_cap_in_sponge(
    surface_forcing, sponge_mask,
)

state = model.step(
    state, dt,
    freshwater=fw,
    surface_forcing=surface_forcing,
    sponge=sponge_forcing,
)
# Apply SSS-only restoring after dynamics:
state = _apply_sss_restoring(state, S_target_woa_monthly, dt, tau_sss)
```

**Effort**: ~1.5 days. Most pieces exist; this is wiring.

### Gap C — Diagnostics and output

The OMIP diagnostics module (Item 7 in the parent plan, ~1.5–2 wk) and the
NetCDF output infrastructure (Item 8, ~3 d) **are separate items** — not
folded into Item 4. The driver wires them in at the end:

```python
# Inside _run_omip_loop, at month boundary:
omip_diag = omip_diagnostics.flush_monthly(omip_diag, output_dir, year, month)
```

For Item 4 itself, only the *placeholder* hooks need to exist — the
`OmipDiagnostics` carry slot and the per-step accumulator call. The
contents are Item 7's job.

## 3. Effort breakdown — Item 4 only

| Sub-task | Effort | Notes |
|---|---|---|
| A. JRA55-do glue (`jra55_to_atm_surface`, `jra55_to_freshwater`) | ½ d | Pure Python; AD-clean by construction |
| B1. New `--forcing-mode jra55_do_tropical` dispatch in `run_omip.py` | ½ d | Adds ~50 LOC of CLI + branch |
| B2. Per-step forcing read + bulk-flux + freshwater + sponge wiring | 1 d | Most pieces exist; integration testing |
| B3. SSS-only restoring + T_freeze cap + sponge mask construction | ½ d | Reuses `restoring_surface_forcing`, `compute_sponge_gamma_latlon` |
| C. Diagnostics carry slot (placeholder only) | ½ d | Real work in Item 7 |
| Smoke test: 30-d run produces sensible AMOC tendency at 26.5°N | 1 d | First-look numbers, not validation |
| **Total** | **4 d** | (was 1.5 wk in parent plan) |

**Net schedule impact**: Item 4 shrinks by ~3 days. That time is **not
freed** — it goes into Item 7 (diagnostics), which the original Explore on
the bulk-flux audit wrongly thought existed. Item 7 is still the longest
single piece (~1.5–2 wk).

## 4. Architectural decision — extend or fork `run_omip.py`?

`run_omip.py` is a 783-LOC multi-grid driver. Adding a tropical-OMIP mode
inside it (via `--forcing-mode`) keeps everything in one place and reuses
the CLI / setup / scalar-output / restart machinery. The alternative is a
fork to `run_omip_tropical.py` that imports the helpers from `run_omip.py`.

**Recommendation: extend in place.** Three reasons:

1. The branching is lightweight (one new `if forcing_mode == ...` block in
   `_run_omip_loop`).
2. Multi-grid testing matters for tropical OMIP too — being able to run
   the same script with `--grid mpas` or `--grid cubed_sphere` for
   sensitivity tests is a feature.
3. Forking creates a maintenance trap (every CLI flag added to one has to
   be back-ported to the other).

If the tropical-OMIP branch grows past ~300 LOC of new code inside
`run_omip.py`, refactor at that point — not preemptively.

## 5. Stage-1 vs Stage-2 (no segment compilation in Stage 1)

`run_omip.py` uses a **naive Python time loop** (no `lax.scan`, no segment
compilation). At 1° / dt=300 s, one year = 105,120 steps, each launching
its own JIT-compiled `model.step()`. Per the Explore: "10–20 wall-hours
for 60 yr in Python loop, ~1–2 hours with segments."

**Decision: stay with the naive loop for cycle 1.** Reasons:

- Cycle 1 (62 yr) projects to ~12 GPU-hr per the parent plan compute
  budget — already feasible.
- Segment compilation adds an `OceanSegmentCarry` + `build_ocean_segment_fn`
  refactor (~800 LOC of new code in `src/legoesm/driver/`) that is
  **out of scope for Item 4**. The atmosphere-side `compiled_segments.py`
  pattern can be ported in a future "Phase B prerequisite" task.
- Diagnostics that flush per-month are *easier* in a naive Python loop —
  the host code can read `state` directly without unpacking a `SegmentCarry`.

If cycle-1 wall time exceeds ~2 days, revisit Stage 2. Until then, ship.

## 6. Risks specific to Item 4

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Forcing-read I/O dominates** the per-step cost (Zarr read every step). | Pre-load 1 day = 8 records into host RAM; pass to `model.step` per step. Driver-level cache. ~5 LOC. |
| 2 | **`OceanSurfaceForcing` schema mismatch** between what `_build_surface_forcing` produces (restoring path) and what we need (bulk-flux path). | Verify the `OceanSurfaceForcing` namedtuple has all fields the lat-lon C-grid `model.step()` consumes. Likely needs no change; `tile_resp` provides everything except sw_down (which we pass through). |
| 3 | **Cosine zenith angle wrong** at high latitudes near sponge → spurious SW heating. | `cos_solar_zenith` returns `max(cos_z, 0)` (no SW at night). Verify with a 1-day diurnal-cycle plot before launching. |
| 4 | **Sponge T_freeze cap interacts badly with restoring** — the cap zeros surface heat flux but restoring keeps pulling T → T_woa. | Apply the cap **after** restoring inside the sponge zone, not before. Order matters. Document. |
| 5 | **JRA55-do `friver` redistribution to coastal cells** lands runoff on the sponge zone for Greenland / Antarctic Peninsula → freshwater piles up. | Item 2 puts the redistribution in `jra55_do.py`. Cap the per-band runoff at "ocean cells inside the active 60°S–60°N domain" only; runoff outside is dropped (not great but defensible for tropical OMIP). |
| 6 | **CFL stability** at 1° with dt=300 s in WBC regions. | `run_omip.py` already has `_check_finite` and CFL diagnostics; add a max-CFL trace to the per-day output. Phase 4(c) realistic-geometry showed dt=300 s stable on ETOPO. |

## 7. Test plan (Stage 1)

- **Unit**: `test_jra55_to_atm_surface` — feeds a fake `JRA55Slice`,
  verifies field shapes, `rho_a` sign, dtype consistency.
- **Unit**: `test_jra55_to_freshwater` — `lhflx > 0` produces
  `evap > 0`, P − E sign correct.
- **Smoke (1 day)**: `run_omip.py --forcing-mode jra55_do_tropical
  --days 1 --grid latlon --resolution 1deg` runs to completion without
  NaN/Inf. AMOC@26.5N is finite (not yet meaningful at 1 day).
- **Integration (30 day)**: same as above with `--days 30`. Diagnostics
  show: realistic surface stress (peak ~0.3 Pa in storm tracks),
  global-mean Q_net within ±10 W/m², MLD evolves seasonally,
  AMOC@26.5N order-of-magnitude correct (a few Sv to ~20 Sv).
- **AD smoke**: not required for the production driver, but run
  `jax.grad` through one step over the new forcing path to verify
  nothing broke the existing AD compatibility.

## 8. Sequencing within Item 4

| Day | Work |
|---|---|
| 1 | Gap A: JRA55-do glue (`jra55_to_atm_surface`, `jra55_to_freshwater`) + unit tests |
| 2 | Gap B1+B2: CLI dispatch + per-step forcing/bulk/freshwater wiring |
| 3 | Gap B3: SSS restoring + T_freeze cap + sponge mask + Gap C placeholder |
| 4 | Smoke test (1d → 30d), debug, profile |

After Day 4, Item 4 is "done" enough that Item 7 (diagnostics) and Item 8
(output) can layer on top without touching the driver core.

## 9. What this plan does *not* commit to

- **Segment compilation** — explicitly deferred. Not needed for cycle 1.
- **Multi-rank MPI ocean** — `run_omip.py` is single-rank. MPI ocean
  exists (`tests/distributed/`) but isn't wired into `run_omip.py`. For
  cycle 1 at 1° single-rank is fine (~12 GPU-hr).
- **Restart from mid-cycle** — `run_omip.py` has restart hooks but
  they're not exercised in the tropical-OMIP path. Add only if needed
  during cycle 1 (e.g., if a debug iteration needs to skip year 1).
- **Online σ₂ MOC binning** — that's Item 7 (diagnostics).

## 10. Net effect on parent plan

Re-baselining `tropical_omip_plan.md` Item 4: **1.5 wk → 4 days**.

The parent plan's total dev budget was 6–8 weeks. After:
- Item 1: 1.5d → 2d (audit + impl, on budget)
- Item 2: 1 wk → 5 d (on budget)
- Item 4: **1.5 wk → 4 d (saved ~5 d)**

Updated single-developer total: **~5–7 weeks** (was 6–8).

The savings flow into either: (a) start cycle 1 sooner, (b) take the freed
days to harden Item 7 diagnostics (which is the real critical-path long
pole), or (c) start Phase B sea-ice validation in parallel with cycle 1.

Recommend (b): the diagnostics module is what makes the difference between
"a forced ocean run" and "an OMIP-quality forced ocean run". Spend the
saved time there.

## 11. Next action

Start Day-1 work: implement `jra55_to_atm_surface` and `jra55_to_freshwater`
glue functions plus their unit tests. These are AD-clean, dependency-free,
and let us begin smoke-testing the dispatch path the moment Item 2's
JRA55-do cache exists (which can run in parallel).
