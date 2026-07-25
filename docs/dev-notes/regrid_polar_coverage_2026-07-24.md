# Conservative-regrid polar coverage: the attainable-coverage reference

**Date:** 2026-07-24 (updated 2026-07-25) · **Status:** shipped — the check now
compares against attainable coverage and is ON at all three callers

## The invariant

A conservative lat-lon remap normalises each destination cell by its **own**
area (`weights = overlap_area / dst_cell_area`, no renormalisation by the
weight sum), so a destination cell the source only partially covers gets
`sum(weights) < 1` and the field comes back **reduced**.
`compute_overlap_weights(..., require_attainable_coverage=True)` raises on that.

The catch: `sum(weights) == 1` is only *attainable* when the source truly spans
the destination. **Every real forcing dataset stops short of the pole**, so its
outermost destination latitude row legitimately under-covers:

Quote the **cell centre** and the **inferred edge** separately — they are not the
same number, and the deficit is set by the edge:

| source | outermost cell centre | inferred outer edge |
|---|---|---|
| CORE-II NYF (94 rows) | ±88.542° | ≈±89.5° |
| JRA55-do TL319 (320 rows) | ±89.570° | ≈±89.85° |

The edges are approximate because `_edges_from_centers_deg` /
`_grid_edges_from_centers` infer them as `c[-1] + (c[1]-c[0])/2` — the *first*
spacing, which on a Gaussian grid is not the local one — so the exact value
depends on the real latitude array (not captured from the run). What is robust,
and all the argument needs: **both land short of ±90**, so `np.clip(..., ±π/2)`
is inert on real forcing. See "What was kept" for when the clamp does bite.

Model→model remaps (`coupler/grid_remap.py`) use pole-clamped v-faces, so for
**global** model grids `sum == 1` *is* attainable and the check stays strict. Note
this does not hold for every grid that path accepts: a regional Mercator/DINO frame
also satisfies `_is_regular_latlon`, its outer row against a global destination is
*not* fully coverable, and it is correctly rejected — see caveat 2.

## What happened

The M8 hardening enabled `require_full_coverage=True` on the two
external-forcing remappers on the premise that "lat edges are clamped to the
pole, so any residual deficit is a bug". That premise is false for a
non-polar source. A real-forcing smoke test (Ginsburg job **9176233**, CORE-II
NYF) raised at **every** production target resolution:

18×36, 60×120 and 90×180 — always in lat row 0 or `n_lat-1`. The only worst-case
figure recorded from the run is **`|sum-1| = 6.6e-2` at 90×180**; the other two
were not captured.

The geometry reproduces that measurement exactly. CORE-II T62 is the 94-point
Gaussian grid, so the latitude array is fully determined —
`asin(leggauss(94))` gives outermost centre **−88.541950°** and, through
`_edges_from_centers_deg`, outer edge **−89.486342°** (gap 0.513658°). The
resulting deficit for the 2°-wide polar row of a 90×180 target is **6.5967e-2**,
i.e. the recorded 6.6e-2. (A cruder reconstruction assuming *uniform* 94 rows gives
6.4e-2 — close, but the Gaussian one is exact, so quote that.)

So it would have **aborted production OMIP preprocessing**. The deficit is
physical (no data poleward of the source's outer edge), not a seam/ghost defect
— it persists at `dd/ds ~ 1.1`, where the ghost-column logic is irrelevant.

JRA55-do was reverted on the same geometric argument rather than on a
measurement: no cached JRA55 was available at the time to reproduce it.

## What was kept

- The **pole clamp** on inferred lat edges (`np.clip(..., ±π/2)`). It prevents an
  extrapolated edge from spilling *past* the pole, where `sin` caps below 1 and
  fabricates a spurious polar gap. Note this is a **coarse-grid** guard, not a
  rounding guard, and it is inert on the real forcing grids above: it bites when
  uniform-spacing inference overshoots by degrees, as in the 8-row test fixture
  (outer centre 86° → inferred edge ≈98°). Pinned by
  `tests/unit/test_jra55_do.py::test_cache_polar_coverage_after_lat_clamp`
  (RED without it: polar `tas` ~279.8 K instead of 290 K).
- The **`ceil(dd/ds)` ghost columns** in `omip2_applicator`. That fix is correct
  and closes the genuine 0/360 seam deficit (a single ghost left the
  seam-straddling column HALVED, and the 10-m pressure iteration then NaN'd).

## SHIPPED: the check now compares against attainable coverage

Turning the flag off at the call site also lost the **seam/ghost** guard, which is
the failure mode that already caused the NaN above. That is fixed:
`require_full_coverage` is now **`require_attainable_coverage`**, it compares the
row sum against what the source can actually supply, and **all three callers have
the guard on again** (`omip2_applicator`, `jra55_do`, `grid_remap`).

The weight construction is separable (`row_sum[j,i] = lat_frac[j]·lon_frac[i]`
exactly, since `weights = (lat_ov·lon_ov)/(dst_lat_area·dst_lon_area)`), so the
reference is a latitude factor with the longitude factor held at 1.0:

```python
# _attainable_lat_coverage(), called from compute_overlap_weights
covered = np.clip(np.minimum(dst_sin[1:], src_sin[-1])
                  - np.maximum(dst_sin[:-1], src_sin[0]), 0.0, None)
frac = np.zeros_like(covered)          # masked divide, NOT an epsilon clamp:
ok = dst_lat_area > 0.0                # max(dst_lat_area, 1e-30) would bind at a
frac[ok] = covered[ok] / dst_lat_area[ok]   # different threshold than the weights'
expected = np.repeat(frac, n_dst_lon)  # own max(dst_cell_area, 1e-30) (a lat*lon
                                       # PRODUCT), and a mismatched pair raises
                                       # spuriously, blaming a longitude seam.
```

`np.repeat` matches the `dst_idx = j*n_dst_lon + i` flattening. The validator took
a new `expected=` parameter, since `dst_sin`/`src_sin`/`dst_lat_area`/`n_dst_lon`
all live in `compute_overlap_weights`, not in it; failures now report
`(lat_row, lon_col)` via `divmod` instead of a flat index.

### Relaxing latitude is not enough on its own — the reference needs a FLOOR

Relaxing the reference alone is strictly **worse** than the bug it fixes: a row
matches its own attainable coverage *by construction*, so a row the source cannot
touch has `expected = 0` matched by `row_sum = 0`, passes, and silently emits a
field of **zeros**. At 1° with a ±89° source band that is `T_air = 0 K` in the
polar model rows — the very bulk-flux / 10-m-pressure NaN this guard exists to
prevent.

**Two attempts failed before the shipped one. Both were caught by adversarial
review, with numbers, and are recorded because each looks obviously right.**

*Attempt 1 — bound shortfall POSITIONALLY* ("only the outermost destination row
may be short"). It assumes the polar gap is narrower than one destination row.
CORE-II's gap is **0.514°**, which spans 4 rows of a 0.5° target and 6 of a 0.25°
target, so it hard-aborts `--latlon-res 360x720` (first failing `n_dst_lat = 351`;
JRA55-do's 0.151° gap fails from 1189, i.e. `--target-resolution-deg 0.125`). It
also never bounded *magnitude*, so the all-zero hole above survived untouched.

*Attempt 2 — bound it GEOMETRICALLY* (short only where the row pokes past the
source band, `(dst_sin[:-1] < src_sin[0]) | (dst_sin[1:] > src_sin[-1])`, and
`lat_frac > tol`). This is **algebraically identical to attempt 1**, which is
worth internalising: for a *contiguous* source band the outer edge cuts **exactly
one** destination row, and every row beyond it is exactly zero-covered. So
"reaches beyond and retains overlap" ≡ "is the row containing the source edge" ≡
the positional rule. Same abort at `n_dst_lat = 351`. Its own regression test
failed, which is how it was caught.

**Shipped: an explicit, caller-chosen floor.** `_required_lat_coverage` returns
`lat_shortfall_floor` for a row extending past the source band and `1.0`
elsewhere; the guard fires on `lat_frac < required - tol`.

- `lat_shortfall_floor = 1.0` (**the default**) reproduces the original hard-1.0
  check exactly. `coupler/grid_remap.py` keeps it — model→model, any shortfall is
  a bug. Relaxing latitude unconditionally there had made a non-global Mercator
  source return `coverage × field` in its outer rows instead of raising: for a
  `lat_max = 70°` source into any global lat-lon destination the measured coverage
  there is **0.0**, i.e. a constant 290 K SST arriving as **0 K**.
- The forcing callers opt in at **0.5**. Weights are *not* renormalised, so a row
  with coverage `f` returns `f × field`; the floor is therefore how much dilution
  the caller ships, **not a numerical epsilon**. Measured CORE-II outer-row
  coverage: 0.997 @ 10°, 0.934 @ 2°, **0.736 @ 1° (production `180x360`)**, 0.531 @
  0.75°, 0 @ 0.5° and finer. JRA55-do's gap is 3× narrower: 0.977 @ 1°, still
  0.633 @ 0.25°.
- Valid range is `[1e-3, 1]`, not `(0, 1]`. A floor at or below `_COVERAGE_TOL`
  drives the comparison threshold negative and the guard **inverts** — an all-zero
  row passes. (Found by a red test at `floor = 1e-9`.) 1e-3 is also where a floor
  stops being a defensible policy: 0.1% coverage turns 250 K into 0.25 K.

**Known limit, not fixed here — two distinct regimes, don't conflate them.** For
CORE-II onto a global target at the shipped floor 0.5:

| `n_dst_lat` | res | outer coverage | why it raises |
|---|---|---|---|
| ≤ 247 | ≥ 0.729° | ≥ 0.500 | passes |
| 248 – 350 | 0.726° – 0.514° | 0.499 → 0.0024 | **floor** violation — data *is* present; a lower floor would accept it |
| ≥ 351 | ≤ 0.513° | exactly 0 | **no source data at all** — no floor above zero can accept it |

So the first abort is at 0.726° (not ~0.75°: 0.75°/240 rows passes at 0.531), and it
is a *policy* abort; only below 0.513° does the row become genuinely empty. JRA55-do
first aborts at `n_dst_lat = 841` (0.214°). This is a behaviour change — such
configurations previously emitted fabricated (or reduced) values there. Making them
usable needs a renormalising or pole-filling remap, i.e. a change to the
**weights**, not to this check. That is the natural follow-up, and it is worth an
issue rather than a sentence here: at production 1° the two polar rows already ship
at 0.736 coverage (`T_air ≈ 184 K`, `slp ≈ 74.6 kPa`), unchanged by this PR — the
weights are byte-identical to `main`, which simply never checked — but the guard now
*blesses* it.

Pinned by `tests/unit/test_conservative_regrid.py`:
`test_default_floor_is_strict_and_reproduces_the_hard_unit_check`,
`test_polar_taper_within_floor_passes_and_is_reduced_not_zero`,
`test_row_below_the_floor_raises_instead_of_shipping_a_diluted_field` (±50° source
→ 0.201 coverage, so it straddles the floor rather than retesting the zero case),
`test_row_entirely_outside_source_raises_at_every_valid_floor` and
`test_floor_outside_valid_range_is_rejected` (which pins `_MIN_LAT_FLOOR >
_COVERAGE_TOL` directly, so the inversion above cannot be reintroduced by lowering
the constant), plus `test_lon_deficit_still_raises_with_a_relaxed_lat_floor` —
relaxing latitude must never relax longitude, tested on a ±50° source so the floor
is genuinely in play. The two caller-level span preconditions are pinned by
`tests/ocean/unit/test_omip2_applicator.py::test_conservative_regrid_rejects_partial_longitude_source`;
without it, deleting the `check_axis_span` call leaves the suite green.

### Two caveats that a first draft of this note got wrong

1. **This does NOT restore a partial-longitude-source guard at the forcing call
   sites.** The tempting claim is "relax latitude only, so a source that does not
   span 360° still raises". False here: both forcing callers *pad the source
   longitude before* calling, with ±360 shifts of the source's own end columns.
   For a source that does not tile 360° that manufactures one enormous ghost
   cell spanning the whole missing sector — measured on a 16-cell source tiling
   only `[0, 180]`: OMIP2 (`n_ghost = ceil(dd/ds) = 4`) yields padded edges
   `[-225, 405]` with two 191.25°-wide cells, and JRA55's single `+360` ghost
   yields a 191.25° final cell. Both give `lon_frac = 1` everywhere, pass
   `_check_edges`, and do **not** raise. The seam/ghost guard itself survives (an
   under-sized `n_ghost` still leaves `lon_frac = 0.625`, the original bug), but
   the partial-lon guarantee does not. If that guard is wanted, it belongs in the
   callers as an explicit pre-pad assertion that the raw source tiles 360°
   (`abs(n_src_lon*ds - 360) < tol`) — which would also make the ghost padding's
   own uniform-spacing assumption (`dc = c[1] - c[0]`) explicit rather than
   implicit. Latent today, since production forcing is global.
   **Now done:** `check_axis_span(raw_lon_edges, 2π)` runs in both forcing callers
   *before* the pad — in the two FORCING callers only, whose sources are global
   datasets. Deliberately **not** in `grid_remap`: a longitude-span assertion
   there rejects DINO's regional-longitude Mercator frame (span 0.84 rad) that
   passes today. Note `cell_edges_1d(...,
   periodic_lon=True)` sets `edges[-1] = edges[0] + 2π` by construction, so the
   span must be inferred with `periodic_lon=False` or the assertion is vacuous —
   a mistake made and caught while writing this.
2. **"`grid_remap.py` is bit-identical" holds only for a latitude-global source.**
   For a global grid `lat_v[0]/[-1]` give `sin = ∓1.0` exactly, so `lat_frac` is
   IEEE-exactly 1.0 and nothing changes. But `make_latlon_remapper` gates only on
   `_is_regular_latlon`, never on global-ness, and `compute_v_face_coords`
   half-cell-extrapolates whatever band it is given — so a regional/nested source
   that **raises today** would silently pass under the sketch. Either qualify the
   claim or assert the source band is global in `grid_remap.py`.
   **Now resolved by keeping `grid_remap` STRICT, not by asserting a global band.**
   It passes no span assertions and takes the default `lat_shortfall_floor = 1.0`,
   which is the original hard-1.0 check — so its strength is preserved exactly,
   whatever band the grids happen to have. Span assertions there were tried and
   reverted, both being *stronger* than the old behaviour: `_is_regular_latlon`
   duck-types on `lat_v`/`lon`/`n_lat`/`n_lon`, which the non-global
   **Mercator/DINO** grids also satisfy (`lat_v` bounded by `lat_max_deg`, never
   ±π/2 — measured span residual −0.74 rad at `lat_max=70°`), so a `lat = π`
   assertion rejects legitimate Mercator→Mercator pairs and a `lon = 2π` one
   rejects DINO's regional frame (span 0.84 rad). Getting this wrong was not
   hypothetical: with the floor relaxed unconditionally and the assert removed, a
   `lat_max = 70°` Mercator source into a global lat-lon destination returned its
   outer rows at 0.0 coverage — a constant 290 K SST as **0 K** — instead of
   raising. (Measured `lat_v` there is ±69.85° for `n_lon ≥ 120`, ±68.79° for
   `n_lon ≤ 90`; span residual −0.70 to −0.74 rad depending on `n_lon`.)

## Related

- `packages/core/legoesm/grids/conservative_regrid.py` — the helper
- `packages/ocean/legoesm/ocean/coupler/omip2_applicator.py` — OMIP2 forcing
- `packages/tools/legoesm/forcing/jra55_do.py` — JRA55-do cache build
- `packages/coupler/legoesm/coupler/grid_remap.py` — model→model remap; keeps the
  strict default floor (1.0), i.e. the original hard-1.0 check, and asserts no spans
