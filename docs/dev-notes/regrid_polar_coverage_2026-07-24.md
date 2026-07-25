# Conservative-regrid polar coverage: why `require_full_coverage` is off on the forcing remappers

**Date:** 2026-07-24 · **Status:** revert shipped; helper fix deferred (see below)

## The invariant

A conservative lat-lon remap normalises each destination cell by its **own**
area (`weights = overlap_area / dst_cell_area`, no renormalisation by the
weight sum), so a destination cell the source only partially covers gets
`sum(weights) < 1` and the field comes back **reduced**.
`compute_overlap_weights(..., require_full_coverage=True)` raises on that.

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

Model→model remaps (`coupler/grid_remap.py`) use pole-clamped v-faces on both
sides, so there `sum == 1` *is* attainable and the flag stays on.

## What happened

The M8 hardening enabled `require_full_coverage=True` on the two
external-forcing remappers on the premise that "lat edges are clamped to the
pole, so any residual deficit is a bug". That premise is false for a
non-polar source. A real-forcing smoke test (Ginsburg job **9176233**, CORE-II
NYF) raised at **every** production target resolution:

18×36, 60×120 and 90×180 — always in lat row 0 or `n_lat-1`. The only worst-case
figure recorded from the run is **`|sum-1| = 6.6e-2` at 90×180**; the other two
were not captured.

Sanity check, not a reproduction: reconstructing the CORE-II grid by hand
(94 uniform rows from ±88.542°, outer edge ≈89.49°) gives 6.4e-2 for the 2°-wide
polar row of a 90×180 target — same magnitude as the recorded 6.6e-2, so the
measurement is *consistent* with a source edge near 89.5°. It is not an exact
match, and cannot be until the real latitude array is captured; treat 6.6e-2 as
the measurement and the rest as corroborating arithmetic.

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

## Deferred, not rejected: fix the check at the helper

Turning the flag off at the call site also loses the **seam/ghost** guard, which
is the failure mode that already caused the NaN above. The fix is to compare the
row sum against the **attainable** coverage instead of a hard `1.0`. The source's
covered extent is `[src_lat_edges[0], src_lat_edges[-1]]`, and the weight
construction is separable (`row_sum[j,i] = lat_frac[j]·lon_frac[i]` exactly,
since `weights = (lat_ov·lon_ov)/(dst_lat_area·dst_lon_area)`), so:

```python
# in compute_overlap_weights, NOT in the validator
lat_frac = np.clip(np.minimum(dst_sin[1:], src_sin[-1])
                   - np.maximum(dst_sin[:-1], src_sin[0]), 0.0, None) / dst_lat_area
expected = np.repeat(lat_frac, n_dst_lon)      # k -> lat_frac[k // n_dst_lon]
```

`np.repeat` matches the `dst_idx = j*n_dst_lon + i` flattening, so the shape and
ordering are right. But note **where** it goes: `_validate_full_coverage` has
none of `dst_sin`, `src_sin`, `dst_lat_area`, `n_dst_lon` in scope — they all
live in `compute_overlap_weights`. The helper needs a new `expected=` parameter
(or four new ones); this is not a self-contained 3-line edit inside the
validator. Reporting `divmod(worst, n_dst_lon)` instead of a flat index would
also name the offending `(lat_row, lon_col)`.

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
2. **"`grid_remap.py` is bit-identical" holds only for a latitude-global source.**
   For a global grid `lat_v[0]/[-1]` give `sin = ∓1.0` exactly, so `lat_frac` is
   IEEE-exactly 1.0 and nothing changes. But `make_latlon_remapper` gates only on
   `_is_regular_latlon`, never on global-ness, and `compute_v_face_coords`
   half-cell-extrapolates whatever band it is given — so a regional/nested source
   that **raises today** would silently pass under the sketch. Either qualify the
   claim or assert the source band is global in `grid_remap.py`.

Not done here: it changes shared-infra behaviour and needs its own PR, its own
tests, and a codex adversarial review.

## Related

- `packages/core/legoesm/grids/conservative_regrid.py` — the helper
- `packages/ocean/legoesm/ocean/coupler/omip2_applicator.py` — OMIP2 forcing
- `packages/tools/legoesm/forcing/jra55_do.py` — JRA55-do cache build
- `packages/coupler/legoesm/coupler/grid_remap.py` — the one caller that keeps
  the flag on (and whose comment still states the now-refined premise)
