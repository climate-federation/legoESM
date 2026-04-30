# Realistic-geometry topology fixes

This document logs every modification applied to the raw ETOPO/GEBCO
land-sea mask before it's used by the lat-lon C-grid ocean model.
Per Phase 2 of `realistic_geometry_lat_lon_plan.md`.

The goal is **transparency**: every place where reality is altered to
make the coarse-resolution model behave reasonably should be visible
and justifiable, not silently baked into the loader.

## Resolution context

| Grid | Cell size (equator) | Cell size (60° N/S) |
|---|---|---|
| 36×72 (5°) | 555 km | 280 km |
| 72×144 (2.5°) | 280 km | 140 km |
| 144×288 (1.25°) | 140 km | 70 km |

Real-world straits with widths < ~1 cell are unresolvable and would
appear closed in the raw mask, blocking dynamically important
exchanges (Bering, Indonesian throughflow, etc.).

## Active modifications (applied by `bathymetry.py`)

These are applied automatically when `BathymetryConfig.enforce_straits = True`
(the default).  Per the **`CRITICAL_STRAITS`** table in
`src/legoesm/ocean/bathymetry.py`:

| Strait | Lat | Lon | Min width (km) | Min depth (m) | Justification |
|---|---|---|---|---|---|
| Drake Passage | -60° | -67° | 300 | 3000 | ACC throughflow; resolves at all our grid sizes but enforced for safety |
| Gibraltar | 36° | -5.5° | 50 | 300 | Mediterranean exchange; <1 cell at all our resolutions |
| Bab el Mandeb | 12.5° | 43.3° | 40 | 150 | Red Sea exchange |
| Hormuz | 26.5° | 56.5° | 40 | 100 | Persian Gulf exchange |
| Malacca | 2.5° | 101.5° | 40 | 50 | Indonesian connection |
| Indonesian Throughflow | -3° | 120° | 200 | 1500 | Pacific↔Indian exchange; dynamically critical for global circulation |
| Mozambique Channel | -17° | 41° | 200 | 2500 | Indian Ocean throughflow |
| Denmark Strait | 66° | -27° | 150 | 600 | NADW overflow |
| Faroe Bank Channel | 61.5° | -8.5° | 100 | 800 | NADW overflow |
| Bering Strait | 65.8° | -169° | 60 | 40 | Arctic-Pacific exchange (typically closed at coarse resolution; this entry forces it open) |
| Torres Strait | -10° | 142° | 50 | 15 | Australia-PNG |
| English Channel | 50.5° | 1.5° | 60 | 40 | Mostly cosmetic at coarse resolution |
| Taiwan Strait | 24° | 119.5° | 80 | 60 | South China Sea |
| Windward Passage | 20° | -73.5° | 60 | 1500 | Caribbean |
| Florida Strait | 25.5° | -79.5° | 100 | 800 | Gulf Stream source |
| Luzon Strait | 20.5° | 121.5° | 100 | 2000 | South China Sea |

**Mechanism**: for each strait, all grid cells within a great-circle
corridor of half-width `min_width_km / 2` (scaled by
`strait_width_factor`) are forced to `ocean_mask = 1` and
`H_bathy ≥ min_depth_m`.

**Implications for diagnostics**:

- Coastlines plotted from the post-processed mask will show some
  straits artificially open compared to ETOPO; this is the design.
- The Indonesian Throughflow at coarse resolution is artificially
  wider and deeper than reality; published Sv estimates from this
  configuration should be treated as upper bounds.
- The Bering Strait is open with `H_bathy ≥ 40 m` even though the
  ocean model's pure z* coordinate makes a 40-m water column
  numerically marginal (column is severely compressed); this is
  enforced for connectivity but should be revisited at higher
  vertical resolution.

## Other modifications (applied by `bathymetry.py`)

### Isolated basin fill

When `fill_isolated_basins = True` (default), any ocean basin
disconnected from the main ocean by the binary land mask is filled
(set to land).  Mechanism: flood-fill from the largest connected
ocean component.

This typically removes:
- The Caspian Sea (lat 40°, lon 50°): a real lake, but
  not coupled to the global ocean
- The Aral Sea (lat 45°, lon 60°)
- Any sub-grid pools that emerge from the bilinear regridding

If you want to retain inland seas as distinct basins (for an
inland-sea diagnostic), set `fill_isolated_basins = False`.

### Minimum depth threshold

Cells with raw `H_bathy < H_min` (default 10 m) are converted to land.
This prevents:
- Thin-layer instabilities in the z* coordinate (very shallow water
  columns get severely compressed)
- Non-physical "ocean" cells over reefs, marshes, ice shelves where
  ETOPO records ocean but the cell is unsuitable for our coarse
  dynamics

### Maximum depth clamp

Depths are clamped at `H_max` (default 5500 m).  Mariana Trench
points (~11000 m) get clipped.  Effect on global mean: <0.1% (trench
volume is a tiny fraction of global ocean).

### Land cell H_bathy = H_max (cosmetic)

After all modifications, `load_bathymetry_latlon_cgrid` sets
`H_bathy = H_max` on land cells.  This is **cosmetic** — the
`land_mask` prevents any flow on land — but it makes the z* Jacobian
`(eta + H) / H_max` smooth at the coastline, simplifying numerics.
The land's "depth" field is never used dynamically.

## Resolution-specific issues

### At 5° (36×72)

- Drake Passage: 1 cell wide.  Marginal.
- Indonesian Throughflow: 1 cell wide.  Marginal.
- Bering Strait: 0 cells wide if not enforced.  Always closed
  unless we force it.

The 5° configuration is **not recommended** for realistic geometry —
use 2.5° as the floor.

### At 2.5° (72×144) — recommended starting resolution

- Drake Passage: 2-3 cells wide.  Reasonable.
- Indonesian Throughflow: 2-3 cells wide.  Marginal.
- Bering Strait: 0-1 cells wide.  Strait enforcement required.
- Most coastal current systems are massively under-resolved.

### At 1.25° (144×288) — production target

- Drake Passage: 5+ cells wide.  Well-resolved.
- Bering Strait: 1 cell wide.  Marginal but no enforcement strictly
  required.
- Continental shelves resolved (~3-5 cells).

## Status

**No modifications applied beyond the defaults listed above** as of
the realistic-geometry implementation.  Phase 4 (50-yr realistic-
geometry run) may surface additional resolution-specific fixes; if
so, they will be documented here.
