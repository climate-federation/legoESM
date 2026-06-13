"""Real-data smoke test for the surface-data producer: HWSD2 raster -> 0.25 deg.

Streams the full FAO HWSD v2.0 30 arc-second soil mapping-unit raster
(``HWSD2.bil``, 21600x43200 uint16) through
:func:`legoesm.land.surface_data.aggregate.aggregate_raster_streaming` to a
coarse land-fraction map, and plots it.  This exercises the *entire* raster ->
grid path on real 1 km data WITHOUT needing the HWSD2 attribute database (the
soil-property join is a separate step): a recognizable continents map confirms
the BIL reader, geotransform, cos(lat) weighting and block aggregation are all
correct.

Usage:
    python3 scripts/check_hwsd2_aggregate.py \
        --bil ~/Downloads/HWSD2_RASTER/HWSD2.bil --res 0.25
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# Importing anything under legoesm.land runs land/__init__, whose config import
# forms a partially-initialized cycle with driver.coupled_config unless driver is
# imported first.  Pre-import it (documented workaround; see tests/land/conftest.py).
import legoesm.driver  # noqa: F401,E402

from legoesm.land.surface_data.raster import open_bil_memmap
from legoesm.land.surface_data.aggregate import (
    aggregate_raster_streaming,
    row_cos_weights,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bil", default=os.path.expanduser("~/Downloads/HWSD2_RASTER/HWSD2.bil"))
    ap.add_argument("--res", type=float, default=0.25, help="target degrees")
    ap.add_argument("--out", default="results/land/hwsd2_landfraction_0p25.png")
    args = ap.parse_args()

    mm, hdr = open_bil_memmap(args.bil)
    print(f"raster: {hdr.nrows}x{hdr.ncols} {hdr.dtype}, NODATA={hdr.nodata}")
    nodata = hdr.nodata

    def land_transform(block, lat_block):
        valid = block != nodata          # SMU id present -> land
        return valid.astype(np.float64), valid

    # The land FRACTION is the valid-area fraction (0 over ocean, fractional at
    # coasts); the value-mean channel is unused for a pure presence mask.
    _mean, landfrac, clat, clon = aggregate_raster_streaming(
        mm, hdr, args.res, land_transform, coarse_rows_per_chunk=60
    )

    # Global area-weighted land fraction sanity (Earth land share ~0.29, but HWSD
    # excludes oceans *and* permanent ice shelves / open water, so expect ~0.27).
    w = row_cos_weights(clat)[:, None]
    global_land = float((landfrac * w).sum() / (np.ones_like(landfrac) * w).sum())
    print(f"coarse grid: {landfrac.shape}, global area-weighted land fraction = {global_land:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.pcolormesh(clon, clat, landfrac, cmap="YlGn", shading="auto", vmin=0, vmax=1)
    ax.set_title(f"HWSD2 land fraction @ {args.res} deg (global={global_land:.3f})")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.colorbar(im, ax=ax, label="land fraction")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
