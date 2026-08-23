"""How much halo does a lat-lon band ship for each row it owns?

The lane splits the globe into latitude bands, one per device. A band's halo
is two full circles of longitude regardless of how thin the band is, so as
devices are added the band gets thinner and the halo does not: past about a
hundred devices a band ships several times more halo than it owns.

Splitting longitude as well turns the band into a tile, whose perimeter
shrinks with it. This prints the ratio for both, so the size of that prize is
a number rather than an intuition.

    python scripts/validate/latlon_halo_surface_ratio.py --devices 64 128

Pure geometry: no model, no GPU, nothing measured on a machine. It says what
the decomposition WOULD ship, not what the step costs -- a two-dimensional
split was measured once at 64 devices and lost 22%, so the bytes are clearly
not the only term. Read this next to that receipt, never instead of it.
"""

from __future__ import annotations

import argparse


def halo_rows(n_lat, n_lon, devices, p_lon, depth):
    """Rows a device sends per exchange, and rows it owns.

    A tile is (n_lat / p_lat) by (n_lon / p_lon). It sends `depth` rows off
    each of its two latitude faces, each of width n_lon / p_lon, and -- only
    when longitude is split -- `depth` columns off each of its two longitude
    faces, each of height n_lat / p_lat. Corners are counted once with the
    faces, which is what the packed exchange sends.
    """
    p_lat = devices // p_lon
    lat_face = n_lon // p_lon
    lon_face = n_lat // p_lat
    sent = 2 * depth * lat_face
    if p_lon > 1:
        sent += 2 * depth * lon_face
    return sent, lon_face * lat_face


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--n-lat", type=int, default=128)
    parser.add_argument("--n-lon", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()

    for devices in args.devices:
        print(f"\n{args.n_lat}x{args.n_lon}, {devices} devices, "
              f"halo depth {args.depth}")
        print(f"{'lon splits':>11} {'sent':>8} {'owned':>8} "
              f"{'sent/owned':>11} {'vs bands':>9}")
        bands = None
        p_lon = 1
        while p_lon <= devices:
            if (devices % p_lon == 0 and args.n_lon % p_lon == 0
                    and args.n_lat % (devices // p_lon) == 0):
                sent, owned = halo_rows(args.n_lat, args.n_lon, devices,
                                        p_lon, args.depth)
                if bands is None:
                    bands = sent
                print(f"{p_lon:>11} {sent:>8} {owned:>8} "
                      f"{sent / owned:>11.2f} {sent / bands:>8.2f}x")
            p_lon *= 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
