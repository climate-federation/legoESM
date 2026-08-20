"""ACC-fix ACCEPTANCE metric: channel z(maxN2), reusing acc_thermal_wind.py's
geometry/EOS/channel-band setup verbatim (imported as a module -- NOT
re-derived) so this is on the SAME mask/EOS/depth-ladder as the recorded
166.6 m / 0.908 / 0.86->0.44 numbers.

acc_thermal_wind.py itself has no z(maxN2) metric; this fills that one gap by
importing its module-level state (tmask, gdept0, gdept1d, J0, J1, rho_of,
load_lego, load_nemo, contrast_profile, depth_split, DEEP_M) rather than
recomputing any of it. N^2 is the same in-situ finite-difference bn2 the
scoreboard side-finding used (drho/dz on the band-mean profile, addendum 19),
NOT NEMO's linearized eosbn2 form -- both models get the identical treatment,
which is what the acceptance test needs (a controlled comparison), not a
NEMO-exact bn2 transcription (that lives in production tke.py/_shared.py,
off-limits here).

Run:
  DINO_TW_LEGO_DIR=<dir with year_seamfix*.npz-style outputs> \\
    .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/acc_acceptance_znmax.py \\
    [year1.npz] [year5.npz]

With no args, uses acc_thermal_wind.LEGO_DIR/year_seamfix{,_y5}.npz (its own
default).

INSTRUMENT CAVEAT (self-check run against the frozen year_seamfix*.npz
baseline pair): this script's z(maxN2) does NOT reproduce the recorded 166.6 m
number -- it gives 118.5 m (y1) / 136.3 m (y5) on the identical baseline
files. The N^2 profile is well-resolved and the maximum is broad (levels 9-11
of 36 are all close), so the discrepancy is most likely a different N^2
definition (this script differences the band-mean in-situ density on
gdept_1d; the original 166.6 m side-finding came from an addendum-19
tilt_measure.py/tw_source.py analysis not present in the repo, whose exact N^2
form/grid is unverified) rather than a channel-band or mask difference (both
reused verbatim from acc_thermal_wind.py). Report this metric's before/after
change as PLAUSIBLE, not CONFIRMED against the literal 166.6 m baseline value.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import acc_thermal_wind as tw  # noqa: E402  (module-level state reused, not copied)


def z_of_max_n2(path):
    """Channel-band-mean z(maxN2) for one legoESM annual-mean npz, same mask/
    EOS/depth ladder as acc_thermal_wind.py's contrast_profile/rho_of."""
    st = tw.load_lego(path)
    wet = tw.tmask & (st["land_mask"][:, :, None] > 0.5)
    rho = tw.rho_of(st, wet)
    rc = tw.contrast_profile(rho, wet)  # unused here; forces the same band gate as tw
    a, b = slice(tw.J0, tw.J1), slice(tw.J0 + 1, tw.J1 + 1)
    pair = wet[a] & wet[b]
    _rb = np.where(wet, rho, np.nan)[tw.J0:tw.J1 + 1]
    nk = np.sum(np.isfinite(_rb), axis=(0, 1))
    rbar = np.where(nk > 0, np.nansum(np.nan_to_num(_rb), axis=(0, 1)) / np.maximum(nk, 1), np.nan)
    n2 = -(tw.constants.g / tw.RHO0) * np.gradient(rbar, -tw.gdept1d)  # z up: dz<0 downward
    k = int(np.nanargmax(n2))
    return float(tw.gdept1d[k]), n2, rc


def main():
    y1 = sys.argv[1] if len(sys.argv) > 1 else f"{tw.LEGO_DIR}/year_seamfix.npz"
    y5 = sys.argv[2] if len(sys.argv) > 2 else f"{tw.LEGO_DIR}/year_seamfix_y5.npz"
    print(f"channel band T-rows {tw.J0}..{tw.J1} ({tw.gphit[tw.J0, 25]:.1f}.."
          f"{tw.gphit[tw.J1, 25]:.1f} lat) -- same band as acc_thermal_wind.py\n")
    for lab, p in (("y1", y1), ("y5", y5)):
        z, n2, _ = z_of_max_n2(p)
        print(f"{lab}: z(maxN2) = {z:7.1f} m   (file {p})")
    print("\nNEMO reference (recorded, addendum 19): z(maxN2) 166.6 m both years (target 110-127 m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
