#!/usr/bin/env python
"""#1226 tier-2 item 1: ldftra ahtu/ahtv vs NEMO's own dump.

Reads the live NEMO ``kt==nit000`` dump of ``ahtu``/``ahtv``/``gphiu``/``gphiv``
(``MY_SRC/ldftra.F90``, units 8960-8963) from a NEMO DINO run directory and
compares against legoESM's ``static_kappa_redi_override`` (T-point field, feeds
the u-face flux unchanged) and its v-face companion field (the tier-2 item-1
fix: NEMO's ``ahtv`` is evaluated independently at the v-point, NOT ``ahtu``
broadcast onto the v-face).

Run (after a DINO NEMO build with the #1226 ldftra dump instrumentation has
produced ``ldftra_dump_{ahtu,ahtv,gphiu,gphiv}.bin`` in RUN_DIR)::

    .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/ldftra_ahtv_compare.py \
        --run-dir /path/to/RUN_1226_AHTU
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

# dump_lane: shared #1455 selector -- this probe had NO hardcoded run-dir
# constant (a REQUIRED --run-dir, no default), so it was never lane-
# switchable at all. ldftra_dump_{ahtu,ahtv,gphiu,gphiv}.bin exist on every
# dump_lane lane (verified: RUN_GDB, RUN_SEQDUMP_D180_1R, RUN_SEQDUMP_Y20_1R
# all have them), so --run-dir now DEFAULTS to dump_lane.RUN_DIR (gdb_y5 by
# default) while staying overridable to any explicit path (e.g. the original
# RUN_1226_AHTU one-off instrumentation run this script's docstring names).
_dl_path = os.path.join(os.path.dirname(__file__), "dump_lane.py")
_dl_spec = importlib.util.spec_from_file_location("_dump_lane", _dl_path)
dump_lane = importlib.util.module_from_spec(_dl_spec)
sys.modules["_dump_lane"] = dump_lane
_dl_spec.loader.exec_module(dump_lane)

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    static_kappa_redi_override,
)
from legoesm.ocean.experiments.dino import (  # noqa: E402
    dino_lat_lon_grid, nemo_faithful_dino_config,
)
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig  # noqa: E402


def _load3(path: str, jpi: int, jpj: int, jpk: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f8").reshape(jpk, jpj, jpi)


def _read_dims(run_dir: str) -> tuple[int, int, int]:
    with open(os.path.join(run_dir, "ocean.output")) as f:
        text = f.read()
    import re
    jpi = int(re.search(r"jpi\s*:\s*(\d+)", text).group(1))
    jpj = int(re.search(r"jpj\s*:\s*(\d+)", text).group(1))
    jpk = int(re.search(r"jpk\s*:\s*(\d+)", text).group(1))
    return jpi, jpj, jpk


def _report(name: str, lego: np.ndarray, nemo: np.ndarray) -> tuple[float, float]:
    m = nemo > 0
    ratio = lego[m] / nemo[m]
    corr = float(np.corrcoef(lego[m], nemo[m])[0, 1])
    ratio_mean = float(ratio.mean())
    print(f"{name}: corr={corr:.10f}  ratio_mean={ratio_mean:.10f}  "
          f"ratio_median={float(np.median(ratio)):.10f}  n={int(m.sum())}")
    return corr, ratio_mean


def main() -> int:
    print(dump_lane.banner())
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=dump_lane.RUN_DIR,
                     help="NEMO DINO run directory containing ldftra_dump_*.bin "
                          "and ocean.output (default: dump_lane's RUN_DIR for "
                          "the selected DINO_1226_LANE)")
    args = ap.parse_args()
    print(f"run-dir used = {args.run_dir}")

    jpi, jpj, jpk = _read_dims(args.run_dir)
    ahtu = _load3(os.path.join(args.run_dir, "ldftra_dump_ahtu.bin"), jpi, jpj, jpk)[0]
    ahtv = _load3(os.path.join(args.run_dir, "ldftra_dump_ahtv.bin"), jpi, jpj, jpk)[0]
    gphiu = np.fromfile(
        os.path.join(args.run_dir, "ldftra_dump_gphiu.bin"), dtype="<f8"
    ).reshape(jpj, jpi)

    cfg = nemo_faithful_dino_config()
    grid = dino_lat_lon_grid(cfg)
    gm = GMRediConfig(kappa_Redi=1501.1854665553683, kappa_redi_lat_scaling=True)
    kappa_T, kappa_v = static_kappa_redi_override(gm, grid)
    kappa_T = np.asarray(kappa_T)
    kappa_v = np.asarray(kappa_v)
    n_lat, n_lon = kappa_T.shape

    # Locate legoESM's interior block inside NEMO's halo-padded (jpi, jpj)
    # domain by matching gphiu's row latitudes (constant along a row) to
    # legoESM's grid.lat; the equivalent column search is unnecessary since
    # gphiu/gphiv/ahtu/ahtv have no longitude dependence on this regular grid.
    lego_lat_deg = np.degrees(np.asarray(grid.lat))
    row_col = gphiu[:, jpi // 2]
    jrow_start = None
    for j0 in range(jpj - n_lat):
        if np.max(np.abs(row_col[j0:j0 + n_lat] - lego_lat_deg)) < 1e-4:
            jrow_start = j0
            break
    if jrow_start is None:
        raise RuntimeError("could not align NEMO dump rows to legoESM grid.lat")
    ahtu_row = ahtu[jrow_start, :]
    nz_cols = np.nonzero(ahtu_row)[0]
    icol_start = int(nz_cols[1])  # skip the first (periodic-wrap halo) column

    ahtu_int = ahtu[jrow_start:jrow_start + n_lat, icol_start:icol_start + n_lon]
    ahtv_int = ahtv[jrow_start:jrow_start + n_lat, icol_start:icol_start + n_lon]

    print(f"aligned interior block: rows [{jrow_start}:{jrow_start + n_lat}), "
          f"cols [{icol_start}:{icol_start + n_lon})")
    _report("U-FACE (kappa_T vs NEMO ahtu)", kappa_T, ahtu_int)
    _report("V-FACE PRE-FIX (kappa_T reused vs NEMO ahtv)", kappa_T, ahtv_int)
    corr_v, ratio_v = _report("V-FACE POST-FIX (kappa_v vs NEMO ahtv)", kappa_v, ahtv_int)

    print()
    print(f"gate numbers -> ldftra ahtv: corr={corr_v:.7f} ratio={ratio_v:.7f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
