"""Rewrite CloudBench artifacts with the runs' free-troposphere relaxation.

The artifacts built before 2026-11-18 record swirl-lm's public relaxation defaults
(z_i 3.0 km, z_r 3.5 km, tau 1 day); the runs used z_i 4 km, a 2 km ramp and tau
6 h (``cloudbench.NUDGE_RECIPE``). Every artifact in SRC is copied to DST with only
``nudge_rate`` and the ``forcing_recipe`` relaxation keys replaced; every other array
is copied byte for byte from the source npz (checked after writing), and SRC is never
written. Files in SRC that are not artifacts (held-out list, failure log) are copied
unchanged.

usage: migrate_cloudbench_nudging.py SRC_DIR DST_DIR
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

from legoesm.atmosphere.les_suite import cloudbench as cb

#: the relaxation the old artifacts record (public swirl-lm defaults)
OLD_RECIPE = {"nudge_z_i_m": 3000.0, "nudge_z_r_m": 3500.0, "tau_scalar_s": 86400.0}


def _old_rate(z):
    zi, zr, tau = OLD_RECIPE["nudge_z_i_m"], OLD_RECIPE["nudge_z_r_m"], OLD_RECIPE["tau_scalar_s"]
    ramp = 0.5 * (1.0 - np.cos(np.pi * (z - zi) / (zr - zi)))
    return np.where(z < zi, 0.0, np.where(z <= zr, ramp, 1.0)) / tau


def migrate_npz(src: Path, dst: Path) -> dict:
    """Write ``dst`` from ``src`` with the runs' relaxation. Returns a record."""
    with np.load(src, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    meta = json.loads(bytes(arrays["__meta__"]).decode("utf-8"))
    setup = meta["les_setup"]
    recipe = setup["forcing_recipe"]
    for k, v in OLD_RECIPE.items():
        if recipe.get(k) != v:
            raise ValueError(f"{src.name}: recipe {k}={recipe.get(k)!r}, expected the old {v!r}")
    z_node = arrays["heights_m"] - float(setup["height_offset_m"])
    # z_node = heights - offset may differ from the builder's node coordinate in the
    # last bit, so the stored rate is matched to 1e-12 of 1/day, not bit for bit
    if not np.allclose(arrays["nudge_rate"], _old_rate(z_node), rtol=0.0, atol=1e-12 / 86400.0):
        raise ValueError(f"{src.name}: stored nudge_rate is not the old recipe's")
    new = dict(arrays)
    new["nudge_rate"] = cb.nudge_rate(z_node).astype(arrays["nudge_rate"].dtype)
    meta["les_setup"] = {**setup, "forcing_recipe": {**recipe, **cb.NUDGE_RECIPE},
                         "nudging_migrated_from": str(src)}
    new["__meta__"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(dst, **new)
    with np.load(dst, allow_pickle=False) as back:
        if set(back.files) != set(arrays):
            raise RuntimeError(f"{dst.name}: field set changed")
        for k in arrays:
            if k in ("nudge_rate", "__meta__"):
                continue
            if back[k].dtype != arrays[k].dtype or back[k].tobytes() != arrays[k].tobytes():
                raise RuntimeError(f"{dst.name}: {k} not byte-identical")
    return {"case": src.name, "max_rate_change_per_day":
            float(86400 * np.max(np.abs(new["nudge_rate"] - arrays["nudge_rate"])))}


def main(argv=None) -> int:
    src, dst = (Path(a) for a in (argv or sys.argv[1:]))
    if src.resolve() == dst.resolve():
        raise SystemExit("DST must differ from SRC")
    if dst.exists() and (dst.is_file() or any(dst.iterdir())):
        raise SystemExit(f"DST {dst} is not empty: refusing to mix runs")
    dst.mkdir(parents=True, exist_ok=True)
    recs, skipped = [], []
    for p in sorted(src.iterdir()):
        if not p.is_file():                  # e.g. the builder's scratch directory
            skipped.append(p.name)
        elif p.name.startswith("cloudbench_") and p.suffix == ".npz":
            recs.append(migrate_npz(p, dst / p.name))
        else:
            shutil.copy2(p, dst / p.name)
    (dst / "nudging_migration.json").write_text(json.dumps(
        {"source_dir": str(src), "new_recipe": cb.NUDGE_RECIPE, "old_recipe": OLD_RECIPE,
         "n": len(recs), "skipped_non_files": skipped, "cases": recs}, indent=1))
    print(f"migrated {len(recs)} artifacts to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
