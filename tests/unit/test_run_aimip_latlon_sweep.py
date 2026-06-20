"""Unit test for the AIMIP lat-lon OAT physics sweep generator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_aimip_latlon_sweep_mod", REPO / "scripts" / "run" / "run_aimip_latlon_sweep.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_combo_count_and_structure():
    combos = mod.build_combos()
    # baseline + 9 conv + 4 turb (tke/mynn25 excluded) + 4 gwd
    assert len(combos) == 1 + 9 + 4 + 4 == 18
    assert combos[0]["name"] == "combo_baseline"
    assert combos[0]["convection"] == mod.BASELINE["convection"]
    for c in combos:
        assert {"name", "convection", "turbulence", "gravity_wave_drag"} <= set(c)


def test_oat_changes_exactly_one_axis():
    base = mod.BASELINE
    for c in mod.build_combos()[1:]:
        diffs = [k for k in ("convection", "turbulence", "gravity_wave_drag")
                 if c[k] != base[k]]
        assert len(diffs) == 1, f"{c['name']} changes {diffs}, expected exactly 1"


def test_index_out_of_range_raises():
    try:
        mod.run_combo(999, "results/x", [])
    except SystemExit:
        return
    raise AssertionError("expected SystemExit for out-of-range combo index")


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("run_aimip_latlon_sweep: all self-checks passed")


if __name__ == "__main__":
    _run_all()
