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
    # baseline + 4 conv + 7 turb + 3 gwd + 5 micro + 1 cloud — every CLI-valid
    # scheme per category, NO 'none' anywhere (all 5 categories always active).
    assert len(combos) == 1 + 4 + 7 + 3 + 5 + 1 == 21
    assert combos[0]["name"] == "combo_baseline"
    assert combos[0]["convection"] == mod.BASELINE["convection"]
    for c in combos:
        assert {"name", "convection", "turbulence", "gravity_wave_drag",
                "microphysics", "clouds"} <= set(c)


def test_no_combo_omits_a_category():
    """User constraint: ALWAYS have all parameterization categories active —
    no combo (incl. baseline) may set any category to 'none'."""
    cats = ("convection", "turbulence", "gravity_wave_drag",
            "microphysics", "clouds")
    for c in mod.build_combos():
        off = [k for k in cats if c[k] == "none"]
        assert not off, f"{c['name']} omits {off} (set to 'none')"


def test_sweep_schemes_are_run_amip_choices():
    """Drift tripwire: every swept scheme (incl. baseline) MUST be a valid
    run_amip CLI choice, else the sweep task dies at argparse (exit 2) and
    wastes a GPU slot — which is exactly how the profile-prognostic
    convection schemes silently failed before."""
    import importlib.util as _u
    rspec = _u.spec_from_file_location(
        "_run_amip_mod", REPO / "scripts" / "run" / "run_amip.py")
    ra = _u.module_from_spec(rspec)
    rspec.loader.exec_module(ra)
    parser = ra.build_arg_parser()
    flag = {"convection": "--convection", "turbulence": "--turbulence",
            "gravity_wave_drag": "--gravity-wave-drag",
            "microphysics": "--microphysics", "clouds": "--clouds"}
    choices = {}
    for action in parser._actions:
        for axis, f in flag.items():
            if f in action.option_strings:
                choices[axis] = set(action.choices or [])
    for axis, alts in mod.SWEEP_SPACE.items():
        valid = choices[axis]
        bad = [s for s in [mod.BASELINE[axis], *alts] if s not in valid]
        assert not bad, f"{axis} schemes {bad} not in run_amip choices {valid}"


def test_oat_changes_exactly_one_axis():
    base = mod.BASELINE
    for c in mod.build_combos()[1:]:
        diffs = [k for k in ("convection", "turbulence", "gravity_wave_drag",
                             "microphysics", "clouds")
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
