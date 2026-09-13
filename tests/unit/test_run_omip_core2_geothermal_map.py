"""NEMO reads a geothermal MAP; we applied one number everywhere.

ORCA1 runs ln_trabbc=.true. with nn_geoflx=2, which reads
geothermal_heat_flux.nc through its own bilinear weights. The field spans
12.45 to 2287 mW/m2 -- a 180-fold contrast concentrated on the mid-ocean
ridges. Our driver could only apply a single constant, so none of that
structure reached the abyss.

The apply helper already accepted a spatially-varying flux; what was missing
was reading the file, remapping it and handing it over. These tests pin that
path, the unit conversion, and the guards. They do not assert what the map
does to the ocean -- that is a measurement.
"""
from __future__ import annotations

import pathlib

import pytest

RUNNER = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "run" / "run_omip_core2.py")


def _parser():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_omip_core2_geo", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._build_arg_parser()


def test_flag_exists_and_is_off_by_default():
    p = _parser()
    assert p.parse_args([]).geothermal_map is False
    assert p.parse_args(["--geothermal-map"]).geothermal_map is True


def test_units_are_converted_not_assumed():
    """Codex mutation-tested the first version of this: it asserted the
    constant was declared and the name appeared in the loader, so replacing
    the multiplication with a DIVISION still passed while producing a flux
    1e6 times too large. So this now RUNS the loader and checks the physical
    magnitude.

    The seafloor geothermal flux is textbook 0.05-0.10 W/m2 in the global
    mean, and the source file's own mean is 79.1 mW/m2 = 0.0791 W/m2. Any
    power-of-ten error leaves this window immediately.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_omip_core2_geo3", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.load_nemo_geothermal_flux("tripole", (332, 362))
    mean = float(out[out > 0].mean())
    assert 0.02 < mean < 0.30, f"geothermal mean {mean} W/m2 is not physical"
    assert float(out.max()) < 10.0, (
        f"peak {float(out.max())} W/m2 exceeds any ridge value")


def test_the_units_attribute_is_read_not_assumed():
    """The finite/positive guard is invariant under a factor of 1000, so it
    can never catch a file that ships in W/m2. Reading the attribute can."""
    src = RUNNER.read_text()
    i = src.index("def load_nemo_geothermal_flux(")
    body = src[i:i + 3000]
    assert 'attrs.get("units"' in body and "are not mW/m^2" in body


def test_the_measured_variable_name_is_used():
    """Probed from the file itself (job 9748059): gh_flux(1,180,360),
    units 'mW m^{-2}'. Guessing this name was how three other names went
    wrong today."""
    assert '"gh_flux"' in RUNNER.read_text()


def test_the_map_reaches_the_apply_helper():
    """A field loaded but not passed is the defect class this series keeps
    finding: it looks wired and changes nothing."""
    assert "flux_wm2=_geo_map)" in RUNNER.read_text()


def test_conflicting_flux_sources_are_refused():
    """A constant and a map both set the same quantity; one would silently
    win."""
    src = RUNNER.read_text()
    assert "both set the " in src and "Pick one." in src


def test_the_map_requires_the_boundary_condition_itself():
    assert "--geothermal-map needs --geothermal" in RUNNER.read_text()


def test_negative_or_nonfinite_map_is_refused():
    """apply_geothermal_step range-checks only SCALARS -- a bad array would
    pass straight through and cool the abyss."""
    src = RUNNER.read_text()
    i = src.index("def load_nemo_geothermal_flux(")
    body = src[i:i + 3000]
    assert "non-finite or negative" in body


def test_the_apply_helper_still_takes_a_field():
    """The whole approach rests on this. If flux_wm2 ever became scalar-only,
    the map would be silently ignored or would raise far from here."""
    import inspect

    from legoesm.ocean.coupler.geothermal_apply import apply_geothermal_step
    assert "flux_wm2" in inspect.signature(apply_geothermal_step).parameters
    src = inspect.getsource(apply_geothermal_step)
    assert "cfg.flux_wm2 if flux_wm2 is None else flux_wm2" in src


@pytest.mark.parametrize("path_const", ["_GHFLUX_NC", "_GHFLUX_SCRIP_WEIGHTS"])
def test_the_input_files_exist(path_const):
    """Both live in the oracle's own INPUTS tree; a moved file should fail
    here rather than at hour six of a production run."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_omip_core2_geo2", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert pathlib.Path(getattr(mod, path_const)).is_file()
