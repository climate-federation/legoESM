"""The wind SCRIP remap existed but nothing could turn it on.

The SCRIP machinery (load_scrip_weights / apply_scrip_weights /
scrip_interior_to_full_tripole) and a ``forcing_remap`` argument on
``_sample_omip2_forcing`` were already in place, but
``compute_omip2_surface_forcing`` -- the function the production loop actually
calls -- did not take or forward the option, and the driver had no flag. So
every run interpolated the winds with our own bilinear scheme while NEMO used
its own weight files (bicubic for u10/v10), and no command line said so.

These tests pin the plumbing only. Whether NEMO's weights change the answer is
a measurement, not something a unit test decides.
"""
from __future__ import annotations

import inspect
import pathlib

import pytest

RUNNER = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "run" / "run_omip_core2.py")


def _parser():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_omip_core2_fr", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._build_arg_parser()


def test_flag_exists_and_defaults_to_the_unchanged_path():
    p = _parser()
    assert p.parse_args([]).forcing_remap == "bilinear"
    assert p.parse_args(["--forcing-remap", "nemo_scrip"]).forcing_remap == \
        "nemo_scrip"


def test_a_typo_is_refused_rather_than_silently_bilinear():
    with pytest.raises(SystemExit):
        _parser().parse_args(["--forcing-remap", "scrip"])


def test_the_production_entry_point_accepts_the_option():
    """This is the defect: the sampler took forcing_remap, the function the
    run loop calls did not."""
    from legoesm.ocean.coupler.omip2_applicator import (
        compute_omip2_surface_forcing)
    sig = inspect.signature(compute_omip2_surface_forcing)
    assert "forcing_remap" in sig.parameters
    assert sig.parameters["forcing_remap"].default == "bilinear", (
        "default must preserve the existing interpolation")


def test_the_entry_point_forwards_it_to_the_sampler():
    """Accepting the argument and dropping it would look wired and do nothing
    -- the same shape as the defect being fixed."""
    from legoesm.ocean.coupler import omip2_applicator as m
    src = inspect.getsource(m.compute_omip2_surface_forcing)
    assert "forcing_remap=forcing_remap" in src


def test_the_driver_passes_the_flag_at_the_call_site():
    assert "forcing_remap=args.forcing_remap" in RUNNER.read_text()


def test_the_sampler_still_refuses_a_non_tripole_grid():
    """The weights encode one destination grid. If this guard went away the
    option would silently interpolate onto the wrong mesh."""
    from legoesm.ocean.coupler import omip2_applicator as m
    src = inspect.getsource(m._sample_omip2_forcing)
    assert 'forcing_remap == "nemo_scrip" and grid_type != "tripole"' in src
