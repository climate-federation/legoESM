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


# --- user decision 2026-10-14 (4a): lanes that cannot apply nemo_scrip refuse it.


@pytest.mark.parametrize("remap,refused", [("nemo_scrip", True),
                                           ("bilinear", False)])
def test_fesom_lane_refuses_nemo_scrip_at_setup(remap, refused):
    """The fesom forced loop never passes forcing_remap to the sampler, so it
    would run bilinear whatever the flag said; the stage gate must refuse it."""
    from scripts.run.run_omip_core2 import (
        _build_arg_parser, validate_fesom_stage)
    p = _build_arg_parser()
    args = p.parse_args(["--grid", "fesom", "--fesom-mesh-dir", "/m",
                         "--output", "/o", "--no-emp",
                         "--forcing-remap", remap])
    if refused:
        with pytest.raises(SystemExit, match="--forcing-remap"):
            validate_fesom_stage(args, p)
    else:
        validate_fesom_stage(args, p)  # no raise


def _scan_remap_guard():
    """The `if args.forcing_remap ...` statement inside main()'s `if use_scan:`
    branch, and whether it precedes the scan block builder call."""
    import ast
    import textwrap

    from scripts.run import run_omip_core2 as c2
    tree = ast.parse(textwrap.dedent(inspect.getsource(c2.main)))
    scan = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and isinstance(n.test, ast.Name) and n.test.id == "use_scan")
    body_src = [ast.unparse(s) for s in scan.body]
    guard = next(i for i, s in enumerate(scan.body)
                 if isinstance(s, ast.If) and "forcing_remap" in body_src[i])
    build = next(i for i, s in enumerate(body_src)
                 if "build_omip2_scan_block_fn(model" in s)
    # Reachable: no unconditional exit between the branch entry and the guard.
    assert not any(isinstance(s, (ast.Return, ast.Raise))
                   for s in scan.body[:guard])
    return scan.body[guard], guard < build


@pytest.mark.parametrize("remap,refused", [("nemo_scrip", True),
                                           ("bilinear", False)])
def test_scan_block_lane_refuses_nemo_scrip(remap, refused):
    """The scan lane forces through a bilinear-only device stack and returns
    before the host loop reads --forcing-remap; the guard must refuse a
    non-bilinear remap before the block builder runs."""
    import ast
    import types
    node, before_build = _scan_remap_guard()
    assert before_build, "guard must run before the scan block builder"
    code = compile(ast.Module(body=[node], type_ignores=[]), "<guard>", "exec")
    env = {"args": types.SimpleNamespace(forcing_remap=remap)}
    if refused:
        with pytest.raises(SystemExit, match="--forcing-remap nemo_scrip"):
            exec(code, env)
    else:
        exec(code, env)  # no raise


# --- user decisions 2026-10-15 (8, 6): every non-tripole grid refuses
# nemo_scrip at setup, before the model build.


class _PastSetupGuards(Exception):
    pass


@pytest.mark.parametrize("grid,remap,refused", [
    ("mpas", "nemo_scrip", True),
    ("latlon_bathy", "nemo_scrip", True),
    ("cubed_sphere", "nemo_scrip", True),
    ("mpas", "bilinear", False),
    ("latlon_bathy", "bilinear", False),
    ("cubed_sphere", "bilinear", False),
    ("tripole", "nemo_scrip", False),
])
def test_non_tripole_grids_refuse_nemo_scrip_before_the_model_build(
        monkeypatch, grid, remap, refused):
    """Without the guard a non-tripole grid builds the model and only then dies
    in the shared sampler at the first forcing step. ``_validate_spmd_persistent_state``
    runs after the setup guards and before any bootstrap or build, so reaching
    it means the combination got past setup."""
    import sys

    from scripts.run import run_omip_core2 as c2

    def _stop(*a, **k):
        raise _PastSetupGuards

    monkeypatch.setattr(c2, "_validate_spmd_persistent_state", _stop)
    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", "--grid", grid,
                                      "--forcing-remap", remap,
                                      "--output", "/o", "--no-emp"])
    if refused:
        with pytest.raises(SystemExit, match="--forcing-remap nemo_scrip"):
            c2.main()
    else:
        with pytest.raises(_PastSetupGuards):
            c2.main()
