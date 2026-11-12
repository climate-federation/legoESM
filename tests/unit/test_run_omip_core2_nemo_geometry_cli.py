"""--nemo-geometry: the tripole card's 3-D cells from NEMO's domain_cfg.

Pins the plumbing: the flag defaults to today's mesh_mask geometry, main()
refuses domain_cfg without its prerequisites before any model build, and the
choice reaches build_tripole.  The geometry itself is checked against the
real ORCA1 files by scripts/validate/ocean_fidelity/
omip_tripole_geometry_vs_domain_cfg.py.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

RUNNER = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "run" / "run_omip_core2.py")


def _core2():
    import scripts.run.run_omip_core2 as core2
    return core2


def test_flag_defaults_to_mesh_mask():
    p = _core2()._build_arg_parser()
    assert p.parse_args([]).nemo_geometry == "mesh_mask"
    assert p.parse_args(["--nemo-geometry", "domain_cfg"]).nemo_geometry == "domain_cfg"


@pytest.mark.parametrize("extra, missing", [
    ([], "--partial-cell"),
    (["--partial-cell", "--nemo-vertical", "--nemo-domain-mask",
      "--bathy-smoothing-passes", "2"], "--bathy-smoothing-passes 0"),
    (["--partial-cell", "--nemo-vertical", "--nemo-domain-mask",
      "--nemo-vertical-file", "/x/other.nc"], "--nemo-vertical-file"),
])
def test_domain_cfg_refused_without_prerequisites(monkeypatch, extra, missing):
    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", "--grid", "tripole",
                                      "--nemo-geometry", "domain_cfg", *extra])
    with pytest.raises(ValueError, match=f"needs .*{missing}"):
        _core2().main()


def test_choice_is_forwarded_to_build_tripole():
    src = RUNNER.read_text()
    assert src.count("nemo_geometry=args.nemo_geometry,") == 1
    assert 'nemo_geometry="mesh_mask",' in src
    assert 'if nemo_geometry == "domain_cfg":' in src
