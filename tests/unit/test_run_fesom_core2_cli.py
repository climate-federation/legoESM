"""Direct tests for scripts/run/run_fesom_core2.py helpers (CLI + snapshot
writer); the model integration itself is exercised by the SLURM arm."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "run_fesom_core2", _ROOT / "scripts/run/run_fesom_core2.py")
m = importlib.util.module_from_spec(_SPEC)
sys.modules["run_fesom_core2"] = m
_SPEC.loader.exec_module(m)


def test_arg_parser_round_trip():
    p = m.build_arg_parser()
    a = p.parse_args(["--mesh-dir", "M", "--ic-dir", "I", "--output", "O",
                      "--days", "30", "--dt", "1800", "--year", "1958"])
    assert (a.mesh_dir, a.ic_dir, a.output) == ("M", "I", "O")
    assert a.days == 30.0 and a.dt == 1800.0 and a.year == 1958


def test_snapshot_writer_conventions(tmp_path):
    """Writer must emit the comparator's node-cloud conventions: (n, nreal)
    tracers (padded bottom slot sliced off), degrees in [-180, 180],
    land_mask 1.0 == OCEAN, positive-down depths.  The state contract is
    NODE-first (nod2D, nl) with nl = nreal + 1 padded slots; the writer
    REFUSES level-first input rather than heuristically transposing (codex
    FESOM-arm hardening) -- the old 'tolerate level-first' expectation was
    stale against that guard."""
    n, nreal = 7, 3
    nl = nreal + 1                                       # +1 padded bottom slot
    mesh = types.SimpleNamespace(
        nod2D=n,
        geo_coord_nod2D=np.stack(
            [np.deg2rad(np.linspace(10.0, 350.0, n)),    # lon rad, wraps >180
             np.deg2rad(np.linspace(-60.0, 60.0, n))], axis=1),
        node_layer_mask=np.ones((n, nl), dtype=bool),
        Z=-np.array([5.0, 15.0, 30.0]),                  # nreal midpoints
        depth=-np.full(n, 100.0),
    )
    state = types.SimpleNamespace(
        T=np.arange(n * nl, dtype=np.float64).reshape(n, nl),  # node-first
        S=np.full((n, nl), 35.0),
        eta_n=np.zeros(n), a_ice=np.zeros(n), m_ice=np.zeros(n),
    )
    p = m.write_snapshot(tmp_path, "day0001", state, mesh)
    z = np.load(p)
    assert z["T"].shape == (n, nreal)                    # padded slot sliced
    assert z["T"][0, 1] == state.T[0, 1]                 # passthrough layout
    assert z["lon_T"].min() >= -180.0 and z["lon_T"].max() <= 180.0
    assert z["land_mask"].min() == 1.0                   # 1.0 == OCEAN
    assert (z["z_center_ref"] > 0).all() and (z["H_bathy"] > 0).all()


def test_snapshot_writer_refuses_level_first(tmp_path):
    """The guard that replaced the transpose heuristic must actually fire."""
    import pytest
    n, nl = 7, 4
    mesh = types.SimpleNamespace(
        nod2D=n,
        geo_coord_nod2D=np.zeros((n, 2)),
        node_layer_mask=np.ones((n, nl), dtype=bool),
        Z=-np.array([5.0, 15.0, 30.0]),
        depth=-np.full(n, 100.0),
    )
    state = types.SimpleNamespace(
        T=np.zeros((nl, n)), S=np.zeros((nl, n)),        # level-first: refuse
        eta_n=np.zeros(n), a_ice=np.zeros(n), m_ice=np.zeros(n),
    )
    with pytest.raises(SystemExit):
        m.write_snapshot(tmp_path, "day0001", state, mesh)


def test_forcing_flag_round_trip_and_default():
    p = m.build_arg_parser()
    base = ["--mesh-dir", "M", "--ic-dir", "I", "--output", "O"]
    assert p.parse_args(base).forcing == "jra55"
    a = p.parse_args(base + ["--forcing", "core2_nyf", "--nyf-zarr", "/x/nyf.zarr"])
    assert a.forcing == "core2_nyf" and a.nyf_zarr == "/x/nyf.zarr"


def test_unknown_forcing_rejected():
    import pytest
    p = m.build_arg_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--mesh-dir", "M", "--ic-dir", "I", "--output", "O",
                      "--forcing", "era5"])


def test_tke_surface_bc_flag():
    import pytest
    p = m.build_arg_parser()
    base = ["--mesh-dir", "M", "--ic-dir", "I", "--output", "O"]
    assert p.parse_args(base).tke_surface_bc == "neumann"
    assert p.parse_args(base + ["--tke-surface-bc", "dirichlet"]
                        ).tke_surface_bc == "dirichlet"
    with pytest.raises(SystemExit):
        p.parse_args(base + ["--tke-surface-bc", "robin"])
