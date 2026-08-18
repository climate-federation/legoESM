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
    """Writer must emit the comparator's node-cloud conventions: (n, nlev)
    tracers, degrees in [-180, 180], land_mask 1.0 == OCEAN, positive-down
    depths -- and tolerate fesom-jax's level-first state orientation."""
    n, nlev = 7, 4
    mesh = types.SimpleNamespace(
        nod2D=n,
        geo_coord_nod2D=np.stack(
            [np.deg2rad(np.linspace(10.0, 350.0, n)),    # lon rad, wraps >180
             np.deg2rad(np.linspace(-60.0, 60.0, n))], axis=1),
        node_layer_mask=np.ones((n, nlev), dtype=bool),
        Z=-np.array([5.0, 15.0, 30.0, 60.0]),            # negative-down inside
        depth=-np.full(n, 100.0),
    )
    state = types.SimpleNamespace(
        T=np.arange(nlev * n, dtype=np.float64).reshape(nlev, n),  # level-first
        S=np.full((nlev, n), 35.0),
        eta_n=np.zeros(n), a_ice=np.zeros(n), m_ice=np.zeros(n),
    )
    p = m.write_snapshot(tmp_path, "day0001", state, mesh)
    z = np.load(p)
    assert z["T"].shape == (n, nlev)                     # node-first out
    assert z["T"][0, 1] == state.T[1, 0]                 # transpose, not reshape
    assert z["lon_T"].min() >= -180.0 and z["lon_T"].max() <= 180.0
    assert z["land_mask"].min() == 1.0                   # 1.0 == OCEAN
    assert (z["z_center_ref"] > 0).all() and (z["H_bathy"] > 0).all()
