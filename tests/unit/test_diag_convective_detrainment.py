"""Direct test for scripts/validate/diag_convective_detrainment.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "diag_convective_detrainment.py")


def _load():
    spec = importlib.util.spec_from_file_location("diag_detrain", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flags():
    m = _load()
    a = m.build_arg_parser().parse_args(["--config", "d.yaml", "--restart", "c.npz"])
    assert a.label == "arm"
    with pytest.raises(SystemExit):
        m.build_arg_parser().parse_args(["--config", "d.yaml"])


def test_column_integral_is_area_weighted_and_mass_weighted():
    m = _load()
    rate = np.array([[1.0, 2.0], [4.0, 0.0]])
    dp = np.array([[100.0, 100.0], [100.0, 100.0]])
    w = np.array([0.25, 0.75])
    got = m.column_integral(rate, dp, w, 10.0)
    # (0.25*(1+2) + 0.75*(4+0)) * 100 / 10
    assert abs(got - 37.5) < 1e-12


def test_column_integral_zero_rate_is_zero():
    m = _load()
    assert m.column_integral(np.zeros((3, 4)), np.ones((3, 4)),
                             np.full(3, 1 / 3), 9.8) == 0.0


def test_phys_state_from_checkpoint_carries_the_runs_pblh():
    """Follow-up row 23: ZM refuses a call without the PBL height, so the
    diagnostic builds the convection's physics state from the checkpoint's
    staged carry (its pblh), and refuses a checkpoint without one."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    m = _load()
    pblh = np.array([350.0, 1200.0, 0.0])
    cf = np.full((3, 4), 0.25)
    ps = m.phys_state_from_checkpoint(
        {"physstate_pblh": pblh, "physstate_cloud_fraction": cf, "other": 1},
        3, 4, PhysicsConfig())
    np.testing.assert_array_equal(np.asarray(ps.pblh), pblh)
    np.testing.assert_array_equal(np.asarray(ps.cloud_fraction), cf)
    with pytest.raises(SystemExit, match="pblh"):
        m.phys_state_from_checkpoint({}, 3, 4, PhysicsConfig())
