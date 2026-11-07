"""strip_physics_state.py: drops exactly the physstate_* entries, keeps the rest bit-for-bit."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_TOOL = (pathlib.Path(__file__).resolve().parents[2]
         / "scripts" / "experiment" / "strip_physics_state.py")
_spec = importlib.util.spec_from_file_location("strip_physics_state", _TOOL)
sps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sps)


def _write(path, **arrays):
    np.savez(path, **arrays)
    return str(path)


def test_strips_only_physstate_keys(tmp_path):
    src = _write(tmp_path / "ckpt.npz", T=np.arange(6.0).reshape(2, 3), p_s=np.ones(2),
                 day=np.array(40.0), physstate_tke=np.zeros((2, 3)),
                 physstate_clubb_moments=np.zeros((2, 1, 1)),
                 physstate_meta_conv_scheme=np.array("bechtold"))
    dst = str(tmp_path / "out.npz")
    dropped = sps.strip_physics_state(src, dst)
    assert dropped == ["physstate_clubb_moments", "physstate_meta_conv_scheme",
                       "physstate_tke"]
    with np.load(dst, allow_pickle=True) as z:
        assert sorted(z.files) == ["T", "day", "p_s"]
        assert np.array_equal(z["T"], np.arange(6.0).reshape(2, 3))
        assert float(z["day"]) == 40.0


def test_refuses_a_checkpoint_with_nothing_to_strip(tmp_path):
    src = _write(tmp_path / "plain.npz", T=np.zeros((2, 3)))
    with pytest.raises(SystemExit):
        sps.strip_physics_state(src, str(tmp_path / "out.npz"))


def test_cli_round_trip(tmp_path, capsys):
    src = _write(tmp_path / "ckpt.npz", T=np.zeros((2, 3)), physstate_tke=np.ones((2, 3)))
    dst = str(tmp_path / "out.npz")
    assert sps.main([src, dst]) == 0
    assert "dropped 1 physics-state entries" in capsys.readouterr().out
    with np.load(dst) as z:
        assert z.files == ["T"]


def test_refuses_in_place(tmp_path):
    src = _write(tmp_path / "ckpt.npz", T=np.zeros((2, 3)), physstate_tke=np.ones((2, 3)))
    with pytest.raises(SystemExit):
        sps.strip_physics_state(src, src)
    with np.load(src) as z:
        assert sorted(z.files) == ["T", "physstate_tke"]   # untouched


def test_no_temporary_file_left_behind(tmp_path):
    src = _write(tmp_path / "ckpt.npz", T=np.zeros((2, 3)), physstate_tke=np.ones((2, 3)))
    sps.strip_physics_state(src, str(tmp_path / "out.npz"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt.npz", "out.npz"]
