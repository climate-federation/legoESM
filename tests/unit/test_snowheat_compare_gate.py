"""The snow-node A/B bit-identity gate comparator must flag any difference."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "cluster" / "levante"
           / "snowheat_oct" / "compare_gate.py")


def _mod():
    spec = importlib.util.spec_from_file_location("compare_gate", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_identical_and_each_kind_of_difference(tmp_path):
    cmp = _mod().checkpoint_differences
    base = {"T": np.array([1.0, np.nan]), "step": np.array(5), "name": np.array("x")}
    np.savez(tmp_path / "a.npz", **base)
    np.savez(tmp_path / "b.npz", **base)
    assert cmp(tmp_path / "a.npz", tmp_path / "b.npz") == []
    np.savez(tmp_path / "c.npz", **{**base, "T": np.array([1.0 + 1e-15, np.nan])})
    assert cmp(tmp_path / "a.npz", tmp_path / "c.npz") == ["T"]
    np.savez(tmp_path / "d.npz", **{**base, "T": base["T"].astype(np.float32)})
    assert cmp(tmp_path / "a.npz", tmp_path / "d.npz") == ["T"]
    np.savez(tmp_path / "z.npz", **{**base, "T": np.array([-0.0, np.nan])})
    np.savez(tmp_path / "z0.npz", **{**base, "T": np.array([0.0, np.nan])})
    assert cmp(tmp_path / "z0.npz", tmp_path / "z.npz") == ["T"]       # signed zero
    p1 = np.array([np.nan, 0.0])
    p2 = p1.copy()
    p2.view(np.uint64)[0] ^= np.uint64(1)                              # NaN payload
    assert np.isnan(p2[0])
    np.savez(tmp_path / "n1.npz", **{**base, "T": p1})
    np.savez(tmp_path / "n2.npz", **{**base, "T": p2})
    assert cmp(tmp_path / "n1.npz", tmp_path / "n2.npz") == ["T"]
    np.savez(tmp_path / "e.npz", **{k: v for k, v in base.items() if k != "step"})
    assert cmp(tmp_path / "a.npz", tmp_path / "e.npz") == ["step"]


def test_expect_differ_needs_a_shared_entry_to_change(tmp_path):
    import subprocess
    import sys
    m = _mod()
    off = {"T": np.array([1.0, 2.0])}
    np.savez(tmp_path / "off.npz", **off)
    np.savez(tmp_path / "on_new_field_only.npz", **off, T_snow=np.array([3.0]))
    np.savez(tmp_path / "on_acts.npz", T=np.array([1.0, 2.5]), T_snow=np.array([3.0]))
    assert m.shared_differences(tmp_path / "on_new_field_only.npz", tmp_path / "off.npz") == []
    assert m.shared_differences(tmp_path / "on_acts.npz", tmp_path / "off.npz") == ["T"]
    np.savez(tmp_path / "off_l.npz", **off, scheme=np.array("off"), step=np.array(1))
    np.savez(tmp_path / "on_label_only.npz", **off, scheme=np.array("on"), step=np.array(2))
    assert m.shared_differences(tmp_path / "on_label_only.npz", tmp_path / "off_l.npz") == []
    np.savez(tmp_path / "on_type_swap.npz", T=np.array("x"))
    assert m.shared_differences(tmp_path / "off.npz", tmp_path / "on_type_swap.npz") == []

    def rc(*args):
        return subprocess.run([sys.executable, str(_SCRIPT), *map(str, args)],
                              capture_output=True).returncode
    assert rc("--expect-differ", tmp_path / "on_acts.npz", tmp_path / "off.npz") == 0
    assert rc("--expect-differ", tmp_path / "on_new_field_only.npz", tmp_path / "off.npz") == 1
    assert rc("--expect-differ", tmp_path / "missing.npz", tmp_path / "off.npz") == 2
    assert rc(tmp_path / "off.npz", tmp_path / "off.npz") == 0
    assert rc(tmp_path / "missing.npz", tmp_path / "off.npz") == 2
    assert rc("--expect-differ", tmp_path / "off.npz") == 2
    assert rc(tmp_path / "off.npz") == 2
