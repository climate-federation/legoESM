"""CLI/arg-contract tests for
scripts/validate/validate_driver_tiled_dycore_parity.py (P4 increment 1c).

The end-to-end 24-virtual-device parity itself runs in the sbatch lane;
here we pin importability, the config contract (tiled-legal: cubed-sphere,
diffusion scales zeroed to stay inside the tiled base-cut envelope,
diag/checkpoint off, n_devices=24), the tolerance parser, and the argument
guards so a broken invocation fails loudly before burning batch walltime.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate"
           / "validate_driver_tiled_dycore_parity.py")


_LOAD_COUNT = [0]


def _load():
    # Unique module name per load; never left behind in sys.modules.
    _LOAD_COUNT[0] += 1
    name = f"_validate_tiled_dycore_parity_{_LOAD_COUNT[0]}"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def test_build_config_is_tiled_legal(tmp_path):
    """The shared config must pass validate_strict BOTH untiled and with
    enable_tiled_dycore=True (the exact replace the tiled lane performs),
    and must sit inside the tiled base-cut envelope: the factory-computed
    damping (hyperdiff/div-damp/A_h) is zeroed, so make_tiled_cc_step's
    coefficient refusals cannot fire."""
    mod = _load()
    cfg = mod._build_config(str(tmp_path))
    cfg.validate_strict()
    cfg._replace(enable_tiled_dycore=True).validate_strict()
    assert cfg.grid.grid_type == "cubed_sphere"
    assert cfg.dycore.hyperdiff_scale == 0.0
    assert cfg.dycore.div_damp_scale == 0.0
    assert cfg.dycore.a_h_scale == 0.0
    assert cfg.n_devices == mod.N_DEVICES == 24
    assert cfg.output.diag_days == 0
    assert cfg.output.checkpoint_days == 0
    # C8 with kt=2 -> nl=4 per-tile edge (the stage needs nl >= 2).
    assert cfg.grid.resolution % 2 == 0


def test_untiled_mode_requires_out(tmp_path):
    mod = _load()
    rc = mod.main(["--mode", "untiled", "--workdir", str(tmp_path)])
    assert rc == 2


def test_tiled_mode_requires_ref(tmp_path):
    mod = _load()
    rc = mod.main(["--mode", "tiled", "--workdir", str(tmp_path)])
    assert rc == 2


def test_bench_bypasses_ref_out_contract(tmp_path, capsys):
    """--bench is timing-only: neither --out (untiled) nor --ref (tiled)
    is required.  The run proceeds past the argument contract (and then
    fails later on the 24-device requirement in this test env — rc 2
    with the device message, never the missing-arg message)."""
    import os
    mod = _load()
    # Force the device guard so the test NEVER proceeds into a real
    # driver build, even in an environment exposing >= 24 devices
    # (codex round-17 Medium — environment-dependent unit test).
    mod.N_DEVICES = 10**6
    try:
        for mode in ("untiled", "tiled"):
            rc = mod.main(["--mode", mode, "--bench",
                           "--workdir", str(tmp_path)])
            out = capsys.readouterr().out
            assert "requires --" not in out
            assert rc == 2
            assert "devices" in out  # device guard, not arg contract
    finally:
        # main() sets the experimental opt-in env for tiled mode.
        os.environ.pop("LEGOESM_TILED_DYCORE_EXPERIMENTAL", None)


def test_tol_parser_rejects_unknown_field():
    mod = _load()
    with pytest.raises(SystemExit, match="p_s"):
        mod._parse_tols(["bogus=1e-3"])


def test_tol_parser_overrides_one_field():
    mod = _load()
    tol = mod._parse_tols(["T=9e-4"])
    assert tol["T"] == 9e-4
    assert tol["u"] == mod._DEFAULT_TOL["u"]  # others untouched
