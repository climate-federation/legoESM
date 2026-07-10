"""CLI/arg-contract tests for scripts/validate/validate_driver_cs_spmd_parity.py.

The end-to-end 2-process parity itself needs a real multi-process launch
(scripts/cluster/scaling_ginsburg/a1_cs_spmd_smoke.sbatch); here we pin the
importability, the config contract (spmd-legal: cubed-sphere, diag/checkpoint
off), and the argument guards so a broken invocation fails loudly before
burning batch walltime.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "validate_driver_cs_spmd_parity.py")


_LOAD_COUNT = [0]


def _load():
    # Unique module name per load; never left behind in sys.modules
    # (test pollution, codex Low).
    _LOAD_COUNT[0] += 1
    name = f"_validate_cs_spmd_parity_{_LOAD_COUNT[0]}"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def test_build_config_is_spmd_legal(tmp_path):
    """The shared config must pass validate_strict BOTH plain and with the
    spmd flags applied (the exact replace the spmd lane performs)."""
    mod = _load()
    cfg = mod._build_config(str(tmp_path))
    cfg.validate_strict()
    cfg._replace(distributed=True,
                 distributed_mode="spmd").validate_strict()
    assert cfg.grid.grid_type == "cubed_sphere"
    assert cfg.output.diag_days == 0
    assert cfg.output.checkpoint_days == 0


def test_serial_mode_requires_out(tmp_path):
    mod = _load()
    rc = mod.main(["--mode", "serial", "--workdir", str(tmp_path)])
    assert rc == 2


def test_spmd_mode_requires_multiprocess(tmp_path):
    """Single-process --mode spmd must refuse (exit 2), not silently run a
    1-process 'parity'."""
    mod = _load()
    rc = mod.main(["--mode", "spmd", "--ref", "/nonexistent.npz",
                   "--workdir", str(tmp_path)])
    assert rc == 2
