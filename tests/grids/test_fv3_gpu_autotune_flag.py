"""Regression coverage for the M6 GPU launcher's XLA flag setup."""

import os
from pathlib import Path
import subprocess

import pytest


_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts/cluster/fv3_native/tiled_m6_model_gate_gpu.sbatch")


@pytest.mark.parametrize("existing", [None, "", "--xla_force_host_platform_device_count=2"],
                         ids=["unset", "empty", "preexisting"])
def test_gpu_autotune_flag_preserves_existing_flags_and_echoes(existing):
    # Execute the launcher's flag setup in source order without starting Slurm.
    lines = _SCRIPT.read_text().splitlines()
    setup = "\n".join(line for line in lines
                      if line.startswith("export XLA_FLAGS=")
                      or line.startswith('echo "XLA_FLAGS='))
    assert "export XLA_FLAGS=" in setup, "Missing XLA_FLAGS export"
    env = os.environ.copy()
    env.pop("XLA_FLAGS", None)
    if existing is not None:
        env["XLA_FLAGS"] = existing
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-uc", setup],
        env=env, capture_output=True, text=True, check=True, timeout=10,
    )
    expected = ((existing + " ") if existing else "") + "--xla_gpu_autotune_level=0"
    assert result.stdout.splitlines() == [f"XLA_FLAGS={expected}"]
