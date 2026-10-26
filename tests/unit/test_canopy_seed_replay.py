"""canopy_seed_replay: the old gate certifies a garbage cached seed, the new one does not."""
import importlib.util
import pathlib
import sys

import jax
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "land" / "unit"))
_spec = importlib.util.spec_from_file_location(
    "canopy_seed_replay", ROOT / "scripts/validate/amip_bias/canopy_seed_replay.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)


def test_replay_counts(tmp_path, capsys):
    if not jax.config.jax_enable_x64:
        import pytest
        pytest.skip("run with JAX_ENABLE_X64=1")
    x = np.array([[250.0, 250.5, 300.0, 300.0, 249.8, 3e-4],      # sane cache
                  [1189.11, 4251.002, 373.5, 373.5, 233.287, 0.0],  # AMIP garbage
                  [np.nan] * 6])
    f = tmp_path / "ck.npz"
    np.savez(f, land_ml_canopy_x=x, T=np.full((3, 4), 248.0),
             trc_q_v=np.full((3, 4), 4e-4), p_s=np.full(3, 99000.0))
    replay.main([str(f), "--lai", "0.5"])
    out = capsys.readouterr().out
    assert "seeds 2 | inadmissible seeds 1" in out
    assert "NOT converged (held) 0" in out
