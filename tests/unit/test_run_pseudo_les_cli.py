"""Direct test for the pseudo-incompressible dry-ABL driver scripts/run/run_pseudo_les.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run" / "run_pseudo_les.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_pseudo_les", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cases_registry_nonempty():
    m = _load()
    assert set(m._CASES) == {"neutral", "ekman", "gabls1", "wangara"}
    for c in m._CASES.values():
        assert c["surface"] in ("free", "flux", "most_cooling")


@pytest.mark.parametrize("case", ["ekman", "gabls1", "wangara"])
def test_one_step_runs_finite(case):
    """Tiny 1-step run per case stays finite (build -> project -> step)."""
    m = _load()

    class A:
        nx = ny = 8
        nz = 12
        Lx = 400.0
        sgs = "vreman"
        nu_floor = None
        ic_amp = 0.1
        maxiter = 40
        hours = None
    args = A()
    cfg, g, st, c = m.build(case, args, jnp.float64)
    forcing = None
    if c["cooling_rate"] > 0:
        forcing = m.pip.PseudoIncompressibleForcing(
            t_sfc=jnp.asarray(c["theta0"], jnp.float64))
    st2 = m.pip.step(st, g, jnp.asarray(c["dt"], jnp.float64), forcing)
    d = m.diagnose(st2, g, cfg)
    assert d["finite"], f"{case}: non-finite after 1 step"
    assert np.isfinite(d["ustar_wall"]) and d["ustar_wall"] >= 0.0


def test_unknown_case_rejected(monkeypatch):
    """argparse choices reject an unknown case at the CLI (dispatch hardening)."""
    m = _load()
    assert "bogus" not in m._CASES
    monkeypatch.setattr("sys.argv", ["run_pseudo_les.py", "--case", "bogus"])
    with pytest.raises(SystemExit):
        m.main()
