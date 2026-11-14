"""run_spectral_sbl.py: the MOST zeta-cap width flag reaches the LES config."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp

from legoesm.core.bulk_flux import MOST_ZETA_CAP_WIDTH

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run" / "run_spectral_sbl.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_spectral_sbl", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flag_default():
    tree = ast.parse(_PATH.read_text())
    for c in ast.walk(tree):
        if (isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "add_argument"
                and c.args and getattr(c.args[0], "value", None) == "--most-zeta-cap-width"):
            return next(ast.unparse(k.value) for k in c.keywords if k.arg == "default")
    raise AssertionError("--most-zeta-cap-width flag missing")


def test_flag_defaults_to_the_shared_width():
    assert _flag_default() == "MOST_ZETA_CAP_WIDTH"


def test_build_threads_the_width_into_the_config():
    m = _load()
    args = SimpleNamespace(nx=4, ny=4, nz=6, Lx=400.0, Ly=400.0, Lz=400.0, z0=0.1,
                           cs=0.18, static=True, sgs_model="smagorinsky",
                           time_scheme="rk3", theta0=265.0, nu_floor=0.05, Ug=8.0,
                           dt=0.1, most_zeta_cap_width=0.0)
    g, _st = m.build(args, jnp.float64)
    assert g.cfg.most_zeta_cap_width == 0.0
    args.most_zeta_cap_width = MOST_ZETA_CAP_WIDTH
    g, _st = m.build(args, jnp.float64)
    assert g.cfg.most_zeta_cap_width == MOST_ZETA_CAP_WIDTH
