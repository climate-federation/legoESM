"""CLI / SGS-default guard for scripts/run/run_spectral_les.py.

The neutral/ekman spectral-LES driver must DEFAULT to the Bou-Zeid LASD dynamic
SGS (like its sibling run_spectral_sbl.py / run_spectral_cbl.py drivers), because
the static constant-coefficient Smagorinsky closure cannot self-transition the
pure-shear neutral/ekman cases from a cold start (it damps the laminar
instabilities with mean-shear eddy viscosity → relaminarises). ``--static`` opts
into the static closure; ``--static`` and ``--dynamic`` are mutually exclusive.

These tests fail if a refactor reverts the default to static (the bug that made
the out-of-box neutral/ekman run laminarise while gabls1/cbl — which default to
dynamic — sustained).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run" / "run_spectral_les.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_spectral_les", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(m, extra):
    """A small-grid args namespace via the real parser (so defaults are exercised)."""
    return m.make_parser().parse_args(
        ["--nx", "8", "--ny", "8", "--nz", "8", "--f32", *extra])


def test_default_is_dynamic_lasd():
    """Bare invocation (no SGS flag) => dynamic LASD is wired into the grid cfg."""
    m = _load()
    args = _args(m, [])
    assert args.static is False and args.dynamic is False
    g, _st, _force = m.build(args, jnp.float32)
    assert g.cfg.smagorinsky_dynamic is True, (
        "neutral/ekman driver must default to dynamic LASD (static cannot "
        "self-transition the pure-shear cases from a cold start)")


def test_static_flag_selects_static():
    """--static opts into the constant-coefficient Smagorinsky closure."""
    m = _load()
    args = _args(m, ["--static"])
    assert args.static is True
    g, _st, _force = m.build(args, jnp.float32)
    assert g.cfg.smagorinsky_dynamic is False


def test_explicit_dynamic_still_accepted():
    """--dynamic is kept for back-compat (many scripts pass it); still dynamic."""
    m = _load()
    args = _args(m, ["--dynamic"])
    g, _st, _force = m.build(args, jnp.float32)
    assert g.cfg.smagorinsky_dynamic is True


def test_static_and_dynamic_mutually_exclusive():
    """Contradictory --static --dynamic is rejected at the parser (SystemExit)."""
    m = _load()
    with pytest.raises(SystemExit):
        _args(m, ["--static", "--dynamic"])


def test_vreman_submodel_requires_static():
    """--sgs-model vreman is a STATIC sub-closure; requesting it while dynamic
    (the default) is active must raise, not silently run LASD (codex review)."""
    m = _load()
    with pytest.raises(ValueError):
        m.build(_args(m, ["--sgs-model", "vreman"]), jnp.float32)
    # Paired with --static it is honoured.
    g, _st, _force = m.build(_args(m, ["--sgs-model", "vreman", "--static"]), jnp.float32)
    assert g.cfg.smagorinsky_dynamic is False and g.cfg.sgs_model == "vreman"
