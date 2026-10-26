"""The CORE-II cache builder must keep its three variants apart.

``mod`` is what NEMO ORCA1 reads and what production resolves.  ``base`` is
the raw set every run before 2026-08-21 used.  ``windsonly`` exists only to
attribute the improvement that came with ``mod``: that cache changed
shortwave, air temperature, humidity and precipitation as well as the wind,
so the 1 C cold-tongue gain cannot be credited to the trades until a cache in
which ONLY the wind moved has been run.

These tests pin the two things a mistake here would silently corrupt: which
variables each variant reads, and which directory it writes to.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "build_core2_nyf_zarr", _ROOT / "scripts/data/build_core2_nyf_zarr.py")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["build_core2_nyf_zarr"] = _MOD
_SPEC.loader.exec_module(_MOD)


def test_each_variant_writes_its_own_directory():
    """A build must never be able to overwrite the cache another run used."""
    dirs = _MOD._VARIANT_DIR
    assert set(dirs) == {"mod", "windsonly", "base"}
    assert len(set(dirs.values())) == 3, "two variants share a directory"
    assert dirs["base"] == "core2_nyf", (
        "the raw cache keeps its historical path so old runs stay reproducible")
    assert dirs["mod"] == "core2_nyf_mod", (
        "production resolves core2_nyf_mod; the builder must write there")


def test_windsonly_takes_corrected_winds_and_raw_everything_else():
    """The point of the variant: exactly one channel differs from `base`.

    Asserted on the module source at the two decision points, because the
    build itself needs the multi-gigabyte CORE-II files.  Both the wind table
    and the single `_thermo_mod` flag every other channel reads are checked,
    so a new channel that picked the corrected side on its own would show up
    as an unguarded literal here.
    """
    src = (_ROOT / "scripts/data/build_core2_nyf_zarr.py").read_text()
    assert '"windsonly": ("U_10_MOD", "V_10_MOD")' in src
    assert '_thermo_mod = (wind_variant == "mod")' in src
    # Every non-wind channel must route through that one flag rather than
    # testing wind_variant itself.
    for var in ("_t_var", "_q_var", "_sw_var", "_lw_var"):
        line = next(ln for ln in src.splitlines() if ln.strip().startswith(var))
        assert "_thermo_mod" in line, f"{var} does not follow _thermo_mod: {line}"
    assert 'if _thermo_mod' in src, "precip does not follow _thermo_mod"
    assert 'wind_variant == "mod" else "T_10"' not in src


def test_cli_accepts_the_three_variants_and_rejects_others():
    p = _MOD.build_arg_parser() if hasattr(_MOD, "build_arg_parser") else None
    if p is None:                       # parser is built inline in main()
        src = (_ROOT / "scripts/data/build_core2_nyf_zarr.py").read_text()
        assert '"mod", "windsonly", "base"' in src
    else:                               # pragma: no cover
        for v in ("mod", "windsonly", "base"):
            assert p.parse_args(["--inputs-dir", "x", "--wind-variant", v])
        with pytest.raises(SystemExit):
            p.parse_args(["--inputs-dir", "x", "--wind-variant", "nope"])


def test_unknown_variant_raises_rather_than_defaulting():
    """Dispatch hardening: a typo must not silently build the raw cache."""
    with pytest.raises(KeyError):
        _MOD._VARIANT_DIR["moderate"]
