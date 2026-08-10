"""``plot_scm_les_turbulence.py`` must never silently pick a case for you.

A multi-case campaign directory holds one ``profiles_<case>.npz`` per case.
The plotter used to take the first glob match, so a driver loop that asked for
five per-case figures got the SAME figure (alphabetically first: bomex) written
under five different per-case filenames -- five byte-identical PNGs presented
as five regimes. These tests pin the selection contract that replaced it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "plot" / "plot_scm_les_turbulence.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("_plot_scm_les", _SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _touch_npz(path: Path):
    np.savez(path, z_scm=np.arange(4.0), mask=np.ones(4, dtype=bool),
             window_hours=np.array([4.0, 6.0]))


def test_explicit_case_selects_that_file(tmp_path, mod):
    for case in ("bomex", "ekman", "gabls1"):
        _touch_npz(tmp_path / f"profiles_{case}.npz")
    npz, label = mod._resolve_profiles(tmp_path, "ekman")
    assert npz.name == "profiles_ekman.npz"
    assert label == "ekman"


def test_ambiguous_directory_raises_instead_of_guessing(tmp_path, mod):
    """The regression that produced five identical PNGs."""
    for case in ("bomex", "ekman"):
        _touch_npz(tmp_path / f"profiles_{case}.npz")
    with pytest.raises(SystemExit) as exc:
        mod._resolve_profiles(tmp_path, None)
    msg = str(exc.value)
    assert "--case" in msg
    # names the options rather than just complaining
    assert "bomex" in msg and "ekman" in msg


def test_unknown_case_raises_and_lists_what_exists(tmp_path, mod):
    _touch_npz(tmp_path / "profiles_bomex.npz")
    with pytest.raises(SystemExit) as exc:
        mod._resolve_profiles(tmp_path, "dycoms")
    msg = str(exc.value)
    assert "dycoms" in msg and "bomex" in msg


def test_single_case_directory_still_needs_no_flag(tmp_path, mod):
    """Backwards compatible: a one-case run keeps working with no --case."""
    _touch_npz(tmp_path / "profiles_bomex.npz")
    npz, label = mod._resolve_profiles(tmp_path, None)
    assert npz.name == "profiles_bomex.npz"
    assert label == "bomex"


def test_legacy_flat_profiles_npz_is_preferred(tmp_path, mod):
    _touch_npz(tmp_path / "profiles.npz")
    _touch_npz(tmp_path / "profiles_bomex.npz")
    npz, label = mod._resolve_profiles(tmp_path, None)
    assert npz.name == "profiles.npz"
    assert label == tmp_path.name


def test_empty_directory_raises(tmp_path, mod):
    with pytest.raises(SystemExit):
        mod._resolve_profiles(tmp_path, None)


def test_excluded_schemes_are_not_drawn(tmp_path, mod):
    """A non-finite arm still has a score_default, so filtering on that alone
    drew a blown-up scheme next to the ranked ones with nothing to say so."""
    (tmp_path / "ranking.csv").write_text(
        "rank,scheme,status,score_default,score_tuned\n"
        "1,louis,tuned,0.48,0.43\n"
        "2,tke,tuned,0.83,0.78\n"
        "EXCLUDED,mynn25,nonfinite_rollout,200.19,\n"
    )
    assert mod._read_ranking(tmp_path / "ranking.csv") == ["louis", "tke"]


def test_missing_ranking_file_is_not_an_error(tmp_path, mod):
    assert mod._read_ranking(tmp_path / "ranking.csv") == []
