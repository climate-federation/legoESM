"""Tests for the LES/CRM forcing cache: the fetch script + the resolver.

Covers ``scripts/data/fetch_les_forcing.py`` (case map + ``copy_deck`` skip
rules) and ``legoesm.atmosphere.forcing.sam_case_forcing.resolve_sam_case_dir`` (the
env / repo-local-cache precedence the gSAM drivers default to).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_FETCH = (
    Path(__file__).resolve().parents[2] / "scripts" / "data" / "fetch_les_forcing.py"
)


def _load_fetch():
    spec = importlib.util.spec_from_file_location("fetch_les_forcing", _FETCH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_case_map_covers_the_eight_gsam_cases():
    m = _load_fetch()
    # The 5 driver-backed cases + the 3 driverless decks the user listed.
    assert set(m.CASE_TO_GSAM_DIR) == {
        "BOMEX", "RICO", "DYCOMSII", "GATE", "LBA", "ARM97", "ASTEX", "TOGA",
    }
    # DYCOMS-II carries both flights; the rest map to a single gSAM dir.
    assert m.CASE_TO_GSAM_DIR["DYCOMSII"] == ("DYCOMS_RF01", "DYCOMS_RF02")
    assert m.CASE_TO_GSAM_DIR["GATE"] == ("GATE_IDEAL",)
    # The dry-PBL cases are analytic — deliberately NOT in the deck map.
    assert set(m.ANALYTIC_CASES) == {"GABLS1", "Wangara", "Ekman"}
    for dry in m.ANALYTIC_CASES:
        assert dry not in m.CASE_TO_GSAM_DIR


def test_copy_deck_skips_fortran_and_oversized(tmp_path):
    m = _load_fetch()
    src = tmp_path / "CASE"
    src.mkdir()
    (src / "snd").write_text("sounding\n")
    (src / "lsf").write_text("forcing\n")
    (src / "setdata.f").write_text("! fortran source\n")        # skipped: .f
    (src / "huge").write_bytes(b"\0" * (m._MAX_FILE_BYTES + 1))  # skipped: >2 MB
    (src / "sub").mkdir()                                        # skipped: not a file

    dest = tmp_path / "out"
    n, nbytes = m.copy_deck(src, dest)

    copied = sorted(p.name for p in dest.iterdir())
    assert copied == ["lsf", "snd"]
    assert n == 2 and nbytes > 0


def test_resolve_prefers_gsam_root_then_local_cache(tmp_path, monkeypatch):
    from legoesm.atmosphere.forcing.sam_case_forcing import resolve_sam_case_dir

    gsam = tmp_path / "gsam"
    cache = tmp_path / "cache"
    (gsam / "CASES" / "BOMEX").mkdir(parents=True)
    (cache / "BOMEX").mkdir(parents=True)   # SAME case present in BOTH
    (cache / "RICO").mkdir(parents=True)    # only in the cache
    monkeypatch.setenv("LEGOESM_GSAM_ROOT", str(gsam))
    monkeypatch.setenv("LEGOESM_LES_FORCING", str(cache))

    # Both present -> the external gSAM checkout wins (would fail if precedence
    # were swapped, since cache/BOMEX also exists).
    assert resolve_sam_case_dir("BOMEX") == str(gsam / "CASES" / "BOMEX")
    # Absent in the gSAM root -> fall through to the repo-local cache.
    assert resolve_sam_case_dir("RICO") == str(cache / "RICO")

    # Nothing anywhere -> the cache path is still returned (actionable error
    # downstream: run the fetch script / pass --case-dir).
    monkeypatch.delenv("LEGOESM_GSAM_ROOT", raising=False)
    assert resolve_sam_case_dir("NOPE") == str(cache / "NOPE")


def test_default_cache_points_at_repo_root_data_les_cases():
    """Regression: the federation restructure moved sam_case_forcing.py deeper
    (added the packages/<pkg>/ nesting); a hardcoded ``parents[N]`` silently
    pointed the default cache at ``packages/data/les_cases`` (nonexistent) so
    LES auto-resolve broke. The monkeypatched test above never caught it because
    it overrides the cache. Pin the DEFAULT to the real repo-root dir.
    """
    from pathlib import Path

    from legoesm.atmosphere.forcing import sam_case_forcing

    cache = Path(sam_case_forcing._LOCAL_FORCING_CACHE)
    assert cache.name == "les_cases"
    assert cache.parent.name == "data"
    # repo root == the dir holding .git; must NOT be under packages/.
    repo_root = cache.parent.parent
    assert (repo_root / ".git").exists(), f"cache not anchored at repo root: {cache}"
    assert "packages" not in cache.parts, f"cache leaked under packages/: {cache}"
    # The committed BOMEX deck must resolve here (no env override, no gSAM root).
    import os

    for var in ("LEGOESM_GSAM_ROOT", "LEGOESM_LES_FORCING"):
        os.environ.pop(var, None)
    assert (cache / "BOMEX").is_dir(), "committed BOMEX deck not found at default cache"


if __name__ == "__main__":
    test_case_map_covers_the_eight_gsam_cases()
    test_default_cache_points_at_repo_root_data_les_cases()
    print("ok (run the tmp_path/monkeypatch tests under pytest)")
