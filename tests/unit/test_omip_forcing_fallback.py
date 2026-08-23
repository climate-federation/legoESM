"""OMIP forcing loaders must not SILENTLY substitute synthetic analytic data.

The CORE-II (OMIP-1) and JRA55-do (OMIP-2) loaders fall back to a deterministic
synthetic climatology when the on-disk cache is missing.  That fallback is a
footgun: a run can look like an OMIP integration while using non-protocol
forcing.  These tests pin the two safe behaviours:

* ``allow_synthetic=False`` raises ``FileNotFoundError`` (fail-loud — the
  OMIP-faithful pipelines, e.g. run_omip_core2.py, use this), and
* ``allow_synthetic=True`` (the convenience default for idealized spin-ups)
  emits a LOUD warning naming it as non-OMIP before returning synthetic data.
"""

from __future__ import annotations

import logging

import pytest

from legoesm.ocean.forcing.core2 import core2_nyf_path, load_core2_nyf
from legoesm.ocean.forcing.jra55_do import load_jra55_do


def test_core2_missing_cache_fails_loud_when_disallowed(tmp_path):
    with pytest.raises(FileNotFoundError, match="CORE-II NYF cache missing"):
        load_core2_nyf(cache_dir=tmp_path, allow_synthetic=False)


def test_core2_synthetic_fallback_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING,
                         logger="legoesm.ocean.forcing.core2"):
        f = load_core2_nyf(cache_dir=tmp_path, allow_synthetic=True, n_time=12)
    assert f.u10.shape[0] == 12  # synthetic returned
    assert any("SYNTHETIC" in r.message and "OMIP-1" in r.message
               for r in caplog.records), "missing loud synthetic-fallback warning"


def test_jra55_missing_cache_fails_loud_when_disallowed(tmp_path):
    with pytest.raises(FileNotFoundError, match="JRA55-do cache missing"):
        load_jra55_do(1990, cache_dir=tmp_path, allow_synthetic=False)


def test_jra55_synthetic_fallback_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING,
                         logger="legoesm.ocean.forcing.jra55_do"):
        load_jra55_do(1990, cache_dir=tmp_path, allow_synthetic=True)
    assert any("SYNTHETIC" in r.message and "OMIP-2" in r.message
               for r in caplog.records), "missing loud synthetic-fallback warning"


def test_core2_nyf_path_is_the_path_the_loader_reads(tmp_path):
    """``core2_nyf_path`` exists so a run can RECORD which archive it used
    (codex r8: with --forcing-path unset the location comes from the
    environment/home, so two identical command lines can read DIFFERENT
    forcing while the restart fingerprint recorded only ``forcing_path=None``).

    It must therefore agree with the loader EXACTLY.  Proven, not asserted:
    the loader names the resolved path in its FileNotFoundError.
    """
    assert core2_nyf_path(tmp_path) == tmp_path / "nyf.zarr"
    with pytest.raises(FileNotFoundError) as exc:
        load_core2_nyf(cache_dir=tmp_path, allow_synthetic=False)
    assert str(core2_nyf_path(tmp_path)) in str(exc.value)

    # ...and the default (cache_dir=None) resolves somewhere else entirely,
    # which is the whole reason the resolved path has to be fingerprinted.
    assert core2_nyf_path() != core2_nyf_path(tmp_path)
    assert core2_nyf_path().name == "nyf.zarr"
