"""Unit tests for ``scripts/validate/ocean_fidelity/omip_forcing_remap_integrals.py``.

``main`` needs the CORE-II cache and the eORCA1 / FESOM meshes and is not run
here; ``compare_methods`` -- which decides every number the probe prints -- is,
on a synthetic global point set with KNOWN answers:

* a constant field: every method and the conservative reference return the
  source global integral (the instrument self-check is exact for constants);
* a field with a strong meridional gradient: the conservative reference still
  reproduces the source integral to quadrature accuracy, while the sampled
  integrals differ from it (so a probe that returned the reference for every
  method would fail here).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_MOD = (Path(__file__).resolve().parents[2]
        / "scripts" / "validate" / "ocean_fidelity"
        / "omip_forcing_remap_integrals.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("omip_remap_integrals", _MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _src():
    lat = np.linspace(-87.5, 87.5, 36)            # 5-deg uniform centres
    lon = np.arange(0.0, 360.0, 7.5)              # 48 columns
    return lat, lon


def _global_points(mod):
    """A global equal-area-ish point set (Fibonacci sphere), with each point
    carrying an equal share of the source grid's area so that the target
    footprint is the source footprint."""
    n = 3000
    k = np.arange(n) + 0.5
    lat = np.degrees(np.arcsin(1.0 - 2.0 * k / n))
    lon = np.degrees(np.pi * (1.0 + 5.0 ** 0.5) * k) % 360.0
    src_lat, src_lon = _src()
    _, src_area = mod.source_grid(src_lat, src_lon)
    area = np.full(n, src_area.sum() / n)
    # keep the points inside the source band so all footprints agree
    keep = np.abs(lat) < 85.0
    return lat[keep], lon[keep], area[keep] * n / keep.sum()


def test_constant_field_every_method_matches_source(mod):
    src_lat, src_lon = _src()
    lat, lon, area = _global_points(mod)
    f = np.full((src_lat.size, src_lon.size), 3.0)
    r = mod.compare_methods(f, src_lat, src_lon, lat, lon, area,
                            {"global": np.ones(lat.size, bool)})["global"]
    for k in ("nearest", "bilinear", "nearest_rms", "bilinear_rms"):
        assert abs(r[k]) < 1e-12, k
    # cons uses the points' own areas; src the source areas: equal by build
    assert abs(r["cons_vs_src"]) < 1e-12


def test_gradient_field_methods_differ_from_reference(mod):
    src_lat, src_lon = _src()
    lat, lon, area = _global_points(mod)
    f = (np.exp(-((src_lat[:, None] - 5.0) / 8.0) ** 2)
         * (1.0 + 0.5 * np.cos(np.radians(3.0 * src_lon))[None, :]))
    res = mod.compare_methods(f, src_lat, src_lon, lat, lon, area,
                              {"global": np.ones(lat.size, bool),
                               "north": lat > 0.0})
    g = res["global"]
    assert g["nearest_rms"] > 1e-3 and g["bilinear_rms"] > 1e-3
    assert g["nearest"] != 0.0 and g["bilinear"] != 0.0
    assert "cons_vs_src" not in res["north"]
    assert np.isfinite(res["north"]["cons"])


def test_non_finite_source_is_fatal(mod):
    src_lat, src_lon = _src()
    lat, lon, area = _global_points(mod)
    f = np.ones((src_lat.size, src_lon.size))
    f[3, 4] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        mod.compare_methods(f, src_lat, src_lon, lat, lon, area,
                            {"global": np.ones(lat.size, bool)})
