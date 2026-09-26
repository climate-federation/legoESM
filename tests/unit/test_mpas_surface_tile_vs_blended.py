"""Direct test for ``scripts/validate/mpas_surface_tile_vs_blended.py`` (#1320).

The probe quantifies how much the MPAS blended surface differs from the tiled
one.  These tests pin the two properties that decide whether its number means
anything:

* in the SINGLE-TILE limit the two paths must agree, because there is nothing
  to mis-average -- if they disagree there, the probe is measuring a config
  mismatch between its two arms rather than the blending approximation;
* at an intermediate fraction with a real stability contrast they must
  DISAGREE, or the probe could report "no error" from a case that could never
  show one.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

from legoesm import constants

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "scripts" / "validate" / "mpas_surface_tile_vs_blended.py"


def _load():
    spec = importlib.util.spec_from_file_location("_tile_vs_blended", _PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tile_vs_blended"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fluxes(mod, *, frac_land, T_land, beta=1.0):
    col = mod._column(T_air=constants.T_freeze, q_air=2.0e-3, wind=8.0,
                      p_sfc=1.0e5, rho=1.25)
    return mod._pair(frac_land=frac_land, frac_ice=0.0,
                     T_ocean=278.15, T_ice=constants.T_freeze, T_land=T_land,
                     beta=beta, col=col, p_sfc=1.0e5)


def test_all_ocean_column_agrees_between_the_two_paths():
    """frac_land = frac_ice = 0: both paths are the ocean scheme on the SST."""
    mod = _load()
    tiled, blended = _fluxes(mod, frac_land=0.0, T_land=253.15)
    for i, name in ((2, "sensible"), (3, "latent")):
        assert float(tiled[i][0]) == pytest.approx(
            float(blended[i][0]), rel=1e-10, abs=1e-8), (
            f"{name} heat flux differs on an all-ocean column: the probe's two "
            f"arms are not the same configuration")


def test_all_land_column_disagrees_only_by_the_bulk_scheme():
    """frac_land = 1: the surface state is identical, the SCHEME is not.

    The blended arm keeps the ocean air-sea scheme (that is the defect being
    measured); the tiled arm runs the land Monin-Obukhov scheme.  So the two
    must differ, and the difference is attributable to the scheme alone.
    """
    mod = _load()
    tiled, blended = _fluxes(mod, frac_land=1.0, T_land=253.15)
    assert abs(float(tiled[2][0]) - float(blended[2][0])) > 1.0


def test_intermediate_fraction_shows_a_real_mis_average():
    """Non-vacuity: at half cover with a stability contrast the paths differ."""
    mod = _load()
    tiled, blended = _fluxes(mod, frac_land=0.5, T_land=253.15, beta=0.4)
    d_turb = ((float(blended[2][0]) + float(blended[3][0]))
              - (float(tiled[2][0]) + float(tiled[3][0])))
    assert abs(d_turb) > 1.0, (
        f"blended and tiled agree to {d_turb:.3g} W/m^2 at half land cover "
        "with a 25 K land-ocean skin contrast — the probe would report no "
        "error from a case that cannot show one")


def test_probe_runs_end_to_end():
    mod = _load()
    assert mod.main([]) == 0
