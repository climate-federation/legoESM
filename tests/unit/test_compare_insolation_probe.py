"""The climlab comparison probe's model side writes every case it compares."""

import importlib.util
from pathlib import Path

import numpy as np

_PATH = (Path(__file__).resolve().parents[2]
         / "scripts/validate/oracle/orbit_insolation/compare_insolation.py")
_spec = importlib.util.spec_from_file_location("compare_insolation", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_model_side_orbits_match_pmip4_table(tmp_path):
    out = tmp_path / "m.npz"
    mod.model(str(out))
    m = np.load(out)
    assert m["q_present"].shape == (mod.DAYS.size, mod.LAT.size)
    for k, (e, obl, peri_m180) in mod.PMIP4.items():
        me, mo, mp = m[f"orb_{k}"]
        # Half the last digit PMIP4 Table 1 prints (6, 3 and 2 decimals).
        assert abs(me - e) <= 5e-7 and abs(mo - obl) <= 5e-4
        assert abs(((mp - 180.0) % 360.0) - peri_m180) <= 5e-3


def test_probe_kepler_reference_agrees_with_model_solver():
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.radiation.solar import (
        OrbitalParameters, solar_declination,
    )
    days = np.linspace(1.0, 365.0, 400)
    lam = np.deg2rad(mod.kepler_true_longitude_deg(days, 0.9, 282.9))
    orbit = OrbitalParameters(0.9, 23.44, 282.9, solver="kepler")
    np.testing.assert_allclose(
        np.sin(np.asarray(solar_declination(jnp.asarray(days), orbit=orbit))),
        np.sin(np.deg2rad(23.44)) * np.sin(lam), atol=1e-12)
