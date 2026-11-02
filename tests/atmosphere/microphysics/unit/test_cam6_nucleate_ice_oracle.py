"""CAM6 cirrus ice nucleation (Liu-Penner with pre-existing ice) vs the CAM Fortran oracle.

Fixture ``oracle_fixtures/cam6_nucleate_ice.npz``: unmodified nucleate_ice.F90 and wv_sat_methods.F90 of
ESCOMP/CAM cam_cesm2_1_rel_60 with a driver transcribing nucleate_ice_cam_calc's per-level loop
(scripts/validate/oracle/cam6_nucleate_ice, build.sh then make_fixture.py), on 48 L32 columns from an A2
dump (24 as dumped, 24 with q_v reset to ice relative humidities 0.9 to 1.6) and synthetic modal aerosol
and sub-grid updrafts.  Tolerance: 1e-10 relative to each field's maximum, fp64.
"""
from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.atmosphere.physics.microphysics.cam6_nucleate_ice import (  # noqa: E402
    cam6_naai, nucleati, wsubi_from_wp2)

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "oracle_fixtures", "cam6_nucleate_ice.npz")
TOL = 1e-10


@pytest.fixture(scope="module")
def fx():
    return dict(np.load(FIXTURE))


def _naai(fx, **over):
    a = {k: jnp.asarray(fx[k]) for k in ("T", "q", "p", "qc", "qi", "ni", "ast", "wsubi", "num_accum",
                                          "num_aitken", "num_coarse", "coarse_dust", "coarse_nacl",
                                          "coarse_so4")}
    a.update(over)
    return np.asarray(cam6_naai(a["T"], a["q"], a["p"], a["qc"], a["qi"], a["ni"], a["ast"], a["wsubi"],
                                a["num_accum"], a["num_aitken"], a["num_coarse"], a["coarse_dust"],
                                a["coarse_nacl"], a["coarse_so4"], jnp.asarray(fx["troplev"]),
                                int(fx["top_lev"])))


def test_fixture_is_not_vacuous(fx):
    naai, oso4 = fx["oracle_naai"], fx["oracle_oso4"]
    assert (naai > 0).sum() > 100
    assert (oso4 > 0).sum() > 10, "homogeneous freezing never occurs in the fixture"
    assert (fx["oracle_nihf"] > naai * 0).any()
    assert "cam_cesm2_1_rel_60" in str(fx["meta"])


def test_naai_matches_cam_oracle(fx):
    ours = _naai(fx)
    ref = fx["oracle_naai"]
    assert np.array_equal(ours > 0, ref > 0)
    assert np.max(np.abs(ours - ref)) / np.max(np.abs(ref)) <= TOL


def test_pre_existing_ice_matters(fx):
    """Non-vacuity: dropping the pre-existing ice changes naai, so the pre-existing ice path is exercised."""
    no_pre = _naai(fx, ni=jnp.zeros_like(jnp.asarray(fx["ni"])))
    ref = fx["oracle_naai"]
    assert np.max(np.abs(no_pre - ref)) / np.max(np.abs(ref)) > 1e-3


def test_wsubi_from_wp2_is_microp_aero():
    wp2 = jnp.asarray([[0.0, 0.0, 0.06, 0.6, 600.0, 0.0]])
    w = np.asarray(wsubi_from_wp2(wp2, top_lev=1))
    tke = 1.5 * np.asarray(wp2)[0]
    ref = np.sqrt(0.5 * (tke[:-1] + tke[1:]) * (2.0 / 3.0))
    ref = np.maximum(0.001, np.minimum(ref, 10.0))
    ref[0] = 0.001
    np.testing.assert_allclose(w[0], ref, rtol=1e-14)


def test_nucleati_zero_without_aerosol():
    one = jnp.ones((1, 1))
    nuci, _, _, _ = nucleati(0.3 * one, 215.0 * one, 2.0e4 * one, 1.3 * one, 0.5 * one, 0.0 * one,
                             0.0 * one, 0.0 * one, 0.3 * one, 0.0 * one, 0.0 * one, 0.0 * one, 1.2)
    assert float(nuci[0, 0]) == 0.0
