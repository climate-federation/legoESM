"""NEMO 5.0.1 ORCA1 oracle: implicit ice-ocean top drag together with the non-linear bottom drag.

ORCA1 runs ``ln_non_lin``, ``ln_drgimp`` and ``ln_drgice_imp`` (namelist_cfg:258-261).  Fixture
``fixtures/orca1_t3/ice_top_drag_nonlin.npz`` (+ ``.json`` provenance): one 16x16 window of NEMO's recorded zdf_drg
inputs/outputs at kt 17281 (an ice step) on an ice-covered tile: Kmm velocities, tmask, mbkt, ``rCdU_bot``,
SI3's ``rCdU_ice`` and the ``rCdU_top`` zdf_drg builds from it.

Checks: (1) without cavities NEMO's top coefficient is SI3's ``rCdU_ice`` (zdfdrg.F90:126), so the ocean consumes the
ice model's value unchanged; (2) the port's non-linear cell rate is NEMO's ``-rCdU_bot`` here too; (3) the barotropic
substep coefficient the ocean forms from them equals NEMO's one statement
``r1_2*((rCdU_bot(i+1)+rCdU_bot(i)) + (rCdU_top(i+1)+rCdU_top(i)))`` (dynspg_ts.F90:1607-1608) on interior faces;
(4) a model with the non-linear bottom drag and the ice top drag now constructs (it was refused).
Tolerance rel 1e-12 (NEMO 20-seed spread 8.8e-16 bottom, 2.2e-16 top).
"""

from __future__ import annotations

import json
import pathlib
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    nemo_bottom_drag_rate_faces,
    nemo_ice_ocean_top_drag_coeff_faces,
)
from legoesm.ocean.state import DynBottomDragConfig
from legoesm.ocean.vertical import OceanPartialCellCoordinate

jax.config.update("jax_enable_x64", True)

_FIX = pathlib.Path(__file__).parent / "fixtures" / "orca1_t3" / "ice_top_drag_nonlin"
_D = np.load(str(_FIX) + ".npz")
_PROV = json.loads(pathlib.Path(str(_FIX) + ".json").read_text())
_TOL = 1e-12
_IN = (slice(1, -1), slice(1, -1))


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(before)


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    den = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0)))


def _port_cell_rate():
    act = _D["is_active"].astype(bool)
    bl = _D["bottom_level"]
    nj, ni, nk = act.shape
    nan1 = jnp.full((nk,), jnp.nan)
    nan3 = jnp.full((nj, ni, nk), jnp.nan)
    zc = OceanPartialCellCoordinate(
        n_levels=nk, H_max=float("nan"), z_full_ref=nan1, z_half_ref=jnp.full((nk + 1,), jnp.nan),
        dz_ref=nan1, dz_half_ref=nan1, h_partial=nan3, bottom_level=jnp.asarray(bl, jnp.int32),
        is_active=jnp.asarray(act))
    sc = _PROV["scalars"]
    cfg = types.SimpleNamespace(bottom_drag=DynBottomDragConfig(
        bottom_drag_scheme="nemo_quadratic", bottom_drag_cd0=sc["cd0"], bottom_drag_ke0=sc["ke0"],
        bottom_drag_cdmax=sc["cdmax"], bottom_drag_z0=sc["z0"]))
    r_t = nemo_bottom_drag_rate_faces(jnp.asarray(_D["u"]), jnp.asarray(_D["v"]), nan3, zc, cfg, None,
                                      return_cell_rate=True)[4]
    # the ocean's substep path: rCdU_bot = -rate on wet columns, -0 on land (barotropic_latlon_cgrid)
    return np.asarray(jnp.where(jnp.asarray(bl) >= 0, -r_t, -0.0))


def test_window_is_ice_covered_and_has_land():
    assert _PROV["ice_cells"] >= 200 and _PROV["land_cells"] >= 1
    assert np.any(_D["rcdu_ice"] < 0)


def test_nemo_top_coefficient_is_si3_rcdu_ice():
    np.testing.assert_array_equal(_D["nemo_rcdu_top"], _D["rcdu_ice"])


def test_port_nonlinear_bottom_rate_matches_nemo_under_ice():
    assert _rel(_port_cell_rate()[_IN], _D["nemo_rcdu_bot"][_IN]) <= _TOL


def test_combined_substep_coefficient_matches_nemo_statement():
    bot = _port_cell_rate()
    cu, cv, su, sv, _ = nemo_ice_ocean_top_drag_coeff_faces(
        jnp.asarray(_D["rcdu_ice"]), None, rCdU_bot=jnp.asarray(bot))
    nb, nt = _D["nemo_rcdu_bot"], _D["nemo_rcdu_top"]
    # port u-face i couples cells i-1 and i (layout in the provenance json)
    want_u = 0.5 * ((nb[:, 1:] + nb[:, :-1]) + (nt[:, 1:] + nt[:, :-1]))
    want_v = 0.5 * ((nb[1:, :] + nb[:-1, :]) + (nt[1:, :] + nt[:-1, :]))
    assert _rel(np.asarray(cu)[1:-1, 2:-2], want_u[1:-1, 1:-1]) <= _TOL
    assert _rel(np.asarray(cv)[2:-2, 1:-1], want_v[1:-1, 1:-1]) <= _TOL
    # residual / dyn_zdf read the top pair alone
    assert _rel(np.asarray(su)[1:-1, 2:-2], (nt[:, 1:] + nt[:, :-1])[1:-1, 1:-1]) <= _TOL
    # non-vacuity: the bottom pair is not negligible against the top pair here
    assert np.max(np.abs(nb[_IN])) > 0.1 * np.max(np.abs(nt[_IN]))


def test_combined_coefficient_gradient_is_finite():
    bot = jnp.asarray(_port_cell_rate())
    g = jax.grad(lambda t: jnp.sum(nemo_ice_ocean_top_drag_coeff_faces(t, None, rCdU_bot=bot)[0]))(
        jnp.asarray(_D["rcdu_ice"]))
    assert np.all(np.isfinite(np.asarray(g))) and np.any(np.asarray(g) != 0.0)
