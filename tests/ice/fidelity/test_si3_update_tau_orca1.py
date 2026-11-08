"""NEMO 5.0.1 ORCA1 oracle for ``si3_ice_update_tau``: implicit ice-ocean drag ``rCdU_ice`` and ice cadence nn_fsbc > 1.

Fixture ``tests/ice/fidelity/fixtures/orca1_t4/si3_update_tau_orca1.npz`` (+ ``.json`` provenance): one 6 x 6 window of
ice-covered interior cells with a one-cell margin, from the tier-4 U4 record of NEMO 5.0.1 SI3 ORCA1 (``ice_update_tau``
at the ice step kt 17281, nn_fsbc 96, ln_drgice_imp T).  ``nn_hls=1`` here is a window device: the margin plays the
halo, and NEMO's own ``tmod_io`` on the margin is carried in (NEMO computes it on halo 1 because ORCA1 runs nn_hls=2).
Pass rule = the tier-4 class rule, max relative error <= max(1e-12, oracle spread); the U4 spreads are <= 1.8e-11.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ice.si3_finalize import SI3StressInputs, si3_ice_update_tau
from legoesm.ice.si3_orca2_state import SI3ORCA2StateConfig

jax.config.update("jax_enable_x64", True)

_FIX = Path(__file__).resolve().parent / "fixtures" / "orca1_t4" / "si3_update_tau_orca1"
# tier-4 U4 seed spread, field max (tier4/update/REPORT.txt section 3)
_SPREAD = dict(tmod_io=1.6e-14, utau_oce=1.8e-11, vtau_oce=1.2e-11, taum=1.0e-11, utau=1.8e-11, vtau=1.2e-11,
               rCdU_ice=1.4e-14)


def _rel(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    d = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.where(d > 0, np.abs(a - b) / np.where(d > 0, d, 1.0), 0.0)))


def _load():
    d = dict(np.load(str(_FIX) + ".npz"))
    prov = json.loads(Path(str(_FIX) + ".json").read_text())
    assert prov["oracle"].startswith("NEMO 5.0.1")
    return d


def _config(d, nn_fsbc: int):
    base = SI3ORCA2StateConfig()
    assert base.ice_constants.rho_ocean == float(d["rho0"])   # NEMO rho0 = the SI3 constant set, no override
    return base._replace(nn_hls=1, nn_fsbc=nn_fsbc)


def _inputs(d, **over):
    shape = d["u_ice"].shape
    one = np.ones(shape)
    kw = dict(
        uu_kbb=d["pu_oce"], vv_kbb=d["pv_oce"], umask_k1=one, vmask_k1=one,
        u_ice=d["u_ice"], v_ice=d["v_ice"], u_oce=d["u_oce"], v_oce=d["v_oce"],
        drag_io=d["drag_io"], at_i=d["at_i"], utau=d["utau"], vtau=d["vtau"],
        tmod_io=d["nemo_tmod_io_blk"], taum=d["taum_in"],
        utau_oce=d["utau"][1:-1, 1:-1], vtau_oce=d["vtau"][1:-1, 1:-1],
        tmask_k1=d["tmask_k1"],
    )
    kw.update(over)
    return SI3StressInputs(**{k: (None if v is None else jnp.asarray(v)) for k, v in kw.items()})


def test_ice_step_matches_nemo_orca1_including_rcdu_ice():
    d = _load()
    out = si3_ice_update_tau(_inputs(d), config=_config(d, 96), ln_drgice_imp=True, ice_step=True)
    inner = (slice(1, -1), slice(1, -1))
    got = dict(tmod_io=out.tmod_io[inner], utau_oce=out.utau_oce, vtau_oce=out.vtau_oce, taum=out.taum,
               utau=out.utau[inner], vtau=out.vtau[inner], rCdU_ice=out.rCdU_ice[inner])
    ref = dict(tmod_io=d["nemo_tmod_io"], utau_oce=d["nemo_utau_oce"], vtau_oce=d["nemo_vtau_oce"],
               taum=d["nemo_taum_out"], utau=d["nemo_utau_out"], vtau=d["nemo_vtau_out"],
               rCdU_ice=d["nemo_rCdU_ice"])
    assert np.all(d["at_i"][1:-1, 1:-1] > 0)        # every scored cell is ice-covered
    assert np.any(np.asarray(ref["rCdU_ice"]) < 0)  # the implicit drag is active, not a zero field
    for k in ref:
        assert _rel(got[k], ref[k]) <= max(1e-12, _SPREAD[k]), k


def test_rcdu_ice_covers_halo_one_and_is_masked():
    d = _load()
    out = si3_ice_update_tau(_inputs(d), config=_config(d, 96), ln_drgice_imp=True)
    assert out.rCdU_ice.shape == d["u_ice"].shape      # DO_2D(1,1,1,1) at nn_hls=1 = the whole array
    tm = d["tmask_k1"].copy()
    tm[0, 0] = 0.0
    masked = si3_ice_update_tau(_inputs(d, tmask_k1=tm), config=_config(d, 96), ln_drgice_imp=True)
    assert float(masked.rCdU_ice[0, 0]) == 0.0


def test_explicit_arm_returns_no_rcdu_ice():
    d = _load()
    out = si3_ice_update_tau(_inputs(d), config=_config(d, 96), ln_drgice_imp=False)
    assert out.rCdU_ice is None


def test_implicit_arm_without_tmask_forms_no_rcdu_ice():
    """No mask, no drag: rCdU_ice is None (the ocean's implicit top drag refuses a None input); the stress is unchanged."""
    d = _load()
    a = si3_ice_update_tau(_inputs(d, tmask_k1=None), config=_config(d, 96), ln_drgice_imp=True)
    b = si3_ice_update_tau(_inputs(d), config=_config(d, 96), ln_drgice_imp=True)
    assert a.rCdU_ice is None
    np.testing.assert_array_equal(a.utau, b.utau)


def test_wrong_tmask_shape_is_refused():
    d = _load()
    with pytest.raises(ValueError, match="tmask_k1"):
        si3_ice_update_tau(_inputs(d, tmask_k1=d["tmask_k1"][1:, :]), config=_config(d, 96), ln_drgice_imp=True)


def test_ocean_step_between_ice_steps_holds_ice_step_quantities():
    """MOD(kt-1, nn_fsbc) /= 0: tmod_io, taum, utau_oce, vtau_oce are held; only the blend uses the new u_ice/pu_oce.
    Known answer = iceupdate.F90:388-404 written out in numpy (no NEMO record exists at a non-ice step)."""
    d = _load()
    cfg = _config(d, 96)
    held = si3_ice_update_tau(_inputs(d), config=cfg, ln_drgice_imp=True, ice_step=True)
    u_new = d["u_ice"] * 1.3 + 0.01
    pu_new = d["pu_oce"] * 0.7
    nxt = _inputs(d, u_ice=u_new, uu_kbb=pu_new, u_oce=d["u_oce"] * 5.0, taum=held.taum,
                  tmod_io=held.tmod_io, utau_oce=held.utau_oce, vtau_oce=held.vtau_oce)
    for imp in (True, False):
        out = si3_ice_update_tau(nxt, config=cfg, ln_drgice_imp=imp, ice_step=False)
        np.testing.assert_array_equal(out.tmod_io, held.tmod_io)
        np.testing.assert_array_equal(out.taum, held.taum)
        np.testing.assert_array_equal(out.utau_oce, held.utau_oce)
        tm = np.asarray(held.tmod_io)[1:-1, 1:-1]
        at = d["at_i"][1:-1, 1:-1]
        flag = 0.0 if imp else 1.0
        urel = (u_new[1:-1, 1:-1] + u_new[:-2, 1:-1]) - flag * (pu_new[1:-1, 1:-1] + pu_new[:-2, 1:-1])
        ref_u = (1.0 - at) * np.asarray(held.utau_oce) + at * (0.5 * tm * urel)
        assert _rel(out.utau[1:-1, 1:-1], ref_u) <= 1e-15
        if imp:
            ref_r = -(1.0 / float(d["rho0"])) * np.asarray(held.tmod_io) * d["at_i"] * d["tmask_k1"]
            assert _rel(out.rCdU_ice, ref_r) <= 1e-15


def test_ice_step_true_equals_previous_single_call_form():
    """nn_fsbc=1: the ice step is every step; the composed call is unchanged from the O16 form (no rCdU_ice change)."""
    d = _load()
    a = si3_ice_update_tau(_inputs(d), config=_config(d, 1), ln_drgice_imp=False)
    b = si3_ice_update_tau(_inputs(d), config=_config(d, 96), ln_drgice_imp=False, ice_step=True)
    for x, y in zip(a, b):
        if x is not None:
            np.testing.assert_array_equal(x, y)


def test_rcdu_ice_gradient_matches_central_difference():
    d = _load()
    cfg = _config(d, 96)

    def f(s):
        return jnp.sum(si3_ice_update_tau(_inputs(d, u_ice=jnp.asarray(d["u_ice"]) * s), config=cfg,
                                          ln_drgice_imp=True).rCdU_ice)

    x = jnp.asarray(1.0)
    g = float(jax.grad(f)(x))
    eps = 1e-6
    fd = (float(f(x + eps)) - float(f(x - eps))) / (2 * eps)
    assert np.isfinite(g) and g != 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-6)
