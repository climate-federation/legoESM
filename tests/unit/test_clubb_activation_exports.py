"""Prognostic CLUBB exports the CAM6 droplet-activation inputs (stage S8).

CAM6 hands dropmixnuc/microp_aero CLUBB's ``khzm_out`` of the LAST sub-step's
``advance_clubb_core`` call (clubb_intr.F90:1915, :1997; computed from that call's
input moments) and the end-of-sub-cycle ``wp2`` left in ``WP2_nadv`` (:1987).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    CLUBBConfig,
    cam_chem_tropopause_level,
    clubb_step,
    clubb_turbulence_prognostic,
    flip_vertical,
    pack_clubb_moments,
)

from tests.unit.test_clubb_clubbtop import _args, _column, _spun_up_moments  # noqa: E402

_DT = 300.0


def _prognostic(c, cfg, moments):
    return clubb_turbulence_prognostic(
        *_args(c), pack_clubb_moments(moments), c["p_full"], c["p_half"], c["z_full"],
        c["z_half"], c["T_sfc"], c["q_sfc"], c["rho"], _DT, cfg,
        lat=c["lat"])


def _step(c, state, moments, dt, cfg):
    u, v, T, q = state
    # CAM's tropopause of the HOST (start-of-call) state, as the production wrapper
    troplev = cam_chem_tropopause_level(c["T"], c["p_full"], c["p_half"], c["lat"])
    return clubb_step(u, v, T, q, moments, c["p_full"], c["p_half"], c["z_full"], c["z_half"],
                      c["T_sfc"], c["q_sfc"], c["rho"], dt, cfg, troplev=troplev)


def _reference(c, moments, n_sub, cfg):
    """The sub-cycle by hand: last sub-step's Kh_zm, final wp2."""
    state, dt_sub = _args(c), _DT / n_sub
    for _ in range(n_sub):
        du, dv, dT, dq, moments, diags = _step(c, state, moments, dt_sub, cfg)
        state = tuple(x + dt_sub * t for x, t in zip(state, (du, dv, dT, dq)))
    return flip_vertical(diags["Kh_zm"]), flip_vertical(moments.wp2)


def test_kh_zm_is_the_start_of_call_diffusivity():
    c = _column(30.0)
    cfg = CLUBBConfig(prognostic=True, clubb_dt=_DT)
    m = _spun_up_moments(c)
    *_, m_short, d_short = _step(c, _args(c), m, 1.0, cfg)
    *_, m_long, d_long = _step(c, _args(c), m, _DT, cfg)
    np.testing.assert_array_equal(np.asarray(d_short["Kh_zm"]), np.asarray(d_long["Kh_zm"]))
    assert not np.array_equal(np.asarray(m_short.wp2), np.asarray(m_long.wp2))


def test_exports_are_last_substep_kh_and_final_wp2():
    c = _column(30.0)
    m = _spun_up_moments(c)
    for clubb_dt, n_sub in ((_DT, 1), (_DT / 2, 2)):
        cfg = CLUBBConfig(prognostic=True, clubb_dt=clubb_dt)
        out, carry = _prognostic(c, cfg, m)
        kh_ref, wp2_ref = _reference(c, m, n_sub, cfg)
        nlev = c["T"].shape[1]
        assert out.Kh_iface.shape == out.wp2_iface.shape == (c["T"].shape[0], nlev + 1)
        np.testing.assert_allclose(np.asarray(out.Kh_iface), np.asarray(kh_ref), rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(np.asarray(out.wp2_iface), np.asarray(wp2_ref), rtol=1e-12, atol=0.0)
        assert np.max(np.asarray(out.Kh_iface)[:, -1]) > 0.0, "surface interface is not zeroed (CAM khzm)"
    # The n_sub = 2 export must differ from the first sub-step's values (non-vacuous).
    cfg2 = CLUBBConfig(prognostic=True, clubb_dt=_DT / 2)
    *_, m1, d1 = _step(c, _args(c), m, _DT / 2, cfg2)
    out2, _ = _prognostic(c, cfg2, m)
    assert not np.allclose(np.asarray(out2.Kh_iface), np.asarray(flip_vertical(d1["Kh_zm"])), rtol=1e-6)
    assert not np.allclose(np.asarray(out2.wp2_iface), np.asarray(flip_vertical(m.wp2)), rtol=1e-6)
    assert not np.allclose(np.asarray(out2.wp2_iface), np.asarray(flip_vertical(m1.wp2)), rtol=1e-6)


def test_other_schemes_default_to_none():
    from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
    assert TurbulenceOutput._field_defaults["wp2_iface"] is None
