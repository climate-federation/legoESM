"""``CLUBBConfig.q_flux_scale``: a moisture-only multiplier on the q_v eddy
diffusivity inside a sigma band of faces (the cloud-base mixing probe).

Shown non-vacuous: with the kernel's ``scale_q_diffusivity_in_band`` call
removed, ``test_scale_changes_only_vapour_inside_the_band`` fails on the
``dq_v_dt`` inequality and the prognostic refusal test fails.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.turbulence.clubb import (
    CLUBBConfig,
    clubb_turbulence_prognostic,
    scale_q_diffusivity_in_band,
)
from tests.unit.test_clubb_trop_top import _l32_column, _run

_BAND = dict(q_flux_scale=2.5, q_flux_scale_sigma_lo=0.80, q_flux_scale_sigma_hi=0.95)


def test_default_is_byte_identical():
    base, _ = _run(CLUBBConfig())
    same, _ = _run(CLUBBConfig(q_flux_scale=1.0, q_flux_scale_sigma_lo=0.8,
                               q_flux_scale_sigma_hi=0.95))
    assert np.array_equal(np.asarray(base.dq_v_dt), np.asarray(same.dq_v_dt))


def test_scale_changes_only_vapour_inside_the_band():
    base, _ = _run(CLUBBConfig())
    scaled, _ = _run(CLUBBConfig(**_BAND))
    # T, momentum and the surface fluxes are untouched: moisture-only.
    for f in ("dT_dt", "du_dt", "dv_dt", "lhflx", "shflx", "Kh"):
        assert np.array_equal(np.asarray(getattr(base, f)), np.asarray(getattr(scaled, f))), f
    dq = np.asarray(scaled.dq_v_dt) - np.asarray(base.dq_v_dt)
    assert np.abs(dq).max() > 0.0
    # Flux form: the column-integrated vapour tendency is unchanged (the
    # surface source is the same, the interior faces only redistribute).
    # The solver's mass weight is rho * dz (top-down: z_half decreases).
    col = _l32_column()
    w = np.asarray(col["rho"]) * np.asarray(col["z_half"][:, :-1] - col["z_half"][:, 1:])
    gross = np.abs(np.asarray(base.dq_v_dt) * w).sum(1)
    assert np.all(np.abs((dq * w).sum(1)) < 1e-5 * gross)
    # The implicit solve couples every level, so the response is not exactly
    # zero away from the band; it must peak at the scaled faces: the largest
    # change in every column sits in a layer touching the band.
    sigma_full = np.asarray(col["p_full"][0] / col["p_half"][0, -1])
    for c in range(dq.shape[0]):
        assert 0.75 <= sigma_full[np.abs(dq[c]).argmax()] <= 1.0


def test_band_outside_the_column_is_a_no_op_and_bad_band_raises():
    base, _ = _run(CLUBBConfig())
    off, _ = _run(CLUBBConfig(q_flux_scale=2.5, q_flux_scale_sigma_lo=0.0,
                              q_flux_scale_sigma_hi=0.0005))
    assert np.array_equal(np.asarray(base.dq_v_dt), np.asarray(off.dq_v_dt))
    col = _l32_column()
    with pytest.raises(ValueError, match="0 <= lo < hi <= 1"):
        scale_q_diffusivity_in_band(
            jnp.zeros((3, 31)), col["p_half"],
            CLUBBConfig(q_flux_scale=2.5, q_flux_scale_sigma_lo=0.9, q_flux_scale_sigma_hi=0.8))


def test_prognostic_path_refuses_the_knob():
    col = _l32_column()
    with pytest.raises(ValueError, match="diagnostic-CLUBB mechanism probe"):
        clubb_turbulence_prognostic(
            col["u"], col["v"], col["T"], col["q_v"], col["tke"],
            col["p_full"], col["p_half"], col["z_full"], col["z_half"],
            col["T_sfc"], col["q_sfc"], col["rho"], dt=1800.0,
            config=CLUBBConfig(prognostic=True, **_BAND))
