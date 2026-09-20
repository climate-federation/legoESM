"""``CLUBBConfig.q_flux_scale``: a moisture-only multiplier on the q_v eddy
diffusivity inside a sigma band of faces (the cloud-base mixing probe).

Shown non-vacuous: with the kernel's ``scale_q_diffusivity_in_band`` call
removed, ``test_scale_changes_only_vapour_inside_the_band`` fails on the
``dq_v_dt`` inequality (the refusal test guards a separate line and is
checked by mutating that guard).
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
    # The default returns the SAME array object, so the solve is untouched.
    col = _l32_column()
    kh = jnp.ones((3, 31))
    assert scale_q_diffusivity_in_band(kh, col["p_half"], CLUBBConfig()) is kh
    assert scale_q_diffusivity_in_band(
        kh, col["p_half"], CLUBBConfig(q_flux_scale=1.0, q_flux_scale_sigma_lo=0.8,
                                       q_flux_scale_sigma_hi=0.95)) is kh


def test_bad_scale_or_band_raises_even_at_scale_one():
    col = _l32_column()
    kh = jnp.ones((3, 31))
    for bad in (0.0, -2.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and > 0"):
            scale_q_diffusivity_in_band(kh, col["p_half"], CLUBBConfig(q_flux_scale=bad))
    with pytest.raises(ValueError, match="0 <= lo < hi <= 1"):
        scale_q_diffusivity_in_band(
            kh, col["p_half"],
            CLUBBConfig(q_flux_scale=1.0, q_flux_scale_sigma_lo=0.9, q_flux_scale_sigma_hi=0.8))


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
    # change in every column sits in a layer ADJACENT to an in-band face
    # (face k separates layers k and k+1 in the top-down interior indexing).
    sigma_face = np.asarray(col["p_half"][0, 1:-1] / col["p_half"][0, -1])
    faces = np.nonzero((sigma_face >= 0.80) & (sigma_face <= 0.95))[0]
    adjacent = set(faces.tolist()) | set((faces + 1).tolist())
    assert 2 <= len(faces) < dq.shape[1] // 2
    for c in range(dq.shape[0]):
        assert int(np.abs(dq[c]).argmax()) in adjacent


def test_band_outside_the_column_is_a_no_op_and_bad_band_raises():
    base, _ = _run(CLUBBConfig())
    col = _l32_column()
    sigma_face = np.asarray(col["p_half"][0, 1:-1] / col["p_half"][0, -1])
    hi = 0.5 * sigma_face.min()   # provably below every interior face
    assert not np.any(sigma_face <= hi)
    off, _ = _run(CLUBBConfig(q_flux_scale=2.5, q_flux_scale_sigma_lo=0.0,
                              q_flux_scale_sigma_hi=hi))
    assert np.array_equal(np.asarray(base.dq_v_dt), np.asarray(off.dq_v_dt))
    with pytest.raises(ValueError, match="0 <= lo < hi <= 1"):
        scale_q_diffusivity_in_band(
            jnp.zeros((3, 31)), col["p_half"],
            CLUBBConfig(q_flux_scale=2.5, q_flux_scale_sigma_lo=0.9, q_flux_scale_sigma_hi=0.8))


def test_prognostic_path_refuses_the_knob():
    """Non-vacuous: deleting the guard at the top of clubb_turbulence_prognostic
    makes this fail with a shape/TypeError from the moment unpacking instead."""
    col = _l32_column()
    with pytest.raises(ValueError, match="diagnostic-CLUBB mechanism probe"):
        clubb_turbulence_prognostic(
            col["u"], col["v"], col["T"], col["q_v"], col["tke"],
            col["p_full"], col["p_half"], col["z_full"], col["z_half"],
            col["T_sfc"], col["q_sfc"], col["rho"], dt=1800.0,
            config=CLUBBConfig(prognostic=True, **_BAND))
