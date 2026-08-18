"""CAM's ``trop_cloud_top_press`` applied to the diagnostic CLUBB scheme.

CAM never runs CLUBB above ``trop_cloud_top_lev`` (clubb_intr.F90 slices every
column there; ref_pres: "Troposphere cloud physics will be done only below the
top defined by this pressure"). This port tapers the diffusivities and wp2
production smoothly in log-pressure instead — a hard per-column level cutoff
is a compile-sensitive branch of exactly the class that made two XLA programs
of the same forward diverge on the 32-level WeatherBench arm (2026-08-17
scene-17 dissection). Default 0.0 = off (CAM's code default), byte-inert.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.clubb import (
    CLUBBConfig, clubb_turbulence,
)

_NCOL, _NLEV = 3, 32


def _l32_column(seed=0):
    """A stable L32-like column, top-down, lid ~16 hPa (uniform sigma)."""
    rng = np.random.default_rng(seed)
    sigma_full = (np.arange(_NLEV) + 0.5) / _NLEV            # top-down
    p_s = 1.0e5
    p_full = np.broadcast_to(sigma_full * p_s, (_NCOL, _NLEV)).copy()
    p_half = np.concatenate([[0.0], (np.arange(_NLEV) + 1.0) / _NLEV]) * p_s
    p_half = np.broadcast_to(p_half, (_NCOL, _NLEV + 1)).copy()
    # Hydrostatic-ish heights for a ~280-K scale height.
    z_half = 8000.0 * np.log(p_s / np.clip(p_half, 100.0, None))
    z_full = 8000.0 * np.log(p_s / p_full)
    T = 288.0 - 60.0 * (1.0 - sigma_full)                    # warm sfc, cold top
    T = np.broadcast_to(T, (_NCOL, _NLEV)).copy()
    q_v = np.broadcast_to(1.0e-2 * sigma_full ** 2, (_NCOL, _NLEV)).copy()
    u = np.broadcast_to(20.0 * (1.0 - sigma_full), (_NCOL, _NLEV)).copy()
    v = np.zeros((_NCOL, _NLEV))
    u += rng.normal(0.0, 0.1, u.shape)
    rho = p_full / (constants.R_d * T)
    tke = np.full((_NCOL, _NLEV), 0.1)
    T_sfc = np.full(_NCOL, 289.0)
    q_sfc = np.full(_NCOL, 1.2e-2)
    j = jnp.asarray
    return dict(u=j(u), v=j(v), T=j(T), q_v=j(q_v), tke=j(tke),
                p_full=j(p_full), p_half=j(p_half), z_full=j(z_full),
                z_half=j(z_half), T_sfc=j(T_sfc), q_sfc=j(q_sfc), rho=j(rho))


def _run(config, **overrides):
    col = _l32_column()
    col.update(overrides)
    return clubb_turbulence(
        col["u"], col["v"], col["T"], col["q_v"], col["tke"],
        col["p_full"], col["p_half"], col["z_full"], col["z_half"],
        col["T_sfc"], col["q_sfc"], col["rho"], dt=1800.0, config=config,
    )


def test_default_off_is_bit_identical_to_saturated_taper():
    """Default 0.0 takes the static no-taper branch; a cutoff so small its
    sigmoid saturates to exactly 1.0 at every level must reproduce it
    BIT-identically — proving the off branch is 'taper == 1', nothing else."""
    out_off, wp2_off = _run(CLUBBConfig())
    out_sat, wp2_sat = _run(
        CLUBBConfig(trop_cloud_top_press=1.0e-6))
    for a, b in zip(jax.tree.leaves(out_off), jax.tree.leaves(out_sat)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(wp2_off), np.asarray(wp2_sat))


def test_cutoff_suppresses_mixing_above_and_preserves_below():
    """50 hPa cutoff on the L32 column: diffusivities above the cutoff are
    suppressed by >20x, below 200 hPa they are essentially untouched, and
    the mean-state tendencies above the cutoff collapse accordingly."""
    out_off, _ = _run(CLUBBConfig())
    out_on, _ = _run(CLUBBConfig(trop_cloud_top_press=5000.0))
    p = np.asarray(_l32_column()["p_full"])

    above = p < 3000.0      # < 30 hPa: taper < ~0.03
    below = p > 20000.0     # > 200 hPa: taper > ~0.999
    km_off = np.asarray(out_off.Km)
    km_on = np.asarray(out_on.Km)
    # Suppressed above (guard the all-zero case: suppressed or already zero).
    assert np.all(km_on[above] <= 0.05 * km_off[above] + 1e-12)
    # Preserved below.
    np.testing.assert_allclose(km_on[below], km_off[below], rtol=1e-3)
    # Tendencies above the cutoff are tiny relative to the scheme's own
    # below-cutoff activity.
    dT_on = np.abs(np.asarray(out_on.dT_dt))
    scale = max(float(np.max(np.abs(np.asarray(out_on.dT_dt)[below]))), 1e-12)
    assert float(np.max(dT_on[above])) < 0.05 * scale
    # The PDF cloud fraction is tapered too (an arm with cloud_scheme="clubb"
    # must not consume stratospheric PDF cloud from a region the mixing no
    # longer maintains). The output is TOP-DOWN like every other field (the
    # ascending array used to be handed out raw — codex P1, fixed).
    cf_on = np.asarray(out_on.cloud_fraction)
    cf_off = np.asarray(out_off.cloud_fraction)
    assert np.all(cf_on[above] <= 0.05 * cf_off[above] + 1e-12)


def test_cloud_fraction_output_is_top_down():
    """Orientation contract: the host consumes cloud_fraction in the same
    top-down layout as T/q (radiation reshapes it into the T column). A
    boundary-layer-cloudy, clear-stratosphere column must come out with its
    cloud in the LAST (near-surface) indices, not the first."""
    out, _ = _run(CLUBBConfig())
    cf = np.asarray(out.cloud_fraction)
    nlev = cf.shape[1]
    top_half = float(np.max(cf[:, : nlev // 2]))
    bottom_half = float(np.max(cf[:, nlev // 2:]))
    # The test column saturates only near the surface; any cloud the PDF
    # diagnoses must sit bottom-half in a top-down array.
    assert bottom_half >= top_half


def test_cutoff_kills_top_interface_mixing_absolutely():
    """Codex P1: averaging tapered full-level K left ~20% of uncapped mixing
    at the 16-47 hPa interface. The interface taper must suppress the
    ABSOLUTE top-level tendencies of all four mean fields and wp2, compared
    with the uncapped run — not merely relative to below-cutoff activity."""
    out_off, wp2_off = _run(CLUBBConfig())
    out_on, wp2_on = _run(CLUBBConfig(trop_cloud_top_press=5000.0))
    for name in ("dT_dt", "du_dt", "dv_dt", "dq_v_dt"):
        t_off = float(np.max(np.abs(np.asarray(getattr(out_off, name))[:, 0])))
        t_on = float(np.max(np.abs(np.asarray(getattr(out_on, name))[:, 0])))
        assert t_on <= 0.05 * t_off + 1e-15, (name, t_on, t_off)
    # wp2 at the top level relaxes toward its floor (production tapered,
    # dissipation active): one 1800-s step must collapse the 0.1 seed by
    # >2 orders. (No cross-run comparison: the uncapped run removes top wp2
    # FASTER via diffusion, so wp2_on <= wp2_off is not the invariant.)
    del wp2_off
    assert float(np.max(np.asarray(wp2_on)[:, 0])) < 1.0e-3


def test_taper_is_differentiable_and_finite():
    """The taper sits inside the training adjoint; gradients through the
    scheme w.r.t. the state must stay finite with the cutoff active."""
    col = _l32_column()
    cfg = CLUBBConfig(trop_cloud_top_press=5000.0)

    def f(T):
        out, wp2 = clubb_turbulence(
            col["u"], col["v"], T, col["q_v"], col["tke"],
            col["p_full"], col["p_half"], col["z_full"], col["z_half"],
            col["T_sfc"], col["q_sfc"], col["rho"], dt=1800.0, config=cfg,
        )
        return jnp.sum(out.dT_dt ** 2) + jnp.sum(wp2)

    g = jax.grad(f)(col["T"])
    assert bool(jnp.all(jnp.isfinite(g)))


def test_factory_threads_the_cam_override():
    """The classical factory's clubb_top_press kwarg must land on
    CLUBBConfig.trop_cloud_top_press (the WB YAML's clubb_top_press_hpa
    plumbs through scale_build into this kwarg)."""
    import inspect

    from legoesm.training import aimip_params as ap

    sig = inspect.signature(ap.make_aimip_classical_spectral_physics)
    assert "clubb_top_press" in sig.parameters
    assert sig.parameters["clubb_top_press"].default is None
    src = inspect.getsource(ap.make_aimip_classical_spectral_physics)
    assert "trop_cloud_top_press=float(clubb_top_press)" in src
    # And the WB loader-side key exists in scale_build's physics branch.
    from legoesm.training import scale_build as sb
    assert "clubb_top_press_hpa" in inspect.getsource(sb)
