"""Hines non-orographic GWD must be able to launch ABOVE the boundary layer.

Measured failure (2026-07-30, 2.5 deg AMIP state): the launch amplitude
``total_rms_wind = 2.0 m/s`` exceeds ``sigma_sat = N/m_star`` (1.25 m/s at
140 m) over 78.5% of the planet's area, because N is SMALLEST in the
well-mixed boundary layer.  The wave is therefore born supersaturated and
breaks at its OWN launch level: 55% of its momentum lands below 1 km and
only 35% above 12 km -- the opposite of what a non-orographic scheme should
do.  Removing Hines entirely recovered ~26% of the missing surface wind.

``launch_p`` releases the wave at a chosen pressure instead, depositing no
drag at or below it.  Default ``None`` keeps the legacy surface launch.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig
from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd

_P_SFC = 1.0e5          # reference surface pressure for the synthetic column
_H_SCALE = 8000.0       # scale height for the z(p) proxy [m]
_LAUNCH_P = 7.0e4       # 700 hPa test launch level


def _column(ncol=3, nlev=30):
    """Top-down column: index 0 = model top, nlev-1 = surface."""
    p_s = np.full(ncol, _P_SFC)
    sh = np.linspace(0.005, 1.0, nlev + 1)
    ds = np.diff(sh)
    sig = np.cumsum(ds) - 0.5 * ds
    p_full = jnp.asarray(sig[None, :] * p_s[:, None])
    p_half = jnp.asarray(sh[None, :] * p_s[:, None])
    # Weakly stratified BL (small N) under a stratified free troposphere —
    # the configuration that makes a surface launch supersaturate.
    T = np.linspace(215.0, 290.0, nlev)
    T[-4:] = 290.0                       # near-neutral bottom layers
    T = jnp.asarray(T[None, :] * np.ones((ncol, 1)))
    z_half = jnp.asarray(
        _H_SCALE * np.log(_P_SFC / np.maximum(np.asarray(p_half), 1.0)))
    z_full = jnp.asarray(
        _H_SCALE * np.log(_P_SFC / np.maximum(np.asarray(p_full), 1.0)))
    rho = p_full / (constants.R_d * T)
    u = jnp.full((ncol, nlev), 20.0)
    v = jnp.zeros((ncol, nlev))
    lat = jnp.zeros(ncol)
    return u, v, T, p_full, p_half, z_full, z_half, rho, lat


def _run(cfg):
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _column()
    return hines_gwd(u, v, T, p_full, p_half, z_full, z_half, rho,
                     lat, 300.0, cfg)


def test_default_is_none_and_launches_at_the_surface():
    cfg = HinesConfig()
    assert cfg.launch_p is None
    du = np.asarray(_run(cfg).du_dt)
    # Legacy behaviour: the bottom levels receive drag.
    assert np.abs(du[:, -3:]).max() > 0.0, (
        "surface launch should deposit drag in the lowest levels")


def test_launch_level_zeroes_drag_at_and_below_it():
    """No drag at or below the launch level; the column above still gets it."""
    out = _run(HinesConfig(launch_p=_LAUNCH_P))
    du = np.asarray(out.du_dt)
    _, _, _, p_full, *_ = _column()
    pmean = np.asarray(p_full).mean(axis=0)
    k_launch = int(np.argmin(np.abs(pmean - _LAUNCH_P)))
    assert np.allclose(du[:, k_launch:], 0.0), (
        f"drag must vanish at/below the launch level k={k_launch} "
        f"(p={pmean[k_launch]:.0f} Pa)")
    assert np.abs(du[:, :k_launch]).max() > 0.0, (
        "the column above the launch level must still receive drag")


def test_launch_level_moves_the_deposition_upward():
    """The physical point: a launch level shifts momentum deposition OUT of
    the boundary layer."""
    _, _, _, _, p_half, *_ = _column()
    # layer mass per unit area, dp/g
    dz = np.abs(np.diff(np.asarray(p_half), axis=1)) / constants.g
    lo = slice(-6, None)                                        # lowest levels

    du_sfc = np.abs(np.asarray(_run(HinesConfig()).du_dt))
    du_lch = np.abs(np.asarray(_run(HinesConfig(launch_p=_LAUNCH_P)).du_dt))
    frac_sfc = (du_sfc[:, lo] * dz[:, lo]).sum() / max(
        (du_sfc * dz).sum(), 1e-30)
    frac_lch = (du_lch[:, lo] * dz[:, lo]).sum() / max(
        (du_lch * dz).sum(), 1e-30)
    assert frac_lch < frac_sfc, (
        f"launch level must reduce the low-level share of the drag "
        f"(surface launch {frac_sfc:.3f} -> launched {frac_lch:.3f})")
    assert frac_lch == pytest.approx(0.0, abs=1e-12)


def test_wave_starts_UNCLIPPED_at_the_launch_level():
    """The defect a drag-output mask alone does NOT fix.

    Zeroing the drag below the launch level still lets the amplitude CARRY
    propagate up through the boundary layer and SATURATE there, so the wave
    arrives at the launch level already clipped and the drag ALOFT is
    bit-identical to a surface launch — the launch level would be cosmetic.
    A real launch holds the carry at the launch amplitude until the launch
    level, so the wave starts there unclipped and deposits MORE aloft.
    """
    _, _, _, p_full, *_ = _column()
    pmean = np.asarray(p_full).mean(axis=0)
    k = int(np.argmin(np.abs(pmean - _LAUNCH_P)))

    sfc = np.abs(np.asarray(_run(HinesConfig()).du_dt))
    lch = np.abs(np.asarray(_run(HinesConfig(launch_p=_LAUNCH_P)).du_dt))
    above_sfc = sfc[:, :k].sum()
    above_lch = lch[:, :k].sum()
    assert not np.allclose(sfc[:, :k], lch[:, :k], rtol=1e-12, atol=1e-30), (
        "drag above the launch level is bit-identical to the surface launch "
        "-> the carry was still clipped in the BL (output-mask-only bug)")
    assert above_lch > above_sfc, (
        f"an unclipped launch must deposit MORE drag aloft "
        f"({above_lch:.4e} vs surface-launch {above_sfc:.4e})")


def test_launch_level_is_PER_COLUMN_not_a_global_index():
    """``launch_p`` is documented in [Pa], so a nominal 700 hPa source must sit
    at 700 hPa over a MOUNTAIN as well as over the ocean.  A column-mean
    profile would pick ONE global level index and make the threshold a
    reference-grid convention instead of a pressure (codex 2026-07-31)."""
    ncol, nlev = 3, 30
    # column 0 = sea level, column 1 = elevated, column 2 = high terrain
    p_s = np.array([1.0e5, 8.0e4, 6.5e4])
    sh = np.linspace(0.005, 1.0, nlev + 1)
    ds = np.diff(sh)
    sig = np.cumsum(ds) - 0.5 * ds
    p_full = jnp.asarray(sig[None, :] * p_s[:, None])
    p_half = jnp.asarray(sh[None, :] * p_s[:, None])
    T = jnp.asarray(np.linspace(215.0, 290.0, nlev)[None, :]
                    * np.ones((ncol, 1)))
    z_half = jnp.asarray(
        _H_SCALE * np.log(_P_SFC / np.maximum(np.asarray(p_half), 1.0)))
    z_full = jnp.asarray(
        _H_SCALE * np.log(_P_SFC / np.maximum(np.asarray(p_full), 1.0)))
    rho = p_full / (constants.R_d * T)
    u = jnp.full((ncol, nlev), 20.0)
    v = jnp.zeros((ncol, nlev))
    out = hines_gwd(u, v, T, p_full, p_half, z_full, z_half, rho,
                    jnp.zeros(ncol), 300.0, HinesConfig(launch_p=_LAUNCH_P))
    du = np.abs(np.asarray(out.du_dt))
    pf = np.asarray(p_full)

    # columns 0 and 1 have surface pressure above 700 hPa -> a real source,
    # and the deepest level receiving drag must sit at ~launch_p in EACH.
    for j in (0, 1):
        active = np.nonzero(du[j] > 0)[0]
        assert active.size > 0, f"column {j} should have a source"
        p_lowest_active = pf[j, active.max()]
        assert abs(p_lowest_active - _LAUNCH_P) < 0.12 * _LAUNCH_P, (
            f"column {j}: lowest active level at {p_lowest_active:.0f} Pa, "
            f"not near launch_p={_LAUNCH_P:.0f} Pa -> index is not per-column")

    # column 2 lies entirely above launch_p (p_s = 650 hPa) -> NO source,
    # rather than a silently relocated one.
    assert np.allclose(du[2], 0.0), (
        "a column whose surface pressure is below launch_p must get no source")


def test_jit_parity_and_finite():
    """Eager vs jit must agree (the level index is a traced argmin, not an
    int(), so the kernel has to stay jit-safe)."""
    cfg = HinesConfig(launch_p=_LAUNCH_P)
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _column()

    def f(*a):
        return hines_gwd(*a, 300.0, cfg).du_dt

    args = (u, v, T, p_full, p_half, z_full, z_half, rho, lat)
    eager = np.asarray(f(*args))
    jitted = np.asarray(jax.jit(f)(*args))
    assert np.isfinite(eager).all() and np.isfinite(jitted).all()
    np.testing.assert_allclose(eager, jitted, rtol=1e-6, atol=1e-12)


def test_param_spec_omits_launch_p_and_stays_valid():
    """``launch_p`` is ``float | None``, so it is NOT spec-eligible: the gate
    computes eligibility from a plain ``: float`` annotation and rejects a
    non-float named in EITHER params or excluded.  Assert it is in neither,
    and that the module's spec still validates (the real gate lives in
    tests/test_param_specs.py)."""
    from legoesm.atmosphere.physics.gravity_wave_drag import config as gcfg
    hines = gcfg.__param_spec__["HinesConfig"]
    assert "launch_p" not in hines["params"]
    assert "launch_p" not in hines["excluded"]
    # the float knobs it sits beside are still classified
    assert "total_rms_wind" in hines["params"]
