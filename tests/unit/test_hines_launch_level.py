"""Hines non-orographic GWD must be able to launch ABOVE the boundary layer.

Measured failure (2026-07-30, 2.5 deg AMIP state): the launch amplitude
``total_rms_wind = 2.0 m/s`` exceeds ``sigma_sat = N/m_star`` (1.25 m/s at
140 m) over 78.5% of the planet's area, because N is SMALLEST in the
well-mixed boundary layer.  The wave is therefore born supersaturated and
breaks at its OWN launch level: 55% of its momentum lands below 1 km and
only 35% above 12 km -- the opposite of what a non-orographic scheme should
do.  Removing Hines entirely recovered ~26% of the missing surface wind.

``launch_p`` releases the wave at a chosen pressure instead, depositing no
drag at or below it.  Default 700 hPa; there is no surface launch.
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


def test_default_launches_at_700hpa_not_the_surface():
    cfg = HinesConfig()
    assert cfg.launch_p == pytest.approx(_LAUNCH_P)
    du = np.asarray(_run(cfg).du_dt)
    assert np.allclose(du[:, -3:], 0.0), (
        "the default must deposit no drag in the lowest levels")
    assert np.abs(du).max() > 0.0, "the default must still deposit drag aloft"


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


def test_wave_starts_UNCLIPPED_at_the_launch_level():
    """The defect a drag-output mask alone does NOT fix.

    Zeroing the drag below the launch level still lets the amplitude CARRY
    propagate up through the boundary layer and SATURATE there, so the wave
    arrives at the launch level already clipped by the BL stratification.
    A real launch holds the carry at the launch amplitude until the launch
    level, so the drag ABOVE it cannot depend on the stratification BELOW it.
    """
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _column()
    pmean = np.asarray(p_full).mean(axis=0)
    k = int(np.argmin(np.abs(pmean - _LAUNCH_P)))
    cfg = HinesConfig(launch_p=_LAUNCH_P)

    # Same column, but the levels strictly below the launch level made
    # strongly stratified instead of near-neutral (changes N and rho there).
    T2 = np.asarray(T).copy()
    nlev = T2.shape[1]
    T2[:, k + 1:] = np.linspace(260.0, 300.0, nlev - k - 1)[None, :]
    T2 = jnp.asarray(T2)
    rho2 = p_full / (constants.R_d * T2)
    assert not np.allclose(np.asarray(T)[:, k + 1:], np.asarray(T2)[:, k + 1:])

    a = np.asarray(hines_gwd(u, v, T, p_full, p_half, z_full, z_half, rho,
                             lat, 300.0, cfg).du_dt)
    b = np.asarray(hines_gwd(u, v, T2, p_full, p_half, z_full, z_half, rho2,
                             lat, 300.0, cfg).du_dt)
    assert np.abs(a[:, :k]).max() > 0.0
    np.testing.assert_allclose(a[:, :k], b[:, :k], rtol=1e-12, atol=1e-30,
        err_msg="drag above the launch level depends on the stratification "
                "below it -> the carry was clipped in the BL")


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


def test_param_spec_excludes_launch_p():
    """``launch_p`` picks a level INDEX by argmin, so it is excluded from
    training with a reason, not tuned (the real gate lives in
    tests/test_param_specs.py)."""
    from legoesm.atmosphere.physics.gravity_wave_drag import config as gcfg
    hines = gcfg.__param_spec__["HinesConfig"]
    assert "launch_p" not in hines["params"]
    assert "launch_p" in hines["excluded"]
    # the float knobs it sits beside are still classified
    assert "total_rms_wind" in hines["params"]


def test_driver_rejects_launch_p_outside_the_legal_range():
    """0.0 used to mean "surface launch"; it must now be refused, and the
    driver default must equal the scheme default."""
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        HINES_LAUNCH_P_RANGE_PA,
    )
    from legoesm.driver.config import ExperimentConfig

    assert ExperimentConfig().hines_launch_p == HinesConfig().launch_p
    lo, hi = HINES_LAUNCH_P_RANGE_PA
    for bad in (0.0, None, lo - 1.0, hi + 1.0, float("nan")):
        with pytest.raises(ValueError, match="hines_launch_p"):
            ExperimentConfig(hines_launch_p=bad).validate_strict()
    ExperimentConfig(hines_launch_p=lo).validate_strict()
    ExperimentConfig(hines_launch_p=hi).validate_strict()


def test_no_launch_in_the_lowest_model_layer():
    """A plateau column whose bottom layer straddles ``launch_p`` would put
    the launch level in its lowest layer, i.e. a surface launch: it must get
    no source.  A sea-level column on the same levels keeps its source."""
    ncol, nlev = 2, 10
    sh = np.linspace(0.005, 1.0, nlev + 1)
    ds = np.diff(sh)
    sig = np.cumsum(ds) - 0.5 * ds
    # column 1: lowest full level ~705 hPa, next ~600 hPa -> nearest to
    # 700 hPa is the bottom level although p_s > 700 hPa.
    p_s = np.array([1.0e5, 0.705e5 / sig[-1]])
    assert p_s[1] > _LAUNCH_P
    p_full = jnp.asarray(sig[None, :] * p_s[:, None])
    k_near = np.argmin(np.abs(np.asarray(p_full) - _LAUNCH_P), axis=1)
    assert k_near[1] == nlev - 1 and k_near[0] < nlev - 1
    p_half = jnp.asarray(sh[None, :] * p_s[:, None])
    T = jnp.asarray(np.linspace(215.0, 290.0, nlev)[None, :] * np.ones((ncol, 1)))
    z_half = jnp.asarray(
        _H_SCALE * np.log(_P_SFC / np.maximum(np.asarray(p_half), 1.0)))
    z_full = jnp.asarray(
        _H_SCALE * np.log(_P_SFC / np.maximum(np.asarray(p_full), 1.0)))
    rho = p_full / (constants.R_d * T)
    u = jnp.full((ncol, nlev), 20.0)
    v = jnp.zeros((ncol, nlev))
    du = np.abs(np.asarray(hines_gwd(
        u, v, T, p_full, p_half, z_full, z_half, rho, jnp.zeros(ncol), 300.0,
        HinesConfig(launch_p=_LAUNCH_P)).du_dt))
    assert du[0].max() > 0.0
    assert np.allclose(du[1], 0.0), "launch in the lowest layer must give no source"


def test_driver_rejects_override_launch_p_outside_the_legal_range():
    """A full ``gravity_wave_drag_override`` is used verbatim, so its Hines
    launch level must be range-checked too."""
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig,
    )
    from legoesm.driver.config import ExperimentConfig

    for bad in (0.0, 2.0e5, None):
        ov = GravityWaveDragConfig(scheme="hines",
                                   hines=HinesConfig(launch_p=bad))
        with pytest.raises(ValueError, match="override.hines.launch_p"):
            ExperimentConfig(gravity_wave_drag="hines",
                             gravity_wave_drag_override=ov).validate_strict()
    ov = GravityWaveDragConfig(scheme="hines", hines=HinesConfig())
    ExperimentConfig(gravity_wave_drag="hines",
                     gravity_wave_drag_override=ov).validate_strict()
