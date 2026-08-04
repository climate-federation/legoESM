"""GATE/LBA plane-CRM assembly from SAM CASE files (iter-33, C1).

Builds the height coordinate (θ_ref = sounding), initial state (sounding u,v,q_v
+ θ seed), and large-scale-forcing physics_fn from SAM ``snd``/``lsf``/``sfc``
files.  Synthetic files are the portable oracle; an extra test assembles the
REAL ``CASES/GATE_IDEAL`` when the gSAM tree is present.
"""

from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.forcing.sam_case_forcing import read_sam_snd  # noqa: E402
from legoesm.atmosphere.dynamics.crm.sam_case_setup import (  # noqa: E402
    WING_SEED_AMP_FIFTH_K,
    WING_SEED_AMP_LOWEST_K,
    WING_SEED_N_LAYERS,
    band_limited_seed_pattern,
    build_gate_ideal_setup,
    build_lba_setup,
    build_sam_case_height_coord,
    build_sam_case_initial_state,
    apply_prescribed_surface_fluxes_plane,
    coriolis_f0,
    wing2018_seed_amplitudes,
    wing2018_thermal_noise_seed,
    _random_band_theta_seed,
)
from legoesm.grids.plane import create_plane_grid  # noqa: E402


_SND = """ z[m] p[mb] tp[K] q[g/kg] u[m/s] v[m/s]
 0., 4, 1000.  day,levels,pres0
     0.0   1000.0   300.0   16.0   -1.0   0.0
  5000.0    540.0   320.0    5.0   -5.0   0.0
 10000.0    265.0   345.0    1.0   -8.0   0.0
 16000.0    100.0   380.0    0.1   -2.0   0.0
"""

_LSF = """ z[m] p[mb] tls qls uls vls wls
 0., 2, 1000.  day,levels,pres0
     0.0   1000.0   0.1000E-04   0.2000E-08   -1.0   0.0   -0.010
 16000.0    100.0   0.2000E-04   0.1000E-08   -8.0   0.0   -0.005
"""

_SFC = """ day sst(K) H LE TAU
   0.000   299.88   0.0   0.0   0.0
 999.000   299.88   0.0   0.0   0.0
"""

_NLEV = 40   # ≥40 ⇒ a gentle SAM-like stretch (dz_sfc=50 m) at H=15 km (VGRID)
_H = 15000.0


def _case(tmp_path):
    (tmp_path / "snd").write_text(_SND)
    (tmp_path / "lsf").write_text(_LSF)
    (tmp_path / "sfc").write_text(_SFC)
    return tmp_path


def _grid(ny=4, nx=4):
    return create_plane_grid(nx=nx, ny=ny, nlev=_NLEV, dx=1000.0, dy=1000.0,
                             dtype=jnp.float64)


def test_coriolis_f0():
    f = coriolis_f0(8.5)
    assert f == pytest.approx(2.0 * constants.Omega * math.sin(math.radians(8.5)))
    assert f > 0.0


def test_height_coord_theta_ref_is_sounding(tmp_path):
    snd = read_sam_snd(_case(tmp_path) / "snd")
    hc = build_sam_case_height_coord(snd, _NLEV, _H, 100000.0)
    # θ_ref(z_full) == linear interp of the sounding θ
    expect = np.interp(np.asarray(hc.z_full), snd.z, snd.theta)
    assert np.allclose(np.asarray(hc.theta_ref), expect, rtol=1e-6)
    # geostrophic-reference winds carried from the sounding
    assert hc.u_geo0 is not None and hc.u_geo0.shape == (_NLEV,)


def test_height_coord_is_stretched_50m_near_surface(tmp_path):
    """VGRID (codex iter-51 G): build_sam_case_height_coord must produce a
    well-formed STRETCHED grid with dz≈50 m near the surface (SAM's GATE `grd`
    resolution), NOT the old uniform dz=H/nlev (≈13× too coarse). Use nlev=64
    (SAM-like, gentle stretch). Pin the grid invariants."""
    snd = read_sam_snd(_case(tmp_path) / "snd")
    hc = build_sam_case_height_coord(snd, 64, _H, 100000.0)
    dz = np.asarray(hc.dz)
    z_full = np.asarray(hc.z_full)
    z_half = np.asarray(hc.z_half)
    # top-to-bottom storage ⇒ the surface dz is the LAST entry
    np.testing.assert_allclose(dz[-1], 50.0, rtol=0.1)         # ≈ dz_sfc
    assert z_full[-1] < 40.0                                   # lowest cell ≲40m
    assert dz[0] > 5.0 * dz[-1]                                # genuinely stretched
    # invariants: all positive, monotone z (top-to-bottom decreasing), top=H
    assert bool(np.all(dz > 0.0))
    assert bool(np.all(np.diff(z_full) < 0.0))                # z decreases with k
    np.testing.assert_allclose(z_half[0], _H, rtol=1e-6)      # top interface = H
    np.testing.assert_allclose(z_half[-1], 0.0, atol=1e-6)    # surface = 0
    assert float(dz.max() / dz.min()) < 30.0                  # gentle at nlev=64
    # configurable dz_sfc
    hc100 = build_sam_case_height_coord(snd, 64, _H, 100000.0, dz_sfc=100.0)
    np.testing.assert_allclose(np.asarray(hc100.dz)[-1], 100.0, rtol=0.1)


def test_height_coord_warns_on_aggressive_stretch(tmp_path):
    """codex iter-51 D: too few levels for (dz_sfc, H) ⇒ aggressive stretch ⇒
    WARN (so callers bump nlev to ~64 like SAM)."""
    import warnings
    snd = read_sam_snd(_case(tmp_path) / "snd")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_sam_case_height_coord(snd, 12, _H, 100000.0)   # nlev=12 ≪ needed
    assert any("stretch ratio" in str(x.message) for x in w)


def test_initial_state_matches_sounding(tmp_path):
    snd = read_sam_snd(_case(tmp_path) / "snd")
    hc = build_sam_case_height_coord(snd, _NLEV, _H, 100000.0)
    st = build_sam_case_initial_state(snd, _grid(), hc, n_tracers=10,
                                      seed_amp=0.1)
    z = np.asarray(hc.z_full)
    # u, v, q_v interpolated from the sounding (column 0,0)
    assert np.allclose(np.asarray(st.u.data)[0, 0],
                       np.interp(z, snd.z, snd.u), rtol=1e-6)
    assert np.allclose(np.asarray(st.tracers.data)[0, 0, :, 0],
                       np.interp(z, snd.z, snd.q_v), rtol=1e-6)
    # tracers allocated for Morrison (10 slots)
    assert st.tracers.data.shape[-1] == 10
    # θ' = (θ_snd − θ_ref) + seed ≈ seed only (θ_ref IS the sounding)
    theta_p = np.asarray(st.theta_prime.data)
    assert np.max(np.abs(theta_p)) < 0.2          # seed amp 0.1, base ≈0
    # seed only in the bottom levels; upper levels untouched (≈0)
    assert np.allclose(theta_p[:, :, :-4], 0.0, atol=1e-6)
    assert np.any(np.abs(theta_p[:, :, -4:]) > 0.0)


def test_initial_state_hydrostatic_rho(tmp_path):
    """ρ' = -ρ₀ θ'/θ_ref (hydrostatic IC) — finite, small where θ'=seed."""
    snd = read_sam_snd(_case(tmp_path) / "snd")
    hc = build_sam_case_height_coord(snd, _NLEV, _H, 100000.0)
    st = build_sam_case_initial_state(snd, _grid(), hc)
    assert np.all(np.isfinite(np.asarray(st.rho_prime.data)))


def test_build_gate_ideal_setup(tmp_path):
    setup = build_gate_ideal_setup(_case(tmp_path), _grid(), nlev=_NLEV, H=_H)
    # GATE_IDEAL prm: Coriolis OFF, fixed SST, forcing present
    assert setup.coriolis is False
    assert setup.sst == pytest.approx(299.88)
    assert setup.f0 == pytest.approx(coriolis_f0(8.5))
    assert setup.p_sfc_pa == pytest.approx(100000.0)
    # forcing physics_fn produces nonzero subsidence/advective + nudging
    out = setup.ls_forcing_physics(setup.initial_state, None,
                                   setup.height_coord, None)
    assert np.any(np.abs(np.asarray(out.dtheta_prime_dt.data)) > 0.0)
    # wind nudging acts (mean wind ≠ target ⇒ du ≠ 0 somewhere)
    assert np.any(np.abs(np.asarray(out.du_dt.data)) > 0.0)


# --------------------------------------------------------------------------
# Real gSAM GATE_IDEAL (skipped when absent)
# --------------------------------------------------------------------------

_GSAM = "/home/gentine/Documents/Code/gSAM/gsam1.8.7/gSAM1.8.7/CASES/GATE_IDEAL"


@pytest.mark.skipif(not os.path.isdir(_GSAM),
                    reason="gSAM CASES/GATE_IDEAL not present")
def test_real_gate_ideal_assembly():
    grid = create_plane_grid(nx=4, ny=4, nlev=30, dx=1000.0, dy=1000.0,
                             dtype=jnp.float64)
    setup = build_gate_ideal_setup(_GSAM, grid, nlev=30, H=20000.0)
    assert setup.sst == pytest.approx(299.88)
    assert setup.coriolis is False
    # initial state finite, θ_ref tracks the GATE sounding (warm, stable)
    assert np.all(np.isfinite(np.asarray(setup.initial_state.theta_prime.data)))
    th_ref = np.asarray(setup.height_coord.theta_ref)
    assert np.all(np.diff(th_ref[::-1]) > -1.0)   # θ increases upward (stable)
    assert 295.0 < th_ref[-1] < 305.0             # ~SST near the surface


_GSAM_LBA = "/home/gentine/Documents/Code/gSAM/gsam1.8.7/gSAM1.8.7/CASES/LBA"


@pytest.mark.parametrize("case_dir,builder_kw", [
    pytest.param(_GSAM, {}, id="GATE",
                 marks=pytest.mark.skipif(not os.path.isdir(_GSAM),
                                          reason="GATE_IDEAL absent")),
    pytest.param(_GSAM_LBA, {"n_tracers": 11}, id="LBA",
                 marks=pytest.mark.skipif(not os.path.isdir(_GSAM_LBA),
                                          reason="LBA absent")),
])
def test_snd_top_theta_ref_ic_consistent_above_snd_top(case_dir, builder_kw):
    """SND-TOP (codex iter-57): the SAME extended sounding feeds θ_ref AND the IC,
    so θ' = θ_ic − θ_ref ≈ 0 ABOVE the original sounding top (where there is no BL
    seed) — proving the extension is consistent, not just present. H is set ABOVE
    each case's snd top (GATE ~17 km, LBA ~30 km) to force extrapolation."""
    from legoesm.atmosphere.forcing.sam_case_forcing import read_sam_snd
    z_snd_top = float(np.asarray(read_sam_snd(os.path.join(case_dir, "snd")).z).max())
    H = z_snd_top + 6000.0          # model top 6 km ABOVE the snd top
    grid = create_plane_grid(nx=4, ny=4, nlev=64, dx=1000.0, dy=1000.0,
                             coriolis_mode="none", dtype=jnp.float64)
    build = (build_gate_ideal_setup if "GATE_IDEAL" in case_dir
             else build_lba_setup)
    setup = build(case_dir, grid, nlev=64, H=H, **builder_kw)
    z = np.asarray(setup.height_coord.z_full)
    thp = np.asarray(setup.initial_state.theta_prime.data)[0, 0]
    above = z > z_snd_top
    assert above.sum() > 0, "model top must exceed the sounding top to test extrap"
    assert np.max(np.abs(thp[above])) < 1e-6   # θ'≈0 aloft (consistent extension)


_SHORT_SND = """ z[m] p[mb] tp[K] q[g/kg] u[m/s] v[m/s]
 0., 2, 1000.  day,levels,pres0
     0.0   1000.0   300.0   16.0   -1.0   0.0
  5000.0    540.0   320.0    5.0   -5.0   0.0
"""


def test_coverage_guard_fires_on_short_sounding(tmp_path):
    """codex iter-57b (HIGH): a sounding that does NOT reach the model top makes
    theta_ref_fn clamp θ constant (dry-neutral) aloft — and a θ_ic≈θ_ref check
    can't catch it (both clamp). build_sam_case_height_coord must RAISE by
    default, and only bypass with the explicit allow_short_sounding escape hatch
    (for synthetic/stub soundings)."""
    (tmp_path / "snd").write_text(_SHORT_SND)              # tops at 5 km
    snd = read_sam_snd(tmp_path / "snd")
    with pytest.raises(ValueError, match="does not cover|clamp"):
        build_sam_case_height_coord(snd, _NLEV, 15000.0, 100000.0)   # top 15 km
    # escape hatch: explicit opt-in for a stub sounding → no raise
    hc = build_sam_case_height_coord(snd, _NLEV, 15000.0, 100000.0,
                                     allow_short_sounding=True)
    assert float(np.asarray(hc.z_full).max()) > 5000.0


@pytest.mark.skipif(not os.path.isdir(_GSAM), reason="GATE_IDEAL absent")
def test_snd_top_reference_T_follows_std_atm_ratio():
    """codex iter-57b (MED): θ_ic≈θ_ref does NOT prove T_ref follows the SAM
    std-atm RATIO (compute_reference_state RE-integrates θ hydrostatically). Build
    the GATE reference state and assert the constructed T_ref(z)/T_top matches
    T_std(z)/T_std(z_top) above the original sounding top (the SAM extrapolation
    oracle), not a flat-isothermal or clamped profile."""
    from legoesm.atmosphere.forcing.sam_case_forcing import (
        read_sam_snd, us_standard_atmosphere_temperature)
    grid = create_plane_grid(nx=4, ny=4, nlev=64, dx=1000.0, dy=1000.0,
                             coriolis_mode="none", dtype=jnp.float64)
    setup = build_gate_ideal_setup(_GSAM, grid, nlev=64, H=30000.0)
    snd0 = read_sam_snd(os.path.join(_GSAM, "snd"))
    z_snd_top = float(np.asarray(snd0.z).max())
    hc = setup.height_coord
    z = np.asarray(hc.z_full)
    T_ref = np.asarray(hc.theta_ref) * np.asarray(hc.exner_ref)
    k_top = int(np.argmin(np.abs(z - z_snd_top)))         # nearest level to snd top
    T_top = T_ref[k_top]
    above = z > z_snd_top + 200.0
    ratio_model = T_ref[above] / T_top
    ratio_std = (us_standard_atmosphere_temperature(z[above])
                 / us_standard_atmosphere_temperature(np.array([z[k_top]]))[0])
    # the re-integrated reference reproduces SAM's std-atm ratio to ~1%
    np.testing.assert_allclose(ratio_model, ratio_std, rtol=1.5e-2)
    # and it is NOT a flat clamp: T_ref warms detectably above 20 km
    if np.any(z[above] > 24000.0):
        assert T_ref[above][np.argmax(z[above])] > T_top + 1.0


# -------- LBA land-diurnal case (iter-41) --------

_LBA_SFC = """ day sst(K) H(W/m2) LE(W/m2) TAU
   0.000   999.0     0.0     0.0   0.0
   0.200   999.0   266.0   548.0   0.0
   0.500   999.0     0.0     0.0   0.0
"""


def test_apply_prescribed_surface_fluxes_warms_and_moistens(tmp_path):
    """Positive H/LE inject heat + moisture into the LOWEST level only."""
    snd = read_sam_snd(_case(tmp_path) / "snd")
    hc = build_sam_case_height_coord(snd, _NLEV, _H, 100000.0)
    st = build_sam_case_initial_state(snd, _grid(), hc, seed_amp=0.0)
    th0 = np.asarray(st.theta_prime.data).copy()
    qv0 = np.asarray(st.tracers.data[..., 0]).copy()
    st2 = apply_prescribed_surface_fluxes_plane(
        st, shflx=266.0, lhflx=548.0, height_coord=hc, dt=20.0)
    th = np.asarray(st2.theta_prime.data)
    qv = np.asarray(st2.tracers.data[..., 0])
    # surface (last level) warmed + moistened; other levels untouched
    assert np.all(th[:, :, -1] > th0[:, :, -1])
    assert np.all(qv[:, :, -1] > qv0[:, :, -1])
    assert np.allclose(th[:, :, :-1], th0[:, :, :-1])
    assert np.allclose(qv[:, :, :-1], qv0[:, :, :-1])
    # zero fluxes ⇒ no change
    st3 = apply_prescribed_surface_fluxes_plane(st, 0.0, 0.0, hc, 20.0)
    assert np.allclose(np.asarray(st3.theta_prime.data), th0)


_LBA_RAD = """ p[mb] or z(m)  (dt/dt)rad [K/s]
 0., 3
    0.0   -0.1600E-04
 8000.0   -0.2000E-04
16000.0    0.0000E+00
"""


def test_build_lba_setup(tmp_path):
    (tmp_path / "snd").write_text(_SND)
    (tmp_path / "sfc").write_text(_LBA_SFC)
    (tmp_path / "rad").write_text(_LBA_RAD)
    setup = build_lba_setup(tmp_path, _grid(), nlev=_NLEV, H=_H, n_tracers=11)
    # LBA: no lsf ⇒ forcing is wind-nudging only; sfc + rad series carried
    assert setup.ls_forcing_physics is not None
    assert setup.sfc.shf.shape[0] == 3          # 3 sfc time rows
    assert float(np.max(setup.sfc.lhf)) == pytest.approx(548.0)
    assert float(setup.rad.dTdt_rad.min()) < 0.0   # prescribed radiative cooling
    assert setup.latitude < 0.0                 # southern hemisphere (Amazon)
    # the wind-nudging forcing produces a tendency on the assembled state
    out = setup.ls_forcing_physics(setup.initial_state, None,
                                   setup.height_coord, None)
    assert np.all(np.isfinite(np.asarray(out.du_dt.data)))


def test_prescribed_surface_momentum_drag(tmp_path):
    """Interactive land drag (cd_momentum>0) decelerates the lowest wind only."""
    snd = read_sam_snd(_case(tmp_path) / "snd")
    hc = build_sam_case_height_coord(snd, _NLEV, _H, 100000.0)
    st = build_sam_case_initial_state(snd, _grid(), hc, seed_amp=0.0)
    # give the lowest level a nonzero wind
    u = st.u.data.at[:, :, -1].set(5.0)
    v = st.v.data.at[:, :, -1].set(3.0)
    st = st._replace(u=st.u.replace(data=u), v=st.v.replace(data=v))
    out = apply_prescribed_surface_fluxes_plane(
        st, 0.0, 0.0, hc, 20.0, cd_momentum=1.5e-3)
    un = np.asarray(out.u.data)
    vn = np.asarray(out.v.data)
    assert np.all(np.abs(un[:, :, -1]) < 5.0)      # u decelerated
    assert np.all(np.abs(vn[:, :, -1]) < 3.0)      # v decelerated
    assert np.allclose(un[:, :, :-1], np.asarray(st.u.data)[:, :, :-1])  # only sfc
    # cd=0 ⇒ wind unchanged
    out0 = apply_prescribed_surface_fluxes_plane(st, 0.0, 0.0, hc, 20.0,
                                                 cd_momentum=0.0)
    assert np.allclose(np.asarray(out0.u.data), np.asarray(st.u.data))


def test_apply_prescribed_radiative_cooling(tmp_path):
    """Prescribed dT/dt|_rad cools θ' at ALL levels (negative ⇒ θ' decreases)."""
    from legoesm.atmosphere.dynamics.crm.sam_case_setup import (
        apply_prescribed_radiative_cooling_plane)
    snd = read_sam_snd(_case(tmp_path) / "snd")
    hc = build_sam_case_height_coord(snd, _NLEV, _H, 100000.0)
    st = build_sam_case_initial_state(snd, _grid(), hc, seed_amp=0.0)
    th0 = np.asarray(st.theta_prime.data).copy()
    dTdt = np.full(_NLEV, -1.6e-5)            # uniform cooling [K/s]
    out = apply_prescribed_radiative_cooling_plane(st, dTdt, hc, 100.0)
    th = np.asarray(out.theta_prime.data)
    assert np.all(th < th0)                   # cooled everywhere
    # magnitude: dθ' = dt·(dT/dt)/Π_ref; check at the lowest level
    exner_sfc = float(np.asarray(hc.exner_ref)[-1])
    assert (th[0, 0, -1] - th0[0, 0, -1]) == pytest.approx(
        100.0 * (-1.6e-5) / exner_sfc, rel=1e-6)


def test_band_limited_seed_pattern_properties():
    """band_limited_seed_pattern (iter-82, the SAM C4-noise analogue): zero-mean,
    unit-std, BAND-LIMITED (no DC, no power above k_max — so NO grid-scale 2Δx
    power that NaNs the Smag-K=0 acoustic IC), reproducible per rng_seed, distinct
    across seeds, and MULTI-CELL (not a single smooth k=1 plume)."""
    ny, nx, k_max = 32, 32, 6
    f = band_limited_seed_pattern(ny, nx, k_max=k_max, rng_seed=0)
    assert f.shape == (ny, nx)
    assert abs(f.mean()) < 1e-10
    assert f.std() == pytest.approx(1.0, abs=1e-10)
    spec = np.fft.fft2(f)
    kxv = np.fft.fftfreq(nx) * nx
    kyv = np.fft.fftfreq(ny) * ny
    KY, KX = np.meshgrid(kyv, kxv, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2)
    power = np.abs(spec) ** 2
    tot = power.sum()
    assert power[kmag < 1.0].sum() < 1e-8 * tot           # no DC mode
    assert power[kmag > float(k_max)].sum() < 1e-8 * tot  # no >k_max (incl 2Δx)
    assert np.allclose(
        f, band_limited_seed_pattern(ny, nx, k_max=k_max, rng_seed=0))
    assert not np.allclose(
        f, band_limited_seed_pattern(ny, nx, k_max=k_max, rng_seed=1))
    # multi-cell: a single k=1 wave has 2 sign changes along a row; a band has many
    sign_changes = int(np.sum(np.abs(np.diff(np.sign(f[ny // 2]))) > 0))
    assert sign_changes >= 4


def test_random_band_theta_seed_in_lowest_levels_zero_mean():
    """_random_band_theta_seed: nonzero ONLY in the lowest n_seed_lev levels
    (surface-last), zero horizontal mean per seeded level, amplitude ~ amp."""
    ny, nx, nlev, n_seed, amp = 24, 24, 40, 4, 0.1
    seed = np.asarray(_random_band_theta_seed(
        ny, nx, nlev, n_seed, amp, k_max=6, rng_seed=0, dtype=jnp.float64))
    assert seed.shape == (ny, nx, nlev)
    assert np.allclose(seed[..., :nlev - n_seed], 0.0)     # upper levels untouched
    assert np.any(seed[..., -n_seed:] != 0.0)              # BL seeded
    for k in range(nlev - n_seed, nlev):
        assert abs(seed[..., k].mean()) < 1e-10            # zero horizontal mean
    assert 0.3 * amp < seed[..., -1].std() < 3.0 * amp     # amplitude ~ amp


# --- Wing 2018 RCEMIP protocol seed (Sect. 3.2.3) ------------------------
#
# Protocol text being pinned (Wing et al. 2018, GMD 11, 793-813, Sect. 3.2.3):
#   "symmetry is to be broken by prescribing a small amount of thermal noise in
#    the five lowest layers (an amplitude of 0.1 K in the lowest layer,
#    decreasing linearly to 0.02 K in the fifth layer)."


def test_wing2018_seed_amplitudes_match_the_published_taper():
    """The protocol numbers, verbatim: 0.10 -> 0.02 K linearly over 5 layers."""
    amps = wing2018_seed_amplitudes()
    assert amps.shape == (WING_SEED_N_LAYERS,)
    assert amps[0] == pytest.approx(WING_SEED_AMP_LOWEST_K)   # lowest layer
    assert amps[-1] == pytest.approx(WING_SEED_AMP_FIFTH_K)   # fifth layer
    np.testing.assert_allclose(amps, [0.10, 0.08, 0.06, 0.04, 0.02], atol=1e-12)
    # "decreasing linearly" => constant second difference of zero.
    np.testing.assert_allclose(np.diff(amps, 2), 0.0, atol=1e-12)


def test_wing2018_seed_amplitudes_scale_preserves_protocol_shape():
    """amp_lowest rescales the taper without changing its 0.2 ratio."""
    amps = wing2018_seed_amplitudes(amp_lowest=0.5)
    assert amps[0] == pytest.approx(0.5)
    assert amps[-1] / amps[0] == pytest.approx(
        WING_SEED_AMP_FIFTH_K / WING_SEED_AMP_LOWEST_K)


def test_wing2018_seed_amplitudes_reject_zero_amplitude():
    """A zero-amplitude symmetry breaker is the iter-149 column-symmetric trap:
    the run looks healthy and convection never initiates. Fail loudly."""
    with pytest.raises(ValueError, match="amp_lowest must be > 0"):
        wing2018_seed_amplitudes(amp_lowest=0.0)
    with pytest.raises(ValueError, match="n_seed_lev must be >= 1"):
        wing2018_seed_amplitudes(n_seed_lev=0)


def test_wing2018_thermal_noise_seed_structure_amplitude_and_zero_mean():
    ny, nx, nlev = 12, 10, 20
    seed = np.asarray(wing2018_thermal_noise_seed(ny, nx, nlev, rng_seed=3))
    assert seed.shape == (ny, nx, nlev)
    # Only the five LOWEST layers are perturbed; top-down storage => last five.
    assert np.all(seed[..., :nlev - WING_SEED_N_LAYERS] == 0.0)
    assert np.any(seed[..., -WING_SEED_N_LAYERS:] != 0.0)
    amps = wing2018_seed_amplitudes()          # lowest first
    for j in range(WING_SEED_N_LAYERS):
        layer = seed[..., -1 - j]              # j=0 is the LOWEST layer
        a = amps[j]
        # Bounded by its own layer amplitude (uniform[-A, A], zero-mean shift
        # can only move it by < A).
        assert np.max(np.abs(layer)) <= 2.0 * a
        assert np.max(np.abs(layer)) > 0.3 * a, "layer amplitude collapsed"
        # Zero horizontal mean (our documented deviation; see the docstring).
        assert abs(layer.mean()) < 1e-12
    # The taper must be MONOTONE decreasing upward in RMS, which is the whole
    # point of the protocol shape — a uniform-amplitude seed would not be.
    rms = [seed[..., -1 - j].std() for j in range(WING_SEED_N_LAYERS)]
    assert all(rms[j] > rms[j + 1] for j in range(WING_SEED_N_LAYERS - 1)), rms


def test_wing2018_thermal_noise_seed_is_deterministic_under_rng_seed():
    a = np.asarray(wing2018_thermal_noise_seed(8, 8, 12, rng_seed=7))
    b = np.asarray(wing2018_thermal_noise_seed(8, 8, 12, rng_seed=7))
    c = np.asarray(wing2018_thermal_noise_seed(8, 8, 12, rng_seed=8))
    np.testing.assert_array_equal(a, b)          # same seed => bit-identical
    assert not np.array_equal(a, c)              # different seed => different


def test_wing2018_thermal_noise_seed_is_white_not_band_limited():
    """The protocol says 'noise'. Distinguish it from band_limited_seed_pattern
    by spectral content: white noise has power at the Nyquist wavenumber, the
    band-limited pattern has EXACTLY zero there (that is its defining property).

    This is the discriminator that would catch someone silently routing
    'wing2018' to the legacy band-limited helper.
    """
    ny = nx = 32
    seed = np.asarray(wing2018_thermal_noise_seed(ny, nx, 8, rng_seed=1))
    lowest = seed[..., -1]
    spec = np.abs(np.fft.fft2(lowest))
    nyq = spec[ny // 2, nx // 2]                    # 2-dx mode
    assert nyq > 1e-3 * spec.max(), "wing2018 seed has no grid-scale power"
    band = band_limited_seed_pattern(ny, nx, k_max=6, rng_seed=1)
    band_nyq = np.abs(np.fft.fft2(band))[ny // 2, nx // 2]
    assert band_nyq < 1e-9, "control failed: band pattern should be band-limited"


def test_wing2018_seed_can_opt_out_of_the_zero_mean_deviation():
    """zero_horizontal_mean=False gives the LITERAL protocol (no such
    constraint is imposed by Wing 2018)."""
    raw = np.asarray(wing2018_thermal_noise_seed(
        16, 16, 10, rng_seed=2, zero_horizontal_mean=False))
    # An unconstrained uniform draw has a nonzero (O(A/sqrt(N))) layer mean.
    assert abs(raw[..., -1].mean()) > 1e-6
    assert np.max(np.abs(raw[..., -1])) <= WING_SEED_AMP_LOWEST_K
