"""Integration of the GFS / WRF / MYNN forms with the independent unstable
selector, the smooth |zeta| cap, and the GFS unstable join width.

Run with JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.core import bulk_flux as bf

REPO = Path(__file__).resolve().parents[2]
STABLE = ("dyer1974", "beljaars_holtslag1991", "grachev2007_sheba", "gryanik2020",
          "gfs", "wrf_revmm5", "mynn_sfc")
NEW_UNSTABLE = ("gfs", "wrf_revmm5", "mynn_sfc")
ZNEG = jnp.linspace(-8.0, -0.01, 300)   # below the cap onset, outside the COARE blend


# --- any stable scheme pairs with any unstable form ---------------------------
@pytest.mark.parametrize("u", NEW_UNSTABLE)
def test_new_unstable_forms_are_selectable_on_every_path(u):
    m, h = bf.UNSTABLE_PSI_FUNCTIONS[u]
    kw = {"sharpness": bf._gfs_sharpness(bf.GFS_UNSTABLE_JOIN_WIDTH)} if u == "gfs" else {}
    for s in STABLE:
        np.testing.assert_array_equal(bf.psi_m(ZNEG, s, unstable_scheme=u), m(ZNEG, **kw))
        h_ref = h(ZNEG, **kw) * (0.98 if s == "gryanik2020" else 1.0)
        np.testing.assert_allclose(bf.psi_h(ZNEG, s, unstable_scheme=u), h_ref, rtol=1e-15)
        np.testing.assert_array_equal(bf.psi_m_coare(ZNEG, s, unstable_scheme=u), m(ZNEG, **kw))
    # non-vacuity: the form really differs from the native Businger-Dyer
    assert np.max(np.abs(np.asarray(bf.psi_m(ZNEG, unstable_scheme=u) - bf.psi_m(ZNEG)))) > 1e-3


def test_wrf_and_mynn_are_kept_apart_because_they_differ():
    """Measured: stable branches agree for zeta > 0; the unstable branches do
    not (WRF exponent 0.33 vs MYNN 1/3)."""
    zp = jnp.logspace(-6, 2, 500)
    np.testing.assert_array_equal(bf._wrf_revmm5_psi_m(zp), bf._mynn_psi_m(zp))
    np.testing.assert_array_equal(bf._wrf_revmm5_psi_h(zp), bf._mynn_psi_h(zp))
    for i in (0, 1):
        a = np.asarray(bf.UNSTABLE_PSI_FUNCTIONS["wrf_revmm5"][i](-zp))
        b = np.asarray(bf.UNSTABLE_PSI_FUNCTIONS["mynn_sfc"][i](-zp))
        assert np.max(np.abs(a - b)) > 1e-2


@pytest.mark.parametrize("u", NEW_UNSTABLE)
def test_coare_width_guard_and_finite_values(u):
    z = jnp.asarray([-0.5, -1e-4, 0.0, 1e-4, 0.5])
    for f in (bf.psi_m_coare, bf.psi_h_coare):
        assert np.all(np.isfinite(f(z, "dyer1974", unstable_scheme=u)))
    zmax = bf._UNSTABLE_ZETA_MAX[u]
    zp = jnp.linspace(0.0, 0.999 * zmax, 1001)
    for i in (0, 1):
        assert np.all(np.isfinite(bf.UNSTABLE_PSI_FUNCTIONS[u][i](zp)))
    with pytest.raises(ValueError, match="neutral_width"):
        bf.psi_h_coare(z, "dyer1974", zmax, unstable_scheme=u)
    if u in ("wrf_revmm5", "mynn_sfc"):
        with pytest.raises(ValueError, match="neutral_width"):
            bf.psi_h_coare(z, "dyer1974", 0.05, unstable_scheme=u)


def test_every_pair_phi_continuous_at_zero_and_finite():
    eps = jnp.asarray([-1e-7, 1e-7])
    zz = jnp.linspace(-14.0, 14.0, 2801)
    for s in STABLE:
        for u in ("native", "coare3") + NEW_UNSTABLE:
            for f in (bf.psi_m, bf.psi_h):
                g = lambda z: f(z, s, unstable_scheme=u)
                lo, hi = np.asarray(bf.phi_from_psi(g, eps))
                assert abs(lo - hi) < 1e-5, (s, u, f.__name__, lo, hi)
                assert np.all(np.isfinite(bf.phi_from_psi(g, zz))), (s, u, f.__name__)


# --- smooth |zeta| cap (RULE 5: hard limit, recorded gap, gradient) ----------
def test_cap_hard_limit_is_the_former_clip():
    z = jnp.linspace(-60.0, 60.0, 2401)
    np.testing.assert_array_equal(bf._zeta_cap(z, 0.0), jnp.clip(z, -10.0, 10.0))
    for s in STABLE:
        for f in (bf.psi_m, bf.psi_h):
            np.testing.assert_array_equal(f(z, s, zeta_cap_width=0.0),
                                          f(jnp.clip(z, -10.0, 10.0), s, zeta_cap_width=0.0))


def test_cap_form_identity_below_onset_and_c2():
    w = bf.MOST_ZETA_CAP_WIDTH
    z = jnp.linspace(-(10.0 - w), 10.0 - w, 801)
    np.testing.assert_array_equal(bf._zeta_cap(z, w), z)
    big = jnp.asarray([-40.0, -12.0, 12.0, 40.0])
    np.testing.assert_allclose(bf._zeta_cap(big, w), jnp.sign(big) * 10.0, rtol=0, atol=1e-14)
    d1 = jax.vmap(jax.grad(lambda x: bf._zeta_cap(x, w)))
    d2 = jax.vmap(jax.grad(jax.grad(lambda x: bf._zeta_cap(x, w))))
    for edge in (10.0 - w, 10.0 + w, -(10.0 - w), -(10.0 + w)):
        e = jnp.asarray([edge - 1e-6, edge + 1e-6])
        assert abs(float(jnp.diff(d1(e))[0])) < 1e-5
        assert abs(float(jnp.diff(d2(e))[0])) < 1e-4


# Measured psi(default) / psi(hard clip) - 1 per stable scheme (momentum, heat)
# at zeta = 10, 20 (stable) -- recorded gap of the default (RULE 5).  At
# |zeta| <= 5 the default is identical (identity below the onset at 8).
_CAP_GAP = {  # scripts/tmp/probe_cap_change.py, 2026-11-07
    'dyer1974': ((-0.03125, 0.0), (-0.03125, 0.0)),
    'beljaars_holtslag1991': ((-0.01694756725001456, 0.0), (-0.029536855942800866, 0.0)),
    'grachev2007_sheba': ((-0.018486872421613176, 0.0), (-0.012967531571088098, 0.0)),
    'gryanik2020': ((-0.02127973648961856, 0.0), (-0.015730838567111438, 0.0)),
    'gfs': ((-0.0186009684699604, 0.0), (-0.0186009684699604, 0.0)),
    'wrf_revmm5': ((-0.010578351367821703, 0.0), (-0.010068863100138215, 0.0)),
    'mynn_sfc': ((-0.010578351367821703, 0.0), (-0.010068863100138215, 0.0)),
}


@pytest.mark.parametrize("s", STABLE)
def test_cap_default_gap_is_the_recorded_one(s):
    z = jnp.asarray([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
    for f in (bf.psi_m, bf.psi_h):
        np.testing.assert_array_equal(f(z, s), f(z, s, zeta_cap_width=0.0))
    zs = jnp.asarray([10.0, 20.0])
    for i, f in enumerate((bf.psi_m, bf.psi_h)):
        got = np.asarray(f(zs, s)) / np.asarray(f(zs, s, zeta_cap_width=0.0)) - 1.0
        np.testing.assert_allclose(got, _CAP_GAP[s][i], rtol=1e-6, atol=1e-12)


def test_cap_gradient_at_default():
    for z0 in (-11.0, -9.0, 9.0, 11.0):
        check_grads(lambda z: bf.psi_m(z, "beljaars_holtslag1991"), (jnp.asarray(z0),),
                    order=2, modes=["rev"])
        check_grads(lambda z: bf.psi_h(z, "dyer1974"), (jnp.asarray(z0),),
                    order=2, modes=["rev"])


def test_cap_width_validated_and_threaded_by_the_solver(monkeypatch):
    x = [jnp.full(2, v) for v in (1.5, 0.0, 290.0, 0.008, 285.0, 0.006, 1.2)]
    for bad in (-0.1, 5.5):
        with pytest.raises(ValueError, match="zeta_cap_width"):
            bf.psi_m(jnp.asarray([1.0]), zeta_cap_width=bad)
        with pytest.raises(ValueError, match="most_zeta_cap_width"):
            bf.compute_most_fluxes(*x, scheme="most", most_zeta_cap_width=bad)
    seen = []
    real = bf._zeta_cap
    monkeypatch.setattr(bf, "_zeta_cap", lambda z, w, *a: (seen.append(w), real(z, w, *a))[1])
    out = bf.compute_most_fluxes(*x, scheme="most", most_zeta_cap_width=1.234)
    assert np.all(np.isfinite(np.asarray(out[2])))
    assert seen and all(w == 1.234 for w in seen), seen


# --- GFS unstable join width field (RULE 5) ----------------------------------
def test_gfs_join_width_hard_limit_and_default():
    zn = jnp.linspace(-5.0, -0.01, 400)
    np.testing.assert_array_equal(
        bf.psi_m(zn, unstable_scheme="gfs", gfs_unstable_join_width=0.0),
        bf._gfs_unstable_psi_m(zn, sharpness=math.inf))
    np.testing.assert_array_equal(
        bf.psi_h(zn, unstable_scheme="gfs"),
        bf._gfs_unstable_psi_h(zn, sharpness=bf._GFS_UNSTABLE_SWITCH_SHARPNESS))
    assert bf._gfs_sharpness(bf.GFS_UNSTABLE_JOIN_WIDTH) == bf._GFS_UNSTABLE_SWITCH_SHARPNESS


def test_gfs_join_gradient_at_default():
    for z0 in (-0.6, -0.5, -0.4):
        check_grads(lambda z: bf.psi_m(z, unstable_scheme="gfs"), (jnp.asarray(z0),),
                    order=2, modes=["rev"])


def test_gfs_width_reaches_the_solver():
    x = [jnp.full(2, v) for v in (1.0, 0.0, 290.0, 0.008, 300.0, 0.012, 1.2)]
    kw = dict(scheme="coare3", unstable_scheme="gfs")
    a = bf.compute_most_fluxes(*x, **kw)
    b = bf.compute_most_fluxes(*x, **kw, gfs_unstable_join_width=0.0)
    assert not np.allclose(np.asarray(a[2]), np.asarray(b[2]))


# --- driver plumbing ---------------------------------------------------------
def _load(name):
    spec = importlib.util.spec_from_file_location(f"_{name}_m", REPO / "scripts" / "run" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_flags_reach_the_configs():
    amip = _load("run_amip")
    p = amip.build_arg_parser()
    cfg = amip.build_config_from_args(p.parse_args(
        ["--surface-bulk-scheme", "coare3", "--surface-unstable-scheme", "gfs",
         "--surface-most-zeta-cap-width", "0", "--surface-gfs-unstable-join-width", "0.1"]))
    assert cfg.surface_unstable_scheme == "gfs"
    assert cfg.surface_most_zeta_cap_width == 0.0
    assert cfg.surface_gfs_unstable_join_width == 0.1
    coupled = _load("run_coupled")
    args = coupled.build_parser().parse_args(
        ["--surface-bulk-scheme", "coare3", "--surface-most-zeta-cap-width", "0",
         "--surface-gfs-unstable-join-width", "0.1"])
    c = coupled.build_coupler_config(args)
    assert (c.most_zeta_cap_width, c.gfs_unstable_join_width) == (0.0, 0.1)


def test_experiment_config_widths_reach_the_surface_layer():
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import turbulence_config_for

    cfg = ExperimentConfig(turbulence="louis", surface_bulk_scheme="most",
                           surface_most_zeta_cap_width=0.0,
                           surface_gfs_unstable_join_width=0.1)
    s = turbulence_config_for(cfg).louis.surface
    assert (s.most_zeta_cap_width, s.gfs_unstable_join_width) == (0.0, 0.1)
    d = turbulence_config_for(ExperimentConfig(turbulence="louis")).louis.surface
    assert (d.most_zeta_cap_width, d.gfs_unstable_join_width) == (
        bf.MOST_ZETA_CAP_WIDTH, bf.GFS_UNSTABLE_JOIN_WIDTH)


def test_non_most_psi_callers_keep_the_hard_clip_omip_ncar_threads_the_width(monkeypatch):
    """CLM canopy and LES surface keep the former +-10 clip (width 0); the
    OMIP NCAR routine forwards its zeta_cap_width (default = the shared one).
    (The NEMO-exact lane returns before the cap.)"""
    from legoesm.atmosphere.dynamics.les import spectral_les_plane as les
    from legoesm.land.canopy import stability as canopy
    from legoesm.ocean import bulk_flux_omip as omip

    seen = []
    real = bf._zeta_cap
    monkeypatch.setattr(bf, "_zeta_cap",
                        lambda z, w, *a: (seen.append(w), real(z, w, *a))[1])
    canopy._friction_velocity(jnp.asarray([10.0]), jnp.asarray([0.1]),
                              jnp.asarray([0.9]), jnp.asarray([2.0]))
    assert seen and all(w == 0.0 for w in seen), seen
    src = Path(les.__file__).read_text()
    assert src.count("zeta_cap_width=0.0") == 2
    wind = jnp.asarray([0.5, 1.0, 4.0])
    for kw, want in (({}, bf.MOST_ZETA_CAP_WIDTH), ({"zeta_cap_width": 0.0}, 0.0)):
        seen.clear()
        omip.ncar_transfer_coefficients(280.0, 0.005, 290.0, 0.010, wind, **kw)
        assert seen and all(w == want for w in seen), seen


def _choices(parser, dest):
    return tuple(next(a for a in parser._actions if a.dest == dest).choices)


def test_every_stable_scheme_is_selectable_from_every_driver():
    """Each driver's stable-scheme choices are exactly the core dispatch set,
    and mynn_sfc round-trips into the configs each driver builds."""
    from legoesm.driver.config import ExperimentConfig
    from scripts.run.run_omip import parse_args as omip_parse

    core = bf._VALID_STABILITY_SCHEMES
    amip = _load("run_amip")
    p = amip.build_arg_parser()
    assert _choices(p, "surface_stability_scheme") == core
    cfg = amip.build_config_from_args(p.parse_args(
        ["--surface-bulk-scheme", "coare3", "--surface-stability-scheme", "mynn_sfc"]))
    assert cfg.surface_stability_scheme == "mynn_sfc"
    ExperimentConfig(turbulence="louis", surface_bulk_scheme="coare3",
                     surface_stability_scheme="mynn_sfc").validate_strict()
    with pytest.raises(ValueError, match="surface_stability_scheme"):
        ExperimentConfig(turbulence="louis", surface_bulk_scheme="coare3",
                         surface_stability_scheme="mynn").validate_strict()

    a = omip_parse(["--grid", "latlon", "--surface-stability-scheme", "mynn_sfc"])
    assert a.surface_stability_scheme == "mynn_sfc"
    with pytest.raises(SystemExit):
        omip_parse(["--grid", "latlon", "--surface-stability-scheme", "mynn"])

    coupled = _load("run_coupled")
    cp = coupled.build_parser()
    assert _choices(cp, "surface_stability_scheme") == core
    assert _choices(cp, "ice_stability_scheme") == core
    args = cp.parse_args(["--surface-bulk-scheme", "coare3",
                          "--surface-stability-scheme", "mynn_sfc"])
    assert coupled.build_coupler_config(args).stability_scheme == "mynn_sfc"
    args = cp.parse_args(["--ice-bulk-scheme", "most", "--ice-stability-scheme", "mynn_sfc"])
    assert coupled.build_sea_ice_config(args).stability_scheme == "mynn_sfc"


def test_omip_core2_cap_flag_reaches_every_ncar_flux_call():
    import ast
    core2 = _load("run_omip_core2")
    p = core2._build_arg_parser()
    assert p.parse_args([]).most_zeta_cap_width == bf.MOST_ZETA_CAP_WIDTH
    assert p.parse_args(["--most-zeta-cap-width", "0"]).most_zeta_cap_width == 0.0
    tree = ast.parse((REPO / "scripts" / "run" / "run_omip_core2.py").read_text())
    names = {"compute_omip2_surface_forcing", "compute_omip2_freshwater_forcing",
             "build_omip2_scan_block_fn"}
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, "id", None) in names]
    assert len(calls) == 6
    for c in calls:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("zeta_cap_width") == "args.most_zeta_cap_width", ast.unparse(c)
    app = ast.parse((REPO / "packages" / "ocean" / "legoesm" / "ocean" / "coupler"
                     / "omip2_applicator.py").read_text())
    calls = [c for c in ast.walk(app) if isinstance(c, ast.Call)
             and getattr(c.func, "id", None) in ("air_sea_fluxes", "compute_omip2_surface_forcing_jax")]
    assert len(calls) == 4
    for c in calls:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("zeta_cap_width") == "zeta_cap_width", ast.unparse(c)
