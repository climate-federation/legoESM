"""Oracle test: OpenIFS 49r1 aerosol path of the ecRad IFS port (NAERMACC=1).

Fixtures: ifs49_driver.x runs with NAERMACC=1 (scripts/validate/oracle/ecrad_ifs,
ifs49_harness_aer.nam), exported like the interface fixture.  Data: 48r1
aerosol files as a declared departure (votes A1a, A2a; LAER3D=F, 2D CAMS
climatology 43R3a).  Stages, each fed the oracle's own inputs:

* climatology + background: drv_in_* -> PAERO / PAER (ifs_aerosol);
* RADIATION_SCHEME mixing ratio: PAERO, PAER -> ifs_in_aerosol_mixing_ratio;
* add_aerosol_optics: aer_in_* -> aer_out_* (aerosol_optics);
* end to end: driver inputs -> rs_* / ri_* / rh_* (fluxes and heating).

fp64 (JAX_ENABLE_X64=1); hard form (sharpness = inf) at rel <= 1e-12.
"""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.atmosphere.physics.radiation.ecrad_ifs import aerosol_optics as ao
from legoesm.atmosphere.physics.radiation.ecrad_ifs import ifs_aerosol as ia
from legoesm.atmosphere.physics.radiation.ecrad_ifs import ifs_interface as ii
from tests.atmosphere import test_ecrad_ifs_interface_oracle as base

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(__file__)
FIXTURES = {
    "iv8": os.path.join(HERE, "oracle_fixtures", "ecrad_ifs49_aer.npz"),
    "d0201": os.path.join(HERE, "oracle_fixtures", "ecrad_ifs49_aer_0201.npz"),
    "x100": os.path.join(HERE, "oracle_fixtures", "ecrad_ifs49_aer_x100.npz"),
}
TOL = 1.0e-12
HARD = math.inf
rel_err = base.rel_err


def load(name):
    with np.load(FIXTURES[name]) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module", params=list(FIXTURES))
def fx(request):
    return dict(load(request.param), case=request.param)


def d(fx, name):
    return fx["dump:" + name]


# --------------------------------------------------------------------------
# synthetic, no oracle needed
# --------------------------------------------------------------------------

def _calc_rh_index(rh, rh_lower):
    """radiation_aerosol_optics_data.F90:645-654, one-based."""
    nrh = len(rh_lower)
    if rh > rh_lower[nrh - 1]:
        return nrh
    i = 1
    while rh > rh_lower[i]:
        i += 1
    return i


def test_rh_bin_weights_hard_is_calc_rh_index():
    rl = ao.load_tables().rh_lower
    rh = np.concatenate([np.linspace(0.0, 1.3, 2001), rl, np.nextafter(rl, 2.0)])
    w = np.asarray(ao.rh_bin_weights(jnp.asarray(rh), rl, HARD))
    assert np.all((w == 0.0) | (w == 1.0)) and np.all(w.sum(-1) == 1.0)
    idx = np.argmax(w, -1) + 1
    ref = np.array([_calc_rh_index(r, rl) for r in rh])
    assert np.array_equal(idx, ref)


def test_rh_bin_weights_smooth_partition_of_unity():
    rl = ao.load_tables().rh_lower
    w = np.asarray(ao.rh_bin_weights(jnp.linspace(0.0, 1.3, 501), rl, 100.0))
    np.testing.assert_allclose(w.sum(-1), 1.0, rtol=0, atol=1e-15)
    assert np.all(w >= 0.0)


def test_month_and_year_by_hand():
    # 20200115, 60 min: ZXTIME = 14*1440+60 = 20220 <= ZYTIME(1) -> IM1=12,
    # ZTIMI = (22320-20220)/44640 (updecaec.F90:154-157)
    iyr, pm = ia.month_and_year(20200115, 60)
    assert iyr == 2020 and pm == 13.0 - (22320.0 - 20220.0) / 44640.0
    # 20200201, 0 min: ZXTIME = 44640 in (ZYTIME(1), ZYTIME(2)] -> IM1=1
    iyr, pm = ia.month_and_year(20200201, 0)
    assert pm == 2.0 - (44640.0 - 64980.0) / (22320.0 - 64980.0)


def test_calc_profiles_conserve_column_mass():
    """2D arm: sum_k mmr * dp / RG = column mass * (cum(sfc) - cum(top))."""
    ph = jnp.asarray(np.linspace(0.0, 1.0e5, 31)[None, :] * np.array([[1.0], [0.97]]))
    eta = np.linspace(0.0, 1.0, 31)
    cum = np.stack([eta ** (8434.0 / h) for h in (1000.0, 2000.0, 4000.0)], -1)
    cm = np.array([[1e-5, 2e-6, 3e-7], [4e-5, 5e-6, 6e-7]])
    mmr = ia.calc_profiles_2d(cm, cum, ph)
    col = jnp.sum(mmr * (ph[:, 1:, None] - ph[:, :-1, None]), axis=1) / ia.constants.g_ifs
    np.testing.assert_allclose(np.asarray(col), cm * (cum[-1] - cum[0]), rtol=1e-13)


def _optics_args(fx):
    o = lambda n: np.transpose(d(fx, n), (2, 1, 0))  # noqa: E731  (ng, nlev, col) -> (col, nlev, ng)
    q = fx["in:q"]
    return dict(od_sw=o("aer_in_od_sw"), ssa_sw=o("aer_in_ssa_sw"), od_lw=o("aer_in_od_lw"),
                pressure_hl=d(fx, "aer_in_pressure_hl"), h2o_mmr=q,
                h2o_sat_liq=d(fx, "aer_in_h2o_sat_liq"),
                mixing_ratio=d(fx, "aer_in_mixing_ratio"))


# --------------------------------------------------------------------------
# oracle stages
# --------------------------------------------------------------------------

def test_tables_are_the_oracle_setup(fx):
    """The vendored npz equals this run's post-setup tables (same oracle exe)."""
    raw = ao.load_raw()
    for k in ("mass_ext_sw_phobic", "ssa_sw_phobic", "g_sw_phobic", "mass_ext_lw_phobic",
              "ssa_lw_phobic", "mass_ext_sw_philic", "ssa_sw_philic", "g_sw_philic",
              "mass_ext_lw_philic", "ssa_lw_philic", "rh_lower"):
        assert np.array_equal(raw[k], d(fx, "aot_" + k)), k
    assert np.array_equal(raw["type_map"], d(fx, "aot_type_map"))
    assert float(raw["trop_bg_mass_ext"]) == float(d(fx, "su_trop_bg_aer_mass_ext"))
    assert float(raw["strat_bg_mass_ext"]) == float(d(fx, "su_strat_bg_aer_mass_ext"))


def test_add_aerosol_optics_stage(fx):
    out = ao.add_aerosol_optics(**_optics_args(fx), sharpness=HARD)
    for name, got in zip(("od_sw", "ssa_sw", "g_sw", "od_lw"), out):
        ref = np.transpose(d(fx, "aer_out_" + name), (2, 1, 0))
        e = rel_err(got, ref)
        print(f"AER_STAGE {name} max_rel={e:.3e}")
        assert e <= TOL, (name, e)


def test_add_aerosol_optics_is_not_vacuous(fx):
    """The aerosol stage changes the gas optics (od_sw, ssa_sw, g_sw, od_lw)."""
    a = _optics_args(fx)
    out = ao.add_aerosol_optics(**a, sharpness=HARD)
    assert np.max(np.abs(np.asarray(out[0]) - a["od_sw"])) > 0.0
    assert np.max(np.abs(np.asarray(out[2]))) > 0.0
    assert np.max(np.abs(np.asarray(out[3]) - a["od_lw"])) > 0.0


CAMS2D = os.environ.get(
    "ECRAD_IFS_CAMS2D",
    "/burg/glab/users/pg2328/oracle_audit/ifs_port/ecrad/agents/aerosol/cams_extract/"
    "aerosol_cams_climatology_43R3a.nc")


def test_climatology_and_background_stage(fx):
    """UPDECAEC + TEAERC (2D) + RADACT: drv_in_* -> aer_radact_paero / _paer."""
    if not os.path.exists(CAMS2D):
        pytest.skip(f"CAMS 2D climatology not present: {CAMS2D} (vote A13: not vendored)")
    clim = ia.load_cams_2d(CAMS2D)
    cm_pts, cum = ia.climatology_at_columns(
        clim, int(fx["nml:NINDAT_H"]), int(d(fx, "aer_kminut")), d(fx, "aer_etah"),
        d(fx, "drv_in_gemu"), d(fx, "drv_in_gelam"))
    paero = ia.calc_profiles_2d(cm_pts, cum, fx["in:pressure_hl"])
    paer1, paer6 = ia.background_aerosol(fx["in:pressure_hl"], d(fx, "drv_in_temperature_hl"), HARD)
    errs = {"paero": rel_err(paero, d(fx, "aer_radact_paero")),
            "paer1": rel_err(paer1, d(fx, "aer_radact_paer")[:, 0, :]),
            "paer6": rel_err(paer6, d(fx, "aer_radact_paer")[:, 5, :])}
    print("CLIM", {k: f"{v:.3e}" for k, v in errs.items()})
    assert max(errs.values()) <= TOL, errs


def test_radintg_glue(fx):
    """radintg assembly (x MMR_SCALE control): PAERO -> drv_in_aerosol."""
    scale = float(d(fx, "drv_in_mmr_scale"))
    paerosol = ia.radintg_aerosol(jnp.asarray(d(fx, "aer_radact_paero"))) * scale
    assert rel_err(paerosol, d(fx, "drv_in_aerosol")) <= TOL


def test_mixing_ratio_stage(fx):
    """radiation_scheme.F90:545-575: drv_in_aerosol(_old) -> ifs_in_aerosol_mixing_ratio."""
    layer_mass = (fx["in:pressure_hl"][:, 1:] - fx["in:pressure_hl"][:, :-1]) * ii._INV_G_ECRAD
    old = d(fx, "drv_in_aerosol_old")
    mr = ia.macc_mixing_ratio(d(fx, "drv_in_aerosol"), old[:, 0, :], old[:, 5, :],
                              layer_mass, **ao.background_mass_ext(), sharpness=HARD)
    e = rel_err(mr, d(fx, "ifs_in_aerosol_mixing_ratio"))
    print(f"MIXING_RATIO max_rel={e:.3e}")
    assert e <= TOL


# --------------------------------------------------------------------------
# gradients and energy
# --------------------------------------------------------------------------

def _rh_functional(sharpness, dtype=jnp.float64, mr_value=1e-8):
    nlev, ntype = 3, ia.N_TYPES
    ph = jnp.asarray([[0.0, 3.0e4, 6.0e4, 9.0e4]], dtype)
    mr = jnp.full((1, nlev, ntype), mr_value, dtype)
    qs = jnp.full((1, nlev), 1.0e-2, dtype)
    od_sw = jnp.full((1, nlev, 112), 0.1, dtype)
    ssa = jnp.full((1, nlev, 112), 0.5, dtype)
    od_lw = jnp.full((1, nlev, 140), 0.2, dtype)

    def f(rh, load=1.0):
        q = jnp.full((1, nlev), rh, dtype) * qs
        o = ao.add_aerosol_optics(od_sw, ssa, od_lw, ph, q, qs, mr * load, sharpness=sharpness)
        return jnp.sum(o[0]) + jnp.sum(o[1]) + jnp.sum(o[2]) + jnp.sum(o[3])

    return f


def test_optics_gradient_across_rh_edge():
    """rh enters only through the bin weights: d/d(rh) is exactly 0 in the hard form
    and nonzero inside the smoothed edge (the control: a hard bin would fail here)."""
    edge = float(ao.load_tables().rh_lower[5])
    f_s, f_h = _rh_functional(100.0), _rh_functional(HARD)
    assert float(jax.grad(f_h)(edge + 1e-3)) == 0.0
    g = float(jax.grad(f_s)(edge + 1e-3))
    assert np.isfinite(g) and g != 0.0, g
    for r in (edge - 2e-2, edge, edge + 2e-2):
        assert np.isfinite(float(jax.grad(f_s)(r)))
    check_grads(f_s, (edge + 1e-3,), order=1, modes=("rev",), eps=1e-6)


_FP32_SCRIPT = """
import jax, jax.numpy as jnp, numpy as np
from legoesm.atmosphere.physics.radiation.ecrad_ifs import aerosol_optics as ao, ifs_aerosol as ia
f32 = jnp.float32
ph = jnp.asarray([[0.0, 3.0e4, 6.0e4, 9.0e4]], f32)
qs = jnp.full((1, 3), 1.0e-2, f32)
args = (jnp.full((1, 3, 112), 0.1, f32), jnp.full((1, 3, 112), 0.5, f32),
        jnp.full((1, 3, 140), 0.2, f32), ph, 0.5 * qs, qs)
def f(load):
    o = ao.add_aerosol_optics(*args, jnp.full((1, 3, ia.N_TYPES), 1e-30, f32) * load, sharpness=100.0)
    return sum(jnp.sum(x) for x in o)
g = jax.grad(f)(f32(1.0))
assert g.dtype == jnp.float32, g.dtype
print("FP32_GRAD", float(g)); assert np.isfinite(float(g))
# radact tropopause weight underflow (codex reproduction): 137 layers, 245/255 K
n = 137
p = jnp.linspace(1000.0, 100000.0, n + 1, dtype=f32)[None, :]
t = (250.0 + 5.0 * jnp.cos(jnp.arange(n + 1, dtype=f32) * jnp.pi))[None, :]
gt = jax.grad(lambda t: sum(jnp.sum(x) for x in ia.background_aerosol(p, t, 100.0)))(t)
print("FP32_BG_GRAD_FINITE", bool(jnp.all(jnp.isfinite(gt)))); assert bool(jnp.all(jnp.isfinite(gt)))
"""


def test_optics_fp32_tiny_load_gradient_is_finite():
    """Vote D2a: x64 off (fp32 throughout), aerosol loads of 1e-30 kg/kg: finite
    reverse-mode gradient (separate process: x64 is process-global)."""
    import subprocess
    import sys
    env = dict(os.environ, JAX_ENABLE_X64="0", JAX_PLATFORMS="cpu")
    r = subprocess.run([sys.executable, "-c", _FP32_SCRIPT], env=env, capture_output=True, text=True)
    print(r.stdout[-300:], r.stderr[-1500:])
    assert r.returncode == 0


def test_background_gradient():
    ph = jnp.asarray([np.concatenate([[0.0], np.geomspace(300.0, 1.0e5, 40)])])
    th = jnp.asarray([np.linspace(220.0, 290.0, 41)])
    # one level moved through T(k)/T(k+1) = 1 (the min(1, .) clamp of radact)
    f = lambda t: sum(jnp.sum(x) for x in  # noqa: E731
                      ia.background_aerosol(ph, th.at[0, 20].set(th[0, 21] + t), 100.0))
    for t in (-0.05, 0.0, 0.05):
        g = float(jax.grad(f)(t))
        assert np.isfinite(g), t
    assert float(jax.grad(f)(-0.5)) != 0.0
    check_grads(f, (-0.5,), order=1, modes=("rev",), eps=1e-6)


# --------------------------------------------------------------------------
# end to end: driver inputs -> RADIATION_SCHEME -> RADINTG -> RADHEATN
# --------------------------------------------------------------------------

def macc(fx):
    old = d(fx, "drv_in_aerosol_old")
    return ia.MaccAerosol(jnp.asarray(d(fx, "drv_in_aerosol")), jnp.asarray(old[:, 0, :]),
                          jnp.asarray(old[:, 5, :]))


def run_composed(fx, sharpness, aerosol=True, jit=False, scale=1.0, uniforms=None):
    """``uniforms``: the McICA stream; default the fixture's own replay (spread runs
    pass the held-seed stream, vote C11)."""
    a = base.composed_args(fx)
    static = {k: a.pop(k) for k in base._STATIC_ARGS}
    ulw, usw = base.replayed_uniforms(fx) if uniforms is None else uniforms
    aer = macc(fx) if aerosol else None

    def f(mu0, arrays, aer, s):
        if aer is not None:
            aer = ia.MaccAerosol(aer.paerosol * s, aer.paer1, aer.paer6)
        rs, carry = ii.radiation_scheme(mu0=mu0, uniforms_lw=ulw, uniforms_sw=usw,
                                        sharpness=sharpness, aerosol=aer, **arrays, **static)
        return rs, carry._asdict()

    rs, carry = (jax.jit(f) if jit else f)(fx["in:cos_solar_zenith_angle"], a, aer, scale)
    return ({k: np.asarray(v) for k, v in rs.items()}, {k: np.asarray(v) for k, v in carry.items()})


@pytest.fixture(scope="module")
def composed(fx):
    return run_composed(fx, HARD)


def test_end_to_end_fluxes(fx, composed):
    rs, carry = composed
    errs = {"rs_" + k: rel_err(rs[k], d(fx, "rs_" + k)) for k in base.RS_KEYS}
    errs.update({"ri_" + k: rel_err(carry[f], d(fx, "ri_" + k)) for f, k in base.RI_KEYS.items()})
    print("E2E max rel:", max(errs.values()), max(errs, key=errs.get))
    assert max(errs.values()) <= TOL, {k: v for k, v in errs.items() if v > TOL}


# Vote C8 method: the heating rates carry the ecRad-core round-off (fluxes <=4e-14)
# amplified by top-layer flux cancellation; those five fields get the aerosol
# case's own 20-seed 1-ulp oracle spread (committed artifact, controls C1-C5 PASS).
SPREAD_ARTIFACT = os.path.join(HERE, "oracle_fixtures", "spread_ecrad_ifs49_aer.json")


def rh_tol(case, k, s):
    if k not in base._C8_FIELDS:
        return TOL
    import json
    with open(SPREAD_ARTIFACT) as f:
        art = json.load(f)
    assert all(art["controls"][case][c] == "PASS" for c in ("C1", "C2", "C3", "C4", "C5"))
    return max(TOL, float(art["spread"][f"{case}:rh_{k}_s{s}"]))


@pytest.mark.parametrize("s", range(4))
def test_end_to_end_heating(fx, composed, s):
    out = base.run_radheatn(fx, ii.RadiationCarry(**composed[1]), s, HARD)
    errs = {k: rel_err(out[k], d(fx, f"rh_{k}_s{s}")) for k in base.RH_KEYS}
    for k, v in errs.items():
        print(f"RH_COMPOSED {fx['case']}:rh_{k}_s{s} max_rel={v:.3e}")
    bad = {k: (v, rh_tol(fx["case"], k, s)) for k, v in errs.items() if v > rh_tol(fx["case"], k, s)}
    assert not bad, bad


def test_aerosol_changes_fluxes(fx, composed):
    """Non-vacuous: the aerosols-on call differs from aerosols off."""
    off, _ = run_composed(fx, HARD, aerosol=False)
    assert np.max(np.abs(composed[0]["flux_sw"] - off["flux_sw"])) > 1e-3
    assert np.max(np.abs(composed[0]["flux_lw"] - off["flux_lw"])) > 1e-6


def test_energy_closure(fx, composed):
    """Column heating = net flux divergence; TOA/surface = the flux profile ends."""
    out = base.run_radheatn(fx, ii.RadiationCarry(**composed[1]), 2, HARD)
    dp = np.diff(fx["in:pressure_hl"], axis=1)
    col = np.sum(np.asarray(out["te"]) * dp, axis=1) * ia.constants.c_pd_ifs / ia.constants.g_ifs
    net = np.asarray(out["frso"] + out["frth"])
    np.testing.assert_allclose(col, net[:, 0] - net[:, -1], rtol=1e-11, atol=1e-9)
    # TOA: ecRad's net SW at the top = incoming - reflected, positive down, <= incoming
    assert np.all(np.asarray(out["frso"][:, 0]) <= np.asarray(out["incsr"]) + 1e-9)


_JIT_SCRIPT = """
import sys, numpy as np, jax
jax.config.update("jax_enable_x64", True)
from tests.atmosphere import test_ecrad_ifs_aerosol_oracle as t
fx = t.load(sys.argv[3])
rs, carry = t.run_composed(fx, float(sys.argv[2]), jit=True)
np.savez(sys.argv[1], **{"rs:" + k: v for k, v in rs.items()}, **{"ri:" + k: v for k, v in carry.items()})
"""


@pytest.mark.parametrize("sharpness", [HARD, 100.0])
def test_eager_vs_jit_without_fma(tmp_path, sharpness):
    """jit in a separate process with --xla_cpu_max_isa=AVX (no FMA; vote J2c) == eager at 1e-12."""
    import subprocess
    import sys
    fx = load("iv8")
    rs, carry = run_composed(fx, sharpness)
    path = tmp_path / "jit.npz"
    env = dict(os.environ, XLA_FLAGS="--xla_cpu_max_isa=AVX", JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    subprocess.run([sys.executable, "-c", _JIT_SCRIPT, str(path), repr(sharpness), "iv8"],
                   env=env, check=True)
    jit = np.load(path)
    diff = {"rs:" + k: rel_err(v, jit["rs:" + k]) for k, v in rs.items()}
    diff.update({"ri:" + k: rel_err(v, jit["ri:" + k]) for k, v in carry.items()})
    print(f"AER_EAGER_VS_JIT sharpness={sharpness} max_rel={max(diff.values()):.3e} "
          f"worst={max(diff, key=diff.get)}")
    assert max(diff.values()) <= TOL, {k: v for k, v in diff.items() if v > TOL}


def test_end_to_end_gradient_wrt_aerosol_load():
    """d(sum of fluxes)/d(aerosol scale) through the whole call: AD == central FD."""
    fx = load("iv8")
    a = base.composed_args(fx)
    static = {k: a.pop(k) for k in base._STATIC_ARGS}
    ulw, usw = base.replayed_uniforms(fx)
    aer = macc(fx)

    @jax.jit
    def f(s):
        rs, carry = ii.radiation_scheme(
            mu0=fx["in:cos_solar_zenith_angle"], uniforms_lw=ulw, uniforms_sw=usw, sharpness=100.0,
            aerosol=ia.MaccAerosol(aer.paerosol * s, aer.paer1, aer.paer6), **a, **static)
        return jnp.sum(rs["flux_sw"]) * 1e-2 + jnp.sum(rs["flux_lw"]) * 1e-2 + jnp.sum(carry.trso)

    g = float(jax.grad(f)(1.0))
    assert np.isfinite(g) and g != 0.0
    check_grads(f, (1.0,), order=1, modes=("rev",), eps=1e-5)
