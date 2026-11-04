"""Oracle test: ecRad 49r1 McICA LW/SW solvers (ecrad_ifs.mcica_solvers).

Inputs are the oracle's own solver inputs (radiation_save writes them after
gas optics, cloud optics, crop_cloud_fraction and the aerosol-off zeroing,
immediately before solver_mcica_lw/sw; lw_emissivity is saved as
1 - lw_albedo and the round trip is checked exact here), plus the McICA
generator outputs od_scaling / total_cloud_cover.  Outputs are every flux:*
variable of the oracle run.  fp64 (JAX_ENABLE_X64=1).
"""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.atmosphere.physics.radiation.ecrad_ifs import mcica_solvers as ms
from legoesm.atmosphere.physics.radiation.ecrad_ifs import two_stream as ts
from legoesm.atmosphere.physics.radiation.rrtmgp.rte.two_stream import compute_heating_rate

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(_HERE, "oracle_fixtures", "ecrad_ifs49.npz")
GEN_FIXTURE = os.path.join(_HERE, "oracle_fixtures", "ecrad_ifs49_gen.npz")
_ALL = "/burg/glab/users/pg2328/oracle_audit/ifs_port/ecrad"
FIXTURE_ALL = os.path.join(_ALL, "ecrad_ifs49_all.npz")
GEN_FIXTURE_ALL = os.path.join(_ALL, "ecrad_ifs49_gen_all.npz")

# config49r1_rrtmg.nam
CLOUD_FRACTION_THRESHOLD = 1.0e-6
SW_ALBEDO_WAVELENGTH_BOUND = (0.25e-6, 0.44e-6, 0.69e-6, 1.19e-6, 2.38e-6)
I_SW_ALBEDO_INDEX = (1, 2, 3, 4, 5, 6)
TOL = 1.0e-12
FD_EPS_LW, FD_EPS_SW = 1e-5, 1e-4  # finite-difference steps from the measured sweep (vote N6b)
SW_FD_OD_MIN = 0.1  # SW check perturbs od > 0.1 only (measured: converges to 6e-8)
JIT_DEFAULT_ISA_BOUND = 1e-6  # vote N5'b; measured max 7.1e-8 (SW canopy diffuse)
HARD = math.inf

LW_KEYS = ("flux_up_lw", "flux_dn_lw", "flux_up_lw_clear", "flux_dn_lw_clear", "lw_derivative")
SW_KEYS = ("flux_up_sw", "flux_dn_sw", "flux_dn_direct_sw", "flux_up_sw_clear",
           "flux_dn_sw_clear", "flux_dn_direct_sw_clear")
SPEC_KEYS = ("spectral_flux_dn_sw_surf", "spectral_flux_dn_direct_sw_surf",
             "spectral_flux_dn_sw_surf_clear", "spectral_flux_dn_direct_sw_surf_clear",
             "canopy_flux_dn_diffuse_sw_surf", "canopy_flux_dn_direct_sw_surf")
ALL_KEYS = LW_KEYS + SW_KEYS + SPEC_KEYS


def load_inputs(path, gen_path):
    d = np.load(path)
    g = np.load(gen_path)
    assert np.array_equal(d["columns"], g["columns"])
    assert tuple(d["lb:dump:lw_planck_planck_hl"]) == (1, 1, 1)
    sw_scal = np.transpose(g["gen:sw_od_scaling"], (0, 2, 1))
    sw_tcc = g["gen:sw_total_cloud_cover"]
    # The oracle never runs the SW generator in night columns (NaN rows); the
    # solver never reads them there.  Zeros keep the dead branch finite.
    night = ~g["gen:sw_present"]
    assert np.all(d["props:cos_solar_zenith_angle"][night] <= 0.0)
    sw_scal = np.where(night[:, None, None], 0.0, sw_scal)
    sw_tcc = np.where(night, 0.0, sw_tcc)
    assert np.all(g["gen:lw_present"])
    inp = dict(
        lw=dict(
            od=d["props:od_lw"], od_cloud=d["props:od_lw_cloud"],
            ssa_cloud=d["props:ssa_lw_cloud"], g_cloud=d["props:asymmetry_lw_cloud"],
            # dump (ng, nlev+1, col) Fortran shape -> (col, nlev+1, ng)
            planck_hl=np.transpose(d["dump:lw_planck_planck_hl"], (2, 1, 0)),
            emission=d["props:lw_emission"], albedo=1.0 - d["props:lw_emissivity"],
            cloud_fraction=d["props:cloud_fraction"],
            od_scaling=np.transpose(g["gen:lw_od_scaling"], (0, 2, 1)),
            total_cloud_cover=g["gen:lw_total_cloud_cover"],
        ),
        sw=dict(
            od=d["props:od_sw"], ssa=d["props:ssa_sw"], g=d["props:asymmetry_sw"],
            od_cloud=d["props:od_sw_cloud"], ssa_cloud=d["props:ssa_sw_cloud"],
            g_cloud=d["props:asymmetry_sw_cloud"],
            albedo_direct=d["props:sw_albedo_direct"], albedo_diffuse=d["props:sw_albedo"],
            incoming_sw=d["props:incoming_sw"], cos_sza=d["props:cos_solar_zenith_angle"],
            cloud_fraction=d["props:cloud_fraction"], od_scaling=sw_scal,
            total_cloud_cover=sw_tcc,
        ),
    )
    return d, inp


def run_all(inp, sharpness):
    lw = ms.solver_mcica_lw(**inp["lw"], cloud_fraction_threshold=CLOUD_FRACTION_THRESHOLD,
                            sharpness=sharpness)
    sw = ms.solver_mcica_sw(**inp["sw"], cloud_fraction_threshold=CLOUD_FRACTION_THRESHOLD,
                            sharpness=sharpness)
    spec = ms.surface_spectral_sw(sw, SW_ALBEDO_WAVELENGTH_BOUND, I_SW_ALBEDO_INDEX)
    return {**lw, **sw, **spec}


def rel_err(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    den = np.maximum(np.abs(a), np.abs(b))
    e = np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0)
    return np.where(np.isfinite(a) & np.isfinite(b), e, np.inf)  # NaN/inf never pass


@pytest.fixture(scope="module")
def fixture8():
    d, inp = load_inputs(FIXTURE, GEN_FIXTURE)
    return d, inp, run_all(inp, HARD)


def _check_oracle(d, out):
    worst = {}
    for k in ALL_KEYS:
        e = rel_err(out[k], d["flux:" + k])
        worst[k] = float(e.max())
        print(f"MATCH {k:42s} max_rel={worst[k]:.3e}")
    for k in ("cloud_cover_lw", "cloud_cover_sw"):
        assert np.array_equal(np.asarray(out[k]), d["flux:" + k]), k
    bad = {k: v for k, v in worst.items() if not v <= TOL}
    assert not bad, bad


def test_saved_props_are_solver_inputs(fixture8):
    d, inp, _ = fixture8
    # lw_emissivity was saved as 1 - lw_albedo: the round trip is exact here.
    assert np.array_equal(inp["lw"]["albedo"], d["dump:in_lw_albedo"].T)
    # Ported sw_albedo_weights reproduce the solver's g-point albedos.
    for src, dst in (("in:sw_albedo", "props:sw_albedo"),
                     ("in:sw_albedo_direct", "props:sw_albedo_direct")):
        got = ms.sw_albedo_to_gpoints(jnp.asarray(d[src]),
                                      SW_ALBEDO_WAVELENGTH_BOUND, I_SW_ALBEDO_INDEX)
        assert rel_err(got, d[dst]).max() <= TOL, src
    assert np.array_equal(ms.band_from_g_lw()[[0, 9, 10, 139]], [0, 0, 1, 15])
    assert np.array_equal(ms.band_from_g_sw()[[0, 5, 6, 111]], [0, 0, 1, 13])


def test_oracle_match_hard_fixture(fixture8):
    d, _, out = fixture8
    _check_oracle(d, out)


@pytest.mark.skipif(not os.path.exists(FIXTURE_ALL), reason="32-column harness file absent")
def test_oracle_match_hard_32_columns():
    d, inp = load_inputs(FIXTURE_ALL, GEN_FIXTURE_ALL)
    _check_oracle(d, run_all(inp, HARD))


# Per-column max relative gap to the oracle at the DEFAULT sharpness (100),
# measured on the 8 fixture columns (RULE 5 KNOWN DIFFERENCE record).
# Measured sbatch 11021132 (votes M1-M5, N1'-N4), recomputed whenever a smooth form changes.
DEFAULT_GAPS = {
    'flux_up_lw': [7.309124273938559e-09, 1.0251059501695129e-08, 1.348507909236235e-08, 2.9269282071573153e-09, 3.768714450857213e-09, 1.2844924227847728e-08, 4.8725325788970095e-09, 2.0193734451794107e-09],
    'flux_dn_lw': [2.2146385965850413e-07, 9.53744502306379e-09, 6.756594656229991e-07, 1.7522152592465897e-07, 8.78180182953363e-08, 8.287970337084516e-07, 5.189312682818122e-08, 1.3670298810183044e-08],
    'flux_up_lw_clear': [7.305300717065507e-09, 1.0251059501695129e-08, 8.325490723253426e-09, 2.8569158923853446e-09, 3.208643502700523e-09, 1.2843964252424377e-08, 4.8426130115153424e-09, 1.607873018059488e-09],
    'flux_dn_lw_clear': [2.2146385965850413e-07, 9.53744502306379e-09, 6.756594656229991e-07, 1.7522152592465897e-07, 8.78180182953363e-08, 8.287970337084516e-07, 5.189312682818122e-08, 1.3670298810183044e-08],
    'lw_derivative': [1.505345470522907e-10, 2.60838346345577e-10, 1.5746082080475168e-09, 2.297879862569986e-13, 0.0, 5.915060993076138e-10, 1.1498630379642366e-12, 7.241587005044353e-10],
    'flux_up_sw': [0.0, 4.769154521014252e-10, 2.4382131905723283e-09, 1.0595769346371636e-10, 4.468956101537339e-09, 6.416826269066659e-11, 1.2021504963619622e-09, 1.942412605171353e-09],
    'flux_dn_sw': [0.0, 4.912796726499683e-10, 2.6517844742394586e-09, 8.343192608373146e-11, 2.7419342643253037e-09, 5.0223089377985256e-11, 7.522865298755225e-10, 1.9436739865394344e-09],
    'flux_dn_direct_sw': [0.0, 1.917318137878392e-16, 0.0, 0.0, 1.4864836679873395e-16, 0.0, 0.0, 0.0],
    'flux_up_sw_clear': [0.0, 4.769154521014252e-10, 1.7702277369486404e-09, 1.433890784199673e-10, 4.719034224647126e-09, 6.416885392437971e-11, 1.2110945666624388e-09, 1.984870823517572e-09],
    'flux_dn_sw_clear': [0.0, 4.912796726499683e-10, 2.246996514967228e-09, 8.898520686539749e-11, 2.87743144193981e-09, 5.0223312289817075e-11, 7.636312694651755e-10, 1.980309270198454e-09],
    'flux_dn_direct_sw_clear': [0.0, 1.917318137878392e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'spectral_flux_dn_sw_surf': [0.0, 9.559066542096899e-10, 1.0279541164169268e-08, 6.071518750257348e-10, 1.1143178438832675e-08, 1.4945479043348596e-10, 2.4361078424131763e-09, 8.479949995667246e-09],
    'spectral_flux_dn_direct_sw_surf': [0.0, 0.0, 2.484074359356578e-16, 0.0, 2.091694529976958e-16, 0.0, 0.0, 1.2347030889040472e-16],
    'spectral_flux_dn_sw_surf_clear': [0.0, 9.559066542096899e-10, 1.0333092896967225e-08, 6.445220784500239e-10, 1.229228807378642e-08, 1.4945637026440875e-10, 2.48079164611346e-09, 8.87075321888759e-09],
    'spectral_flux_dn_direct_sw_surf_clear': [0.0, 0.0, 1.6829961596712186e-16, 0.0, 0.0, 0.0, 0.0, 0.0],
    'canopy_flux_dn_diffuse_sw_surf': [0.0, 2.0105355760810277e-09, 1.1099490298573818e-08, 9.972643094666828e-10, 2.8800027344854974e-08, 9.744502957648079e-09, 1.0713387277170115e-08, 5.0429361001864604e-08],
    'canopy_flux_dn_direct_sw_surf': [0.0, 0.0, 1.6107942740963246e-16, 0.0, 0.0, 0.0, 0.0, 0.0],
}


def test_default_sharpness_gap_recorded(fixture8):
    d, inp, _ = fixture8
    out = run_all(inp, 100.0)
    measured = {}
    for k in ALL_KEYS:
        e = rel_err(out[k], d["flux:" + k])
        measured[k] = [float(x) for x in e.reshape(e.shape[0], -1).max(axis=1)]
        print(f"GAP {k} {measured[k]!r}")
    for k in ALL_KEYS:
        np.testing.assert_allclose(measured[k], DEFAULT_GAPS[k], rtol=1e-6, atol=1e-12, err_msg=k)


_JIT_SCRIPT = """
import sys, numpy as np, jax
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, sys.argv[1])
import test_ecrad_ifs_mcica_solver_oracle as t
_, inp = t.load_inputs(t.FIXTURE, t.GEN_FIXTURE)
out = jax.jit(lambda i: t.run_all(i, 100.0))(inp)
np.savez(sys.argv[2], **{k: np.asarray(out[k]) for k in t.ALL_KEYS})
"""


def _jit_outputs(tmp_path, xla_flags):
    import subprocess
    import sys

    path = tmp_path / f"jit{len(xla_flags)}.npz"
    env = dict(os.environ, XLA_FLAGS=xla_flags, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    subprocess.run([sys.executable, "-c", _JIT_SCRIPT, _HERE, str(path)], env=env, check=True)
    return np.load(path)


def test_eager_vs_jit(fixture8, tmp_path):
    """jit == eager to 1e-12 when XLA may not emit FMA (--xla_cpu_max_isa=AVX;
    the oracle is built with -ffp-contract=off).  At the default ISA XLA
    contracts a*b+c into FMA and the ill-conditioned SW thin-layer formulas
    amplify it (KNOWN DIFFERENCE, bounded by JIT_DEFAULT_ISA_BOUND)."""
    _, inp, _ = fixture8
    eager = run_all(inp, 100.0)
    no_fma = _jit_outputs(tmp_path, "--xla_cpu_max_isa=AVX")
    default = _jit_outputs(tmp_path, "")
    diff = {k: float(rel_err(eager[k], no_fma[k]).max()) for k in ALL_KEYS}
    gap = {k: float(rel_err(eager[k], default[k]).max()) for k in ALL_KEYS}
    print(f"EAGER_VS_JIT no-FMA max_rel={diff!r}")
    print(f"EAGER_VS_JIT default-ISA max_rel={gap!r}")
    assert max(diff.values()) <= TOL
    assert max(gap.values()) <= JIT_DEFAULT_ISA_BOUND

# --------------------------------------------------------------------------
# Energy conservation (outranks oracle fidelity)
# --------------------------------------------------------------------------

def test_sw_layers_absorb_nonnegative(fixture8):
    """No layer creates SW energy: net (dn - up) flux never increases downward."""
    _, _, out = fixture8
    for key in ("", "_clear"):
        fnet = np.asarray(out["flux_dn_sw" + key]) - np.asarray(out["flux_up_sw" + key])
        absorbed = fnet[:, :-1] - fnet[:, 1:]
        scale = np.max(np.abs(fnet), axis=1, keepdims=True) + 1.0
        assert np.all(absorbed >= -1e-12 * scale), float(absorbed.min())


def test_grad_finite_at_zero_gamma_sum():
    """ssa = g = 1 makes gamma1 + gamma2 = 0; the unused thick branch must not
    put NaN in the gradient (codex review finding)."""
    g = jax.grad(lambda w: sum(ts.calc_ref_trans_lw(1e-4, w, 1.0, 50.0, 55.0, 100.0)))(1.0)
    assert np.isfinite(float(g))


def test_heating_integral_equals_toa_minus_surface(fixture8):
    """Heating-rate conversion closes: column integral = TOA net - surface net.
    (Telescoping identity of the conversion; the solver energy checks are the
    continuity, absorption-sign and conservative-column tests.)"""
    d, _, out = fixture8
    p = d["props:pressure_hl"]
    dp = np.diff(p, axis=1)
    for up, dn in (("flux_up_lw", "flux_dn_lw"), ("flux_up_sw", "flux_dn_sw")):
        fnet = np.asarray(out[dn]) - np.asarray(out[up])
        hr = compute_heating_rate(jnp.asarray(fnet)[:, None, :], None,
                                  dp=jnp.pad(jnp.asarray(dp), ((0, 0), (0, 1)), constant_values=1.0)[:, None, :])
        hr = np.asarray(hr)[:, 0, :-1]
        integral = np.sum(hr * dp, axis=1) * constants.c_pd / constants.g
        expect = fnet[:, 0] - fnet[:, -1]
        scale = np.max(np.abs(fnet), axis=1) + 1.0
        assert np.all(np.abs(integral - expect) <= 1e-12 * scale * dp.shape[1]), up


def _layer_residuals_lw(col, inp):
    """Interface flux continuity of the cloudy LW adding solution, column ``col``."""
    lw = {k: jnp.asarray(v[col]) for k, v in inp["lw"].items()}
    band = ms.band_from_g_lw()
    pt, pb = lw["planck_hl"][:-1], lw["planck_hl"][1:]
    ocn = lw["od_scaling"] * lw["od_cloud"][:, band]
    odt = lw["od"] + ocn
    scat = lw["ssa_cloud"][:, band] * ocn
    ssa = jnp.where(odt > 0, scat / jnp.where(odt > 0, odt, 1.0), 0.0)
    R, T, su, sd = ts.calc_ref_trans_lw(odt, ssa, lw["g_cloud"][:, band], pt, pb, HARD)
    fup, fdn = ms.fast_adding_lw(R, T, su, sd, lw["emission"], lw["albedo"],
                                 jnp.ones(R.shape[0]))
    r_dn = fdn[1:] - (T * fdn[:-1] + R * fup[1:] + sd)
    r_up = fup[:-1] - (T * fup[1:] + R * fdn[:-1] + su)
    scale = float(jnp.max(fup))
    return float(jnp.max(jnp.abs(r_dn))) / scale, float(jnp.max(jnp.abs(r_up))) / scale


def test_flux_continuity_lw(fixture8):
    _, inp, _ = fixture8
    for col in range(inp["lw"]["od"].shape[0]):
        rd, ru = _layer_residuals_lw(col, inp)
        assert rd <= 1e-13 and ru <= 1e-13, (col, rd, ru)


def test_flux_continuity_and_conservative_column_sw(fixture8):
    _, inp, _ = fixture8
    sw = {k: jnp.asarray(v[4]) for k, v in inp["sw"].items()}
    mu0 = sw["cos_sza"]
    props = ts.calc_ref_trans_sw(mu0, sw["od"], sw["ssa"], sw["g"], HARD)
    fup, fdiff, fdir = ms.adding_sw(sw["incoming_sw"], sw["albedo_diffuse"], sw["albedo_direct"],
                                    mu0, *props)
    R, T, rd, td, _ = props
    fdn_beam = fdir / mu0
    r_dn = fdiff[1:] - (T * fdiff[:-1] + R * fup[1:] + td * fdn_beam[:-1])
    r_up = fup[:-1] - (T * fup[1:] + R * fdiff[:-1] + rd * fdn_beam[:-1])
    scale = float(jnp.max(sw["incoming_sw"]))
    assert float(jnp.max(jnp.abs(r_dn))) <= 1e-13 * scale
    assert float(jnp.max(jnp.abs(r_up))) <= 1e-13 * scale
    # Conservative scattering over a white surface: net flux zero everywhere.
    ones = jnp.ones_like(sw["od"])
    props = ts.calc_ref_trans_sw(mu0, sw["od"] + 1.0, ones, 0.5 * ones, HARD)
    alb = jnp.ones_like(sw["albedo_diffuse"])
    fup, fdiff, fdir = ms.adding_sw(sw["incoming_sw"], alb, alb, mu0, *props)
    net = fdiff + fdir - fup
    # At ssa = 1 the oracle's k floor (k = 1e-6) makes 1/(k+g1+(k-g1)e^2) a
    # cancellation of O(1) terms to O(k): round-off bound eps*g1/k per layer
    # (vote M9a).  Measured 3.2e-10 relative.
    g1 = 2.0 - 1.0 * (1.25 + 0.75 * 0.5)
    bound = np.finfo(np.float64).eps * (g1 / math.sqrt(ts._K_SQUARED_FLOOR)) * net.shape[0]
    print(f"CONSERVATIVE max|net|/incoming = {float(jnp.max(jnp.abs(net))) / scale:.3e} (bound {bound:.3e})")
    assert float(jnp.max(jnp.abs(net))) <= bound * scale


# --------------------------------------------------------------------------
# Differentiability across each smoothed transition
# --------------------------------------------------------------------------

def _finite_nonzero(gs):
    gs = np.asarray(gs)
    assert np.all(np.isfinite(gs)), gs
    assert np.all(gs != 0.0), gs


def test_grad_clamp_helpers_across_band():
    s = 100.0
    for lo, scale in ((ts._K_SQUARED_FLOOR, ts._K_SQUARED_FLOOR), (0.0, 1.0)):
        xs = lo - scale * np.array([0.2, 0.5, 0.8]) / s
        _finite_nonzero([jax.grad(lambda x: ts.lower_clamp(x, lo, scale, s))(x) for x in xs])
    xs = 1.0 + np.array([0.2, 0.5, 0.8]) / s
    _finite_nonzero([jax.grad(lambda x: ts.upper_clamp(x, 1.0, 1.0, s))(x) for x in xs])
    xs = 1e-3 * (1.0 + np.array([-0.4, 0.0, 0.4]) / s)
    _finite_nonzero([jax.grad(lambda x: ts.step(x, 1e-3, 1e-3, s, True))(x) for x in xs])


def test_grad_kernels_across_od_and_k_floor():
    s = 100.0
    pt, pb = 50.0, 55.0
    for od in 1e-3 * (1.0 + np.array([-0.4, 0.0, 0.4]) / s):
        _finite_nonzero(jax.grad(lambda o: sum(ts.calc_ref_trans_lw(o, 0.5, 0.8, pt, pb, s)))(od))
        _finite_nonzero(jax.grad(lambda o: sum(ts.calc_no_scattering_transmittance_lw(o, pt, pb, s)))(od))
    # k^2 = (1-ssa)*(4 - ssa) at g = 0: put k^2 inside the floor band.
    for frac in (0.2, 0.5, 0.8):
        one_minus = ts._K_SQUARED_FLOOR * (1.0 - frac / s) / 3.0
        ssa = 1.0 - one_minus
        _finite_nonzero(jax.grad(lambda w: sum(ts.calc_ref_trans_sw(0.6, 2.0, w, 0.0, s)[:2]))(ssa))
    check_grads(lambda o, w, g: sum(ts.calc_ref_trans_lw(o, w, g, pt, pb, s)), (0.3, 0.5, 0.8),
                order=1, modes=["rev"])
    check_grads(lambda o, w, g: sum(ts.calc_ref_trans_sw(0.6, o, w, g, s)), (0.3, 0.9, 0.8),
                order=1, modes=["rev"])


def test_grad_solver_cloud_switches(fixture8):
    _, inp, _ = fixture8
    s = 100.0
    thr = CLOUD_FRACTION_THRESHOLD
    col = 2
    lw = {k: jnp.asarray(v[col:col + 1]) for k, v in inp["lw"].items()}
    lev = int(np.argmax(np.asarray(lw["cloud_fraction"][0])))

    def toa_up(cf_lev, tcc):
        cf = lw["cloud_fraction"].at[0, lev].set(cf_lev)
        a = dict(lw, cloud_fraction=cf, total_cloud_cover=jnp.reshape(tcc, (1,)))
        o = ms.solver_mcica_lw(**a, cloud_fraction_threshold=thr, sharpness=s)
        return o["flux_up_lw"][0, 0] + o["lw_derivative"][0, 0]

    tcc0 = lw["total_cloud_cover"][0]
    _finite_nonzero([jax.grad(toa_up, 0)(thr * (1.0 + f / s), tcc0) for f in (-0.4, 0.0, 0.4)])
    _finite_nonzero([jax.grad(toa_up, 1)(1.0, thr * (1.0 + f / s)) for f in (-0.4, 0.0, 0.4)])
    _finite_nonzero([jax.grad(toa_up, 1)(1.0, 1.0 - thr * (1.0 + f / s)) for f in (-0.4, 0.0, 0.4)])
    # Relative (log-od) perturbations keep the finite-difference probe inside
    # od >= 0 (raw od spans 1e-15 .. 1e3).  Step sizes from the measured
    # eps sweep (vote N6b): LW converges to 1.3e-7 at eps 1e-5; the SW sum is
    # evaluation-noise limited (1 - exp(-2 k od) at od ~ 1e-9) and FD across the
    # narrow C1 clamp bands is truncation-limited, so its check perturbs only
    # od > 0.1 elements (sweep: 6e-8 at eps 1e-4; od > 1e-2 or > 1 do not converge).
    log_od = jnp.log(jnp.maximum(lw["od"], 1e-30))
    check_grads(lambda x: ms.solver_mcica_lw(**dict(lw, od=jnp.exp(x)), cloud_fraction_threshold=thr,
                                             sharpness=s)["flux_up_lw"].sum(),
                (log_od,), order=1, modes=["rev"], atol=1e-6, rtol=1e-6, eps=FD_EPS_LW)
    sw = {k: jnp.asarray(v[col:col + 1]) for k, v in inp["sw"].items()}
    mask = np.asarray(sw["od"]) > SW_FD_OD_MIN
    x0 = jnp.log(sw["od"][mask])
    check_grads(lambda x: ms.solver_mcica_sw(**dict(sw, od=sw["od"].at[mask].set(jnp.exp(x))),
                                             cloud_fraction_threshold=thr, sharpness=s)["flux_dn_sw"].sum(),
                (x0,), order=1, modes=["rev"], atol=1e-6, rtol=1e-6, eps=FD_EPS_SW)


def test_exp_libm_matches_oracle_libm():
    """exp_libm: bitwise glibc (the oracle's libm; math.exp is the same
    /lib64/libm-2.28.so) for |x| <= 0.01083."""
    rng = np.random.default_rng(0)
    mag = np.concatenate([np.logspace(-20, np.log10(0.01083), 150000),
                          rng.uniform(2.0**-28, 2.0**-18, 20000), rng.uniform(2.0**-18, 0.01083, 30000)])
    small = np.concatenate([-mag, mag, [0.0, -0.0, 2.0**-28, -(2.0**-18)]])
    f = jax.jit(ts.exp_libm)
    got = np.asarray(f(jnp.asarray(small)))
    glibc = np.array([math.exp(v) for v in small])
    assert np.array_equal(got, glibc), int(np.sum(got != glibc))

    edge = np.asarray(f(jnp.asarray([-746.5, -np.inf, 710.0, np.inf, np.nan])))
    assert edge[0] == 0.0 and edge[1] == 0.0 and np.isinf(edge[2]) and np.isinf(edge[3]) and np.isnan(edge[4])
    eager = np.asarray(ts.exp_libm(jnp.asarray(small)))  # no jit: no XLA FMA contraction
    assert np.array_equal(eager, np.array([math.exp(v) for v in small]))
    _finite_nonzero([jax.grad(ts.exp_libm)(v) for v in (-3.0, -1e-5, 0.0, -700.0)])


def test_exp_libm_correctly_rounded_outside_the_glibc_band():
    mpmath = pytest.importorskip("mpmath")
    rng = np.random.default_rng(0)
    big = np.concatenate([rng.uniform(-708.0, -0.0109, 20000), rng.uniform(0.0109, 709.0, 2000)])
    got = np.asarray(jax.jit(ts.exp_libm)(jnp.asarray(big)))
    with mpmath.workprec(200):
        cr = np.array([float(mpmath.exp(mpmath.mpf(v))) for v in big])
    assert np.array_equal(got, cr), int(np.sum(got != cr))
    glibc = np.array([math.exp(v) for v in big])
    print(f"EXP glibc != correctly-rounded for |x| > 0.01083: {np.mean(glibc != cr):.2e} of samples")


def test_grad_finite_dead_thin_branch_and_fp32_guards():
    """Codex round-2 findings: the unused thin branch at huge od (zero thin
    divisor) and the fp32 division / night floors must stay finite."""
    g = jax.grad(lambda w: sum(ts.calc_ref_trans_lw(1090512.5406112145, w, 0.9999999, 50.0, 55.0, 100.0)))(1.0)
    assert np.isfinite(float(g))
    z = jnp.float32(0.0)
    assert np.isfinite(float(ms._guarded_div(z, z, 100.0)))


def test_grad_finite_dead_cloudy_layer():
    """Codex round-3 finding: a clear layer with singular cloud optics
    (gas od 0, cloud ssa = g = 1) must not put NaN in the gradient."""
    nlev, ng = 3, ms.N_G_LW
    one = jnp.ones((1, nlev, 16))

    def f(ssa_c):
        o = ms.solver_mcica_lw(
            jnp.zeros((1, nlev, ng)), 0.1 * one, ssa_c * one, one,
            jnp.full((1, nlev + 1, ng), 50.0), jnp.full((1, ng), 5.0), jnp.zeros((1, ng)),
            jnp.zeros((1, nlev)), jnp.ones((1, nlev, ng)), jnp.zeros((1,)),
            cloud_fraction_threshold=CLOUD_FRACTION_THRESHOLD, sharpness=100.0)
        return o["flux_up_lw"].sum()

    assert np.isfinite(float(jax.grad(f)(1.0)))
    assert np.isinf(rel_err(np.nan, 1.0))


def test_finite_in_od_band_at_ssa_g_one():
    """Codex round-4 finding: ssa = g = 1 inside the od band (oracle thin
    branch finite) must give finite values and gradients (vote N9)."""
    out = ts.calc_ref_trans_lw(0.000999, 1.0, 1.0, 50.0, 55.0, 100.0)
    assert all(np.isfinite(float(v)) for v in out)
    g = jax.grad(lambda o: sum(ts.calc_ref_trans_lw(o, 1.0, 1.0, 50.0, 55.0, 100.0)))(0.000999)
    assert np.isfinite(float(g))


def test_grad_finite_dead_clear_derivative():
    """Codex round-5 finding: fully cloudy column whose (unused) clear-sky
    surface up-flux is 0 must not put 0/0 into the lw_derivative gradient."""
    nlev, ng = 1, ms.N_G_LW

    def f(planck, sharpness):
        o = ms.solver_mcica_lw(
            jnp.zeros((1, nlev, ng)), jnp.full((1, nlev, 16), 0.1), jnp.zeros((1, nlev, 16)),
            jnp.zeros((1, nlev, 16)), planck * jnp.ones((1, nlev + 1, ng)), jnp.zeros((1, ng)),
            jnp.ones((1, ng)), jnp.ones((1, nlev)), jnp.ones((1, nlev, ng)), jnp.ones((1,)),
            cloud_fraction_threshold=CLOUD_FRACTION_THRESHOLD, sharpness=sharpness)
        return o["lw_derivative"].sum()

    for sharpness in (100.0, HARD):
        assert np.isfinite(float(jax.grad(f)(50.0, sharpness)))


def test_grad_finite_tiny_gas_od_clear_column():
    """Codex round-6 finding: clear column with od = 1e-200 (fp64), at the hard
    form too, must give finite gradients (dead cloudy divisions)."""
    nlev, ng = 2, ms.N_G_LW

    def f(od, sharpness):
        o = ms.solver_mcica_lw(
            od * jnp.ones((1, nlev, ng)), jnp.ones((1, nlev, 16)), jnp.ones((1, nlev, 16)),
            jnp.ones((1, nlev, 16)), 5.0 * jnp.ones((1, nlev + 1, ng)), jnp.ones((1, ng)),
            jnp.zeros((1, ng)), jnp.zeros((1, nlev)), jnp.ones((1, nlev, ng)), jnp.zeros((1,)),
            cloud_fraction_threshold=CLOUD_FRACTION_THRESHOLD, sharpness=sharpness)
        return o["flux_up_lw"].sum()

    for sharpness in (100.0, HARD):
        assert np.isfinite(float(jax.grad(f)(1e-200, sharpness)))


def test_grad_finite_dead_sw_cloudy_optics():
    """Codex round-7 finding: dead SW cloudy combination must yield valid
    optics (ssa <= 1), so a clear layer with od=100, ssa=.02, g=.5 has a
    finite od gradient at both sharpness settings, day and night."""
    nlev, ng = 1, ms.N_G_SW

    def f(od, cos, sharpness):
        o = ms.solver_mcica_sw(
            od * jnp.ones((1, nlev, ng)), 0.02 * jnp.ones((1, nlev, ng)), 0.5 * jnp.ones((1, nlev, ng)),
            jnp.ones((1, nlev, 14)), jnp.ones((1, nlev, 14)), 0.5 * jnp.ones((1, nlev, 14)),
            0.2 * jnp.ones((1, ng)), 0.2 * jnp.ones((1, ng)), 10.0 * jnp.ones((1, ng)),
            jnp.reshape(cos, (1,)), jnp.zeros((1, nlev)), jnp.ones((1, nlev, ng)), jnp.zeros((1,)),
            cloud_fraction_threshold=CLOUD_FRACTION_THRESHOLD, sharpness=sharpness)
        return o["flux_up_sw"].sum()

    for sharpness in (100.0, HARD):
        for cos in (0.5, -0.3):
            assert np.isfinite(float(jax.grad(f)(100.0, cos, sharpness)))
