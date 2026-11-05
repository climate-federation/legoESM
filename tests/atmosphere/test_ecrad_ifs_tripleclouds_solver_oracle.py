"""Oracle test: ecRad 49r1 Tripleclouds LW/SW solvers (ecrad_ifs.tripleclouds_solvers).

The oracle is the 49r1 ecRad driver on the ecRad meridian file with the IFS
Tripleclouds settings (lw/sw_solver_name = "Tripleclouds",
cloud_fraction_threshold = 2.5e-5 as su_radiation_scheme.F90 sets them for
NLWSOLVER = NSWSOLVER = 3), aerosols off.  Inputs are the oracle's own solver
inputs (radiation_save writes them immediately before the solvers);
outputs are every flux:* variable of that run.  fp64 (JAX_ENABLE_X64=1).
"""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.atmosphere.physics.radiation.ecrad_ifs import mcica_solvers as ms
from legoesm.atmosphere.physics.radiation.ecrad_ifs import tripleclouds_solvers as tc

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(_HERE, "oracle_fixtures", "ecrad_ifs49_tripleclouds.npz")
_ALL = "/burg/glab/users/pg2328/oracle_audit/port_tripleclouds"
FIXTURE_ALL = os.path.join(_ALL, "ecrad_ifs49_tripleclouds_all.npz")
# Same run on the meridian file with an overcast liquid top layer (cf = 1,
# q_liquid = 1e-5 in layer 0): exercises the oracle's TOA quirk (below).
FIXTURE_TOPCLOUD_ALL = os.path.join(_ALL, "ecrad_ifs49_tripleclouds_topcloud_all.npz")
# Regimes the meridian file lacks (it has fractional_std = 1 everywhere):
# fractional_std over [0.2, 4.4] varying with height (both lower-fraction
# clamps and the interior), every 5th cloudy layer overcast, every 7th at
# cf = 3e-5; built by oracle_audit/port_tripleclouds/make_stress.py.
FIXTURE_STRESS_ALL = os.path.join(_ALL, "ecrad_ifs49_tripleclouds_stress_all.npz")

# su_radiation_scheme.F90:632 (Tripleclouds); radiation_config.F90:214 default.
CLOUD_FRACTION_THRESHOLD = 2.5e-5
DECORRELATION_SCALING = 0.5
SW_ALBEDO_WAVELENGTH_BOUND = (0.25e-6, 0.44e-6, 0.69e-6, 1.19e-6, 2.38e-6)
I_SW_ALBEDO_INDEX = (1, 2, 3, 4, 5, 6)
TOL = 1.0e-12
JIT_DEFAULT_ISA_BOUND = 1e-6  # McICA solver vote N5'b
# Finite-difference step from the measured sweep (fdprobe, job 11036214): at
# sharpness 100 central differences converge to the VJP as eps^2 down to 1e-6
# (rel 3e-9 SW, 5e-8 LW); the default 1e-4 is truncation-dominated (1e-4 rel).
FD_EPS = 1e-6
HARD = math.inf

LW_KEYS = ("flux_up_lw", "flux_dn_lw", "flux_up_lw_clear", "flux_dn_lw_clear", "lw_derivative")
SW_KEYS = ("flux_up_sw", "flux_dn_sw", "flux_dn_direct_sw", "flux_up_sw_clear",
           "flux_dn_sw_clear", "flux_dn_direct_sw_clear")
SPEC_KEYS = ("spectral_flux_dn_sw_surf", "spectral_flux_dn_direct_sw_surf",
             "spectral_flux_dn_sw_surf_clear", "spectral_flux_dn_direct_sw_surf_clear",
             "canopy_flux_dn_diffuse_sw_surf", "canopy_flux_dn_direct_sw_surf")
ALL_KEYS = LW_KEYS + SW_KEYS + SPEC_KEYS + ("cloud_cover_lw", "cloud_cover_sw")
# Measured production-sharpness gaps on the 8-column fixture (job 11036079), x2
# (vote T19a); fields not listed match at the hard bar.  The cloudy gaps equal
# the clear-sky ones: the two-stream kernels' bands, not the Tripleclouds switches.
_MEASURED_DEFAULT_GAPS = {
    "flux_up_lw": 1.285e-08,
    "flux_dn_lw": 8.288e-07,
    "flux_up_lw_clear": 1.284e-08,
    "flux_dn_lw_clear": 8.288e-07,
    "lw_derivative": 1.691e-08,
    "flux_up_sw": 4.159e-09,
    "flux_dn_sw": 2.728e-09,
    "flux_up_sw_clear": 4.719e-09,
    "flux_dn_sw_clear": 2.877e-09,
    "spectral_flux_dn_sw_surf": 1.093e-08,
    "spectral_flux_dn_sw_surf_clear": 1.229e-08,
    "canopy_flux_dn_diffuse_sw_surf": 3.569e-08,
}
DEFAULT_GAP_BOUND = {k: max(2.0 * _MEASURED_DEFAULT_GAPS.get(k, 0.0), TOL) for k in ALL_KEYS}


def load_inputs(path):
    d = np.load(path)
    p = lambda k: d["props:" + k]  # noqa: E731
    cloud = dict(cloud_fraction=p("cloud_fraction"), fractional_std=p("fractional_std"),
                 overlap_param=p("overlap_param"))
    inp = dict(
        lw=dict(od=p("od_lw"), od_cloud=p("od_lw_cloud"), ssa_cloud=p("ssa_lw_cloud"),
                g_cloud=p("asymmetry_lw_cloud"), planck_hl=p("planck_hl"),
                emission=p("lw_emission"), albedo=1.0 - p("lw_emissivity"), **cloud),
        sw=dict(od=p("od_sw"), ssa=p("ssa_sw"), g=p("asymmetry_sw"), od_cloud=p("od_sw_cloud"),
                ssa_cloud=p("ssa_sw_cloud"), g_cloud=p("asymmetry_sw_cloud"),
                albedo_direct=p("sw_albedo_direct"), albedo_diffuse=p("sw_albedo"),
                incoming_sw=p("incoming_sw"), cos_sza=p("cos_solar_zenith_angle"), **cloud),
    )
    return d, inp


def run_all(inp, sharpness):
    kw = dict(cloud_fraction_threshold=CLOUD_FRACTION_THRESHOLD,
              decorrelation_scaling=DECORRELATION_SCALING, sharpness=sharpness)
    lw = tc.solver_tripleclouds_lw(**inp["lw"], **kw)
    sw = tc.solver_tripleclouds_sw(**inp["sw"], **kw)
    spec = ms.surface_spectral_sw(sw, SW_ALBEDO_WAVELENGTH_BOUND, I_SW_ALBEDO_INDEX)
    return {**lw, **sw, **spec}


def rel_err(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    den = np.maximum(np.abs(a), np.abs(b))
    e = np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0)
    return np.where(np.isfinite(a) & np.isfinite(b), e, np.inf)  # NaN/inf never pass


def check_oracle(d, out):
    worst = {}
    for k in ALL_KEYS:
        worst[k] = float(rel_err(out[k], d["flux:" + k]).max())
        print(f"MATCH {k:42s} max_rel={worst[k]:.3e}")
    bad = {k: v for k, v in worst.items() if not v <= TOL}
    assert not bad, bad
    return worst


@pytest.fixture(scope="module")
def fixture8():
    d, inp = load_inputs(FIXTURE)
    return d, inp, run_all(inp, HARD)


def test_fixture_is_tripleclouds_cloudy_and_has_night(fixture8):
    d, inp, _ = fixture8
    cf = inp["lw"]["cloud_fraction"]
    # The fixture exercises cloudy, clear and sub-threshold layers and both day and night.
    assert np.any(cf >= CLOUD_FRACTION_THRESHOLD) and np.any(cf == 0.0)
    assert np.any(inp["sw"]["cos_sza"] < 1e-10) and np.any(inp["sw"]["cos_sza"] > 0.1)
    # Tripleclouds and McICA differ on cloudy columns: the oracle ran the solver named.
    assert np.any(d["flux:flux_up_sw"] != d["flux:flux_up_sw_clear"])


def test_oracle_match_hard_fixture(fixture8):
    d, _, out = fixture8
    check_oracle(d, out)


@pytest.mark.parametrize("path", [FIXTURE_ALL, FIXTURE_TOPCLOUD_ALL, FIXTURE_STRESS_ALL])
def test_oracle_match_hard_all_columns(path):
    if not os.path.exists(path):
        pytest.skip("32-column files are harness-only")
    d, inp = load_inputs(path)
    check_oracle(d, run_all(inp, HARD))


_JIT_SCRIPT = """
import sys, numpy as np, jax
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, sys.argv[1])
import test_ecrad_ifs_tripleclouds_solver_oracle as t
_, inp = t.load_inputs(t.FIXTURE)
out = jax.jit(lambda i: t.run_all(i, float(sys.argv[3])))(inp)
np.savez(sys.argv[2], **{k: np.asarray(out[k]) for k in t.ALL_KEYS})
"""


def _jit_outputs(tmp_path, xla_flags, sharpness):
    import subprocess
    import sys

    path = tmp_path / f"jit{len(xla_flags)}_{sharpness}.npz"
    env = dict(os.environ, XLA_FLAGS=xla_flags, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    subprocess.run([sys.executable, "-c", _JIT_SCRIPT, _HERE, str(path), str(sharpness)],
                   env=env, check=True)
    return np.load(path)


@pytest.mark.parametrize("sharpness", [HARD, 100.0])
def test_eager_vs_jit(fixture8, tmp_path, sharpness):
    """jit == eager to 1e-12 when XLA may not emit FMA (--xla_cpu_max_isa=AVX;
    the oracle is built with -ffp-contract=off).  At the default ISA XLA
    contracts a*b+c into FMA (KNOWN DIFFERENCE, bounded by
    JIT_DEFAULT_ISA_BOUND, the McICA solver's vote N5'b bound)."""
    _, inp, out = fixture8
    eager = out if sharpness == HARD else run_all(inp, sharpness)
    no_fma = _jit_outputs(tmp_path, "--xla_cpu_max_isa=AVX", sharpness)
    default = _jit_outputs(tmp_path, "", sharpness)
    diff = {k: float(rel_err(eager[k], no_fma[k]).max()) for k in ALL_KEYS}
    gap = {k: float(rel_err(eager[k], default[k]).max()) for k in ALL_KEYS}
    print(f"EAGER_VS_JIT s={sharpness} no-FMA max_rel={diff!r}")
    print(f"EAGER_VS_JIT s={sharpness} default-ISA max_rel={gap!r}")
    assert max(diff.values()) <= TOL
    assert max(gap.values()) <= JIT_DEFAULT_ISA_BOUND


def test_default_sharpness_gaps(fixture8):
    """KNOWN DIFFERENCE at the production sharpness: only inputs inside a band
    differ from the oracle; the measured per-field gaps are printed and pinned."""
    d, inp, _ = fixture8
    out = run_all(inp, 100.0)
    measured = {k: float(rel_err(out[k], d["flux:" + k]).max()) for k in ALL_KEYS}
    print(f"DEFAULT_GAPS {measured!r}")
    for k in ALL_KEYS:
        assert measured[k] <= DEFAULT_GAP_BOUND[k], (k, measured[k])


def test_sw_layers_absorb_nonnegative(fixture8):
    """No layer creates SW energy: net (dn - up) flux never increases downward."""
    _, _, out = fixture8
    for key in ("", "_clear"):
        fnet = np.asarray(out["flux_dn_sw" + key]) - np.asarray(out["flux_up_sw" + key])
        absorbed = fnet[:, :-1] - fnet[:, 1:]
        scale = np.max(np.abs(fnet), axis=1, keepdims=True) + 1.0
        assert np.all(absorbed[:, 1:] >= -1e-12 * scale), float(absorbed[:, 1:].min())


def _column(inp, side, icol):
    return {k: np.asarray(v)[icol:icol + 1] for k, v in inp[side].items()}


def test_overlap_matrices_conserve_flux(fixture8):
    _, inp, _ = fixture8
    cf = jnp.asarray(inp["lw"]["cloud_fraction"][0])
    fsd = jnp.asarray(inp["lw"]["fractional_std"][0])
    op = jnp.asarray(inp["lw"]["overlap_param"][0])
    fr, _ = tc.region_properties(cf, fsd, CLOUD_FRACTION_THRESHOLD, HARD)
    np.testing.assert_allclose(np.sum(fr, axis=-1), 1.0, rtol=0, atol=1e-15)
    u, v, _ = tc.overlap_matrices(fr, op, DECORRELATION_SCALING, CLOUD_FRACTION_THRESHOLD, HARD)
    clear = np.array([1.0, 0.0, 0.0])
    fu = np.concatenate([clear[None], np.asarray(fr)])
    fl = np.concatenate([np.asarray(fr), clear[None]])
    # Every occupied upper region hands all its flux to the regions below, and
    # every occupied lower region receives its source from the regions above.
    occupied_u = fu >= CLOUD_FRACTION_THRESHOLD
    occupied_l = fl >= CLOUD_FRACTION_THRESHOLD
    col_v = np.asarray(v).sum(axis=1)  # sum over lower of v(lower, upper)
    col_u = np.asarray(u).sum(axis=1)  # sum over upper of u(upper, lower)
    np.testing.assert_allclose(col_v[occupied_u], 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(col_u[occupied_l], 1.0, rtol=0, atol=1e-12)


def test_cloud_free_input_gives_clear_sky_fluxes(fixture8):
    _, inp, _ = fixture8
    kw = dict(cloud_fraction_threshold=CLOUD_FRACTION_THRESHOLD,
              decorrelation_scaling=DECORRELATION_SCALING, sharpness=HARD)
    lw_in = _column(inp, "lw", 1)
    sw_in = _column(inp, "sw", 1)
    for x in (lw_in, sw_in):
        x["cloud_fraction"] = np.zeros_like(x["cloud_fraction"])
    lw = tc.solver_tripleclouds_lw(**lw_in, **kw)
    sw = tc.solver_tripleclouds_sw(**sw_in, **kw)
    # Equal to rounding (XLA may contract the clear-sky a*b + c into an FMA).
    for k in ("flux_up_lw", "flux_dn_lw"):
        assert rel_err(lw[k], lw[k + "_clear"]).max() <= TOL, k
    for k in ("flux_up_sw", "flux_dn_sw", "flux_dn_direct_sw"):
        assert rel_err(sw[k], sw[k + "_clear"]).max() <= TOL, k
    assert float(lw["cloud_cover_lw"][0]) == 0.0


def test_homogeneous_overcast_equals_plane_parallel(fixture8):
    """fsd = 0, cloud fraction 1: both cloudy regions carry the in-cloud optics
    unscaled, so Tripleclouds equals one plane-parallel cloudy column (the McICA
    solver with od_scaling = 1 and total cover 1)."""
    _, inp, _ = fixture8
    kw = dict(cloud_fraction_threshold=CLOUD_FRACTION_THRESHOLD,
              decorrelation_scaling=DECORRELATION_SCALING, sharpness=HARD)
    icol = 2
    lw_in = _column(inp, "lw", icol)
    sw_in = _column(inp, "sw", icol)
    assert float(sw_in["cos_sza"][0]) > 0.1
    for x in (lw_in, sw_in):
        x["cloud_fraction"] = np.ones_like(x["cloud_fraction"])
        x["fractional_std"] = np.zeros_like(x["fractional_std"])
    lw = tc.solver_tripleclouds_lw(**lw_in, **kw)
    sw = tc.solver_tripleclouds_sw(**sw_in, **kw)
    nlev = lw_in["od"].shape[1]

    def mcica(x, ng):
        y = {k: v for k, v in x.items() if k not in ("fractional_std", "overlap_param")}
        y["od_scaling"] = np.ones((1, nlev, ng))
        y["total_cloud_cover"] = np.ones((1,))
        return y

    lw_pp = ms.solver_mcica_lw(**mcica(lw_in, ms.N_G_LW),
                               cloud_fraction_threshold=CLOUD_FRACTION_THRESHOLD, sharpness=HARD)
    sw_pp = ms.solver_mcica_sw(**mcica(sw_in, ms.N_G_SW),
                               cloud_fraction_threshold=CLOUD_FRACTION_THRESHOLD, sharpness=HARD)
    for k in ("flux_up_lw", "flux_dn_lw"):
        assert rel_err(lw[k], lw_pp[k]).max() <= 1e-12, k
    for k in ("flux_up_sw", "flux_dn_sw", "flux_dn_direct_sw"):
        assert rel_err(sw[k][:, 1:], sw_pp[k][:, 1:]).max() <= 1e-12, k
    # Oracle quirk (radiation_tripleclouds_sw.F90): the TOA up-flux multiplies
    # the top layer's region direct fluxes by the albedo of the regions ABOVE
    # TOA (clear only), so an overcast top layer reflects nothing at TOA.
    # Measured on the oracle (topcloud run): 0.0 at TOA, 157.05 W/m2 one level
    # down (column 8).  The port reproduces it.
    assert float(sw["flux_up_sw"][0, 0]) == 0.0
    assert float(sw_pp["flux_up_sw"][0, 0]) > 100.0


def test_inhomogeneity_brightens_less_than_homogeneous(fixture8):
    """Shonk & Hogan (2008): splitting the cloud into thin and thick regions
    reduces the reflected SW (plane-parallel albedo bias)."""
    _, inp, _ = fixture8
    kw = dict(cloud_fraction_threshold=CLOUD_FRACTION_THRESHOLD,
              decorrelation_scaling=DECORRELATION_SCALING, sharpness=HARD)
    sw_in = _column(inp, "sw", 2)
    up = []
    for fsd in (0.0, 1.0):
        x = dict(sw_in, fractional_std=np.full_like(sw_in["fractional_std"], fsd))
        up.append(float(tc.solver_tripleclouds_sw(**x, **kw)["flux_up_sw"][0, 0]))
    assert up[1] < up[0]


def _small_column(inp, side, icol, nlev_keep):
    """Top-truncated column: the lowest nlev_keep layers (surface kept)."""
    x = _column(inp, side, icol)
    out = {}
    for k, v in x.items():
        if k in ("planck_hl",):
            out[k] = v[:, -(nlev_keep + 1):]
        elif k == "overlap_param":
            out[k] = v[:, -(nlev_keep - 1):]
        elif v.ndim >= 2 and v.shape[1] == x["cloud_fraction"].shape[1]:
            out[k] = v[:, -nlev_keep:]
        else:
            out[k] = v
    return out


@pytest.mark.parametrize("side", ["lw", "sw"])
def test_gradients_finite_and_check_grads(fixture8, side):
    _, inp, _ = fixture8
    kw = dict(cloud_fraction_threshold=CLOUD_FRACTION_THRESHOLD,
              decorrelation_scaling=DECORRELATION_SCALING, sharpness=100.0)
    # The column whose lowest 14 layers hold the most cloud optical depth.
    icol = int(np.argmax(np.asarray(inp[side]["od_cloud"])[:, -14:].sum(axis=(1, 2))))
    x = _small_column(inp, side, icol, 24)
    solver = tc.solver_tripleclouds_lw if side == "lw" else tc.solver_tripleclouds_sw
    key = "flux_up_lw" if side == "lw" else "flux_up_sw"

    def toa(cf, fsd, op):
        y = dict(x, cloud_fraction=cf, fractional_std=fsd, overlap_param=op)
        return solver(**y, **kw)[key][0, 0]

    cf = jnp.asarray(x["cloud_fraction"])
    # One layer inside each smoothed transition: at the region threshold, just
    # above zero (layer presence), and an overcast layer.
    cf = cf.at[0, 3].set(CLOUD_FRACTION_THRESHOLD).at[0, 5].set(1e-9).at[0, 7].set(1.0)
    # The inhomogeneity only acts where the layer holds cloud optical depth.
    j = 10 + int(np.argmax(np.asarray(x["od_cloud"])[0, 10:].sum(axis=-1)))
    assert float(np.asarray(x["od_cloud"])[0, j].sum()) > 0.0
    cf = cf.at[0, j].set(0.4)
    fsd = jnp.asarray(x["fractional_std"]).at[0, j].set(1.5)  # lower-fraction clamp corner
    op = jnp.asarray(x["overlap_param"])
    grads = jax.grad(toa, argnums=(0, 1, 2))(cf, fsd, op)
    for gr in grads:
        assert np.all(np.isfinite(np.asarray(gr)))
    assert float(jnp.abs(grads[0][0, j])) > 0.0 and float(jnp.abs(grads[1][0, j])) > 0.0
    # Smooth interior point: moderate cloud everywhere.
    cf_in = jnp.full_like(cf, 0.3)
    fsd_in = jnp.full_like(fsd, 0.8)
    check_grads(lambda c, f: toa(c, f, op), (cf_in, fsd_in), order=1, modes=("rev",),
                atol=1e-6, rtol=1e-4, eps=FD_EPS)


def test_max_kink_exact():
    a = jnp.array([0.3, 0.5, 0.0, 1.0, 0.2])
    b = jnp.array([0.3, 0.1, 0.0, 1.0, 0.2005])
    got = np.asarray(tc.max_kink_exact(a, b, 100.0))
    want = np.maximum(np.asarray(a), np.asarray(b))
    assert np.array_equal(got[:4], want[:4])  # exact at a = b, 0, 1 and outside the band
    assert want[4] - 4.0 / 27.0 * 0.4005 / 100.0 <= got[4] <= want[4]  # inside the band
    g = jax.grad(lambda x: tc.max_kink_exact(x, 0.3, 100.0))(0.3)
    assert float(g) == 0.5  # symmetric subgradient at the kink
    np.testing.assert_array_equal(np.asarray(tc.max_kink_exact(a, b, HARD)), want)


def test_night_zero_scattering_gradient_finite(fixture8):
    """Night column, cloudy layers, no scattering: the unused cloudy-optics
    division must not put NaN in the gradient (codex review finding)."""
    _, inp, _ = fixture8
    kw = dict(cloud_fraction_threshold=CLOUD_FRACTION_THRESHOLD,
              decorrelation_scaling=DECORRELATION_SCALING, sharpness=100.0)
    x = _small_column(inp, "sw", 2, 12)
    x.update(cos_sza=np.array([-0.5]), cloud_fraction=np.full_like(x["cloud_fraction"], 0.5),
             ssa=np.zeros_like(x["ssa"]), ssa_cloud=np.zeros_like(x["ssa_cloud"]))

    def f(od):
        return tc.solver_tripleclouds_sw(**dict(x, od=od), **kw)["flux_up_sw"].sum()

    g = jax.grad(f)(jnp.asarray(x["od"]))
    assert np.all(np.isfinite(np.asarray(g)))
