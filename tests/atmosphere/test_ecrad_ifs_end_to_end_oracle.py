"""Oracle test: ecRad 49r1 end-to-end call (ecrad_ifs.ecrad.radiation).

From the oracle run's INPUTS only -- the input file (``in:*``), the gas mass
mixing ratios ecRad hands to its gas optics (``dump:in_gas_mmr``: the driver's
vmr -> mmr conversion is not part of the port) and the replayed McICA uniform
stream (``rng:*``) -- to every ``flux:*`` output, at ``sharpness = inf``.
fp64 (JAX_ENABLE_X64=1).  The 32-column file runs when present (sbatch).

Stream replay: each oracle block (one per ``initialize_random_numbers`` call)
is placed into the generator layout as in the generator test, except that
the cloudy runs that locate each inhomogeneity fill come from the PORT's own
generator, g-point by g-point (cloud occurrence does not depend on the
inhomogeneity draws), so no oracle intermediate is needed.  The recorded
fill sizes must agree, and where the generator dump exists the port's mask
is checked equal to the oracle's.

Bar: max |a-b| / max(|a|,|b|) <= 1e-12 per element (cloud covers exact).
"""

import json
import math
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.radiation.ecrad_ifs import cloud_optics as co
from legoesm.atmosphere.physics.radiation.ecrad_ifs import ecrad
from legoesm.atmosphere.physics.radiation.ecrad_ifs import mcica_generator as mg
from legoesm.atmosphere.physics.radiation.ecrad_ifs import rrtmg_prepare as prep
from legoesm.atmosphere.physics.radiation.rrtmgp.rte.two_stream import compute_heating_rate
from tests.atmosphere.test_ecrad_ifs_mcica_generator_oracle import _blocks, _seeds
from tests.atmosphere.test_ecrad_ifs_mcica_solver_oracle import rel_err

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(_HERE, "oracle_fixtures", "ecrad_ifs49.npz")
GEN_FIXTURE = os.path.join(_HERE, "oracle_fixtures", "ecrad_ifs49_gen.npz")
_ALL = "/burg/glab/users/pg2328/oracle_audit/ifs_port/ecrad"
FIXTURE_ALL = os.environ.get("ECRAD_IFS49_ALL", os.path.join(_ALL, "ecrad_ifs49_all.npz"))
GEN_FIXTURE_ALL = os.environ.get("ECRAD_IFS49_GEN_ALL", os.path.join(_ALL, "ecrad_ifs49_gen_all.npz"))
TOL = 1.0e-12
# Vote V50 (coordinator + codex + GLM): campaign bar rel <= 1e-12 OR within the
# oracle's empirical 20-seed 1-ulp round-off spread. canopy_flux_dn_diffuse_sw_surf
# misses 1e-12 (1.77e-12: diffuse = total - direct, total/diffuse = 7350 at the
# worst element) and gets its measured spread, read from the committed spread
# artifact (copy of oracle_audit/spread/ecrad_ifs49/runs/spread_ecrad_ifs49.json,
# controls C1-C5 PASS), never retyped. Every other field stays at 1e-12.
SPREAD_ARTIFACT = os.path.join(_HERE, "oracle_fixtures", "spread_ecrad_ifs49.json")
_V50_FIELDS = ("canopy_flux_dn_diffuse_sw_surf",)
# Vote V53: eager-vs-jit bound for the same field is 5e-12 (port-vs-port, not the
# oracle spread); every other field stays at 1e-12.
_V53_JIT_TOL = {"canopy_flux_dn_diffuse_sw_surf": 5.0e-12}


def tol(key):
    if key not in _V50_FIELDS:
        return TOL
    with open(SPREAD_ARTIFACT) as f:
        art = json.load(f)
    assert all(art["controls"][c] == "PASS" for c in ("C1", "C2", "C3", "C4", "C5"))
    return max(TOL, float(art["spread"][key]))
HARD = math.inf
FILL = 0.5  # uniform in slots the oracle never drew (never read: generator test)
NG = {"lw": ecrad.N_G_LW, "sw": ecrad.N_G_SW}
FLUX_KEYS = tuple(k for k in ecrad.OUTPUT_KEYS if not k.startswith("cloud_cover"))


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def inputs(z):
    f = lambda k: np.asarray(z["in:" + k], np.float64)  # noqa: E731
    return dict(
        pressure_hl=f("pressure_hl"), temperature_hl=f("temperature_hl"),
        gas_mmr=np.asarray(z["dump:in_gas_mmr"], np.float64),
        skin_temperature=f("skin_temperature"), cos_sza=f("cos_solar_zenith_angle"),
        solar_irradiance=float(z["in:solar_irradiance"]), sw_albedo=f("sw_albedo"),
        sw_albedo_direct=f("sw_albedo_direct"), lw_emissivity=f("lw_emissivity"),
        cloud_fraction=f("cloud_fraction"), q_liquid=f("q_liquid"), q_ice=f("q_ice"),
        re_liquid=f("re_liquid"), re_ice=f("re_ice"), overlap_param=f("overlap_param"),
        fractional_std=f("fractional_std"),
    )


def replay(z, inp, spec):
    """Oracle stream -> generator uniforms (4-tuple), the per-column cloud
    mask (ncol, ng, nlev) used to place the fills, and the count placed."""
    frac = np.asarray(co.crop_cloud_fraction(inp["cloud_fraction"], inp["q_liquid"],
                                             inp["q_ice"], HARD)[0])
    ncol, nlev = frac.shape
    ng = NG[spec]
    seeds = _seeds(z, spec)
    blocks = dict(_blocks(z))
    top = np.full((ncol, ng), FILL)
    cloud, inh1, inh2 = (np.full((ncol, ng, nlev), FILL) for _ in range(3))
    mask = np.zeros((ncol, ng, nlev), bool)
    calls = {c: list(blocks[seeds[c]]) for c in range(ncol) if seeds[c] in blocks}
    placed = 0
    for c, cl in calls.items():
        v = cl.pop(0)
        assert v.size == ng
        top[c] = v
        placed += ng
    pos = {c: np.flatnonzero(frac[c] > 0.0) for c in calls}
    one = jax.jit(lambda rt, rc: mg.cloud_generator(
        frac, inp["overlap_param"], inp["fractional_std"], rt, rc, jnp.full_like(rc, FILL),
        jnp.full_like(rc, FILL), decorrelation_scaling=ecrad._CLOUD_INHOM_DECORR_SCALING,
        frac_threshold=ecrad._CLOUD_FRACTION_THRESHOLD, sharpness=HARD)[0][:, 0, :])
    for jg in range(ng):
        itrigs = {}
        for c in calls:
            rc = calls[c].pop(0)
            iend = pos[c][-1]
            itrigs[c] = itrig = iend + 1 - rc.size
            assert pos[c][0] <= itrig <= iend
            cloud[c, jg, itrig:iend + 1] = rc
            placed += rc.size
        m = np.asarray(one(top[:, jg:jg + 1], cloud[:, jg:jg + 1, :])) > 0.0
        for c in calls:
            iend, itrig, cl = pos[c][-1], itrigs[c], m[c]
            mask[c, jg] = cl
            assert not cl[:itrig].any() and not cl[iend + 1:].any() and cl[itrig]
            edges = np.flatnonzero(np.diff(np.concatenate([[0], cl[itrig:iend + 1].astype(int), [0]])))
            runs = [(itrig + a, b - a) for a, b in zip(edges[::2], edges[1::2])]
            if not cl[iend]:
                runs.append((iend + 1, 0))  # final fill of zero layers (jlev = iend+1)
            for a, n in runs:
                r1, r2 = calls[c].pop(0), calls[c].pop(0)
                assert r1.size == n and r2.size == n, "fill sizes disagree with port cloud runs"
                inh1[c, jg, a:a + n] = r1
                inh2[c, jg, a:a + n] = r2
                placed += 2 * n
    assert all(not cl for cl in calls.values()), "stream calls left over"
    return (top, cloud, inh1, inh2), mask, placed


def run(z, sharpness=HARD, jit=False):
    inp = inputs(z)
    rlw, mlw, n_lw = replay(z, inp, "lw")
    rsw, msw, n_sw = replay(z, inp, "sw")
    assert n_lw + n_sw == z["rng:values"].size, "stream not fully consumed"
    f = lambda i, a, b: ecrad.radiation(**i, uniforms_lw=a, uniforms_sw=b, sharpness=sharpness)  # noqa: E731
    out = (jax.jit(f) if jit else f)(inp, rlw, rsw)
    return {k: np.asarray(v) for k, v in out.items()}, {"lw": mlw, "sw": msw}


def check_oracle(z, out):
    worst = {}
    for k in FLUX_KEYS:
        worst[k] = float(rel_err(out[k], z["flux:" + k]).max())
        print(f"E2E {k:40s} max_rel={worst[k]:.3e}")
    for k in ("cloud_cover_lw", "cloud_cover_sw"):
        assert np.array_equal(out[k], z["flux:" + k]), k
    bad = {k: v for k, v in worst.items() if not v <= tol(k)}
    assert not bad, bad
    return worst


def check_mask(z, gen_path, masks):
    if not os.path.exists(gen_path):
        return
    g = load(gen_path)
    assert np.array_equal(g["columns"], z["columns"])
    for spec, m in masks.items():
        ok = g[f"gen:{spec}_present"] & (np.nan_to_num(g[f"gen:{spec}_total_cloud_cover"])
                                          >= ecrad._CLOUD_FRACTION_THRESHOLD)
        assert np.array_equal(m[ok], g[f"gen:{spec}_od_scaling"][ok] > 0.0), spec


@pytest.fixture(scope="module")
def z8():
    return load(FIXTURE)


@pytest.fixture(scope="module")
def hard8(z8):
    return run(z8)


def test_inputs_reach_gas_optics_unchanged(z8):
    """The oracle's gas-optics inputs equal the file inputs / their full levels,
    and the ported get_albedos reproduces its LW albedo bitwise."""
    inp = inputs(z8)
    for a, b in (("pressure_hl", "in_pressure_hl"), ("temperature_hl", "in_temperature_hl"),
                 ("cos_sza", "in_cos_sza"), ("skin_temperature", "in_skin_temperature")):
        assert np.array_equal(inp[a], z8["dump:" + b]), a
    assert inp["solar_irradiance"] == float(z8["dump:in_solar_irradiance"])
    pfl, tfl = prep.full_levels(jnp.asarray(inp["pressure_hl"]), jnp.asarray(inp["temperature_hl"]))
    assert np.array_equal(np.asarray(pfl), z8["dump:in_pressure_fl"])
    assert np.array_equal(np.asarray(tfl), z8["dump:in_temperature_fl"])
    lw_alb = np.asarray(ecrad.lw_albedo_from_emissivity(jnp.asarray(inp["lw_emissivity"])))
    assert np.array_equal(lw_alb, z8["dump:in_lw_albedo"].T)
    # namelist values agree across modules (one threshold, not two)
    assert ecrad._CLOUD_FRACTION_THRESHOLD == co._CLOUD_FRACTION_THRESHOLD
    # inputs are top-down (radiation_reverse is not ported)
    assert np.all(inp["pressure_hl"][:, 1] > inp["pressure_hl"][:, 0])


def test_oracle_match_hard_fixture(z8, hard8):
    out, masks = hard8
    check_mask(z8, GEN_FIXTURE, masks)
    check_oracle(z8, out)


@pytest.mark.skipif(not os.path.exists(FIXTURE_ALL), reason="32-column harness file absent")
def test_oracle_match_hard_32_columns():
    z = load(FIXTURE_ALL)
    out, masks = run(z)
    check_mask(z, GEN_FIXTURE_ALL, masks)
    check_oracle(z, out)


# Per-column max relative gap to the oracle at the DEFAULT sharpness (100) on
# the 8 fixture columns (RULE 5 KNOWN DIFFERENCE record; production smoothing
# widths are an open user question). Measured sbatch 11026118; recompute
# whenever a section's smooth form changes. Pinned as in the solver test.
DEFAULT_GAPS = {
    'flux_up_lw': [0.0012578697473902495, 2.4235961613014467e-06, 0.002134325245562061, 0.0061434461096185715, 0.02115560589330642, 0.00012595005759272096, 0.0014201399351165698, 0.0008426348938212514],
    'flux_dn_lw': [0.007685855384476735, 5.223771874911262e-05, 0.0130099981086808, 0.01457715367511537, 0.015572551815942243, 9.076431125191096e-05, 0.015506313297078697, 0.00606402650582211],
    'flux_up_lw_clear': [3.5391619648350542e-06, 2.4235961613014467e-06, 5.360573053934103e-05, 0.00012073010171306553, 0.00010702229140577257, 0.00012595359507173998, 0.00010301465350212079, 1.1711207056527021e-05],
    'flux_dn_lw_clear': [6.773315878025866e-05, 5.223771874911262e-05, 8.554092906394421e-05, 0.00010874221517496763, 0.00012928095169025284, 9.078288146705561e-05, 8.051725468307255e-05, 8.246419097094804e-05],
    'lw_derivative': [0.006415788309833112, 6.481270067580142e-06, 0.033941376091203256, 0.22964996275097088, 0.06927417457421073, 0.00021766391217546168, 0.05813793736708698, 0.003170409909060478],
    'flux_up_sw': [0.0, 0.0013596679920680193, 0.055770850000791204, 0.14631874737099498, 0.15486297904176918, 0.0019983832805201963, 0.15290784027178217, 0.0007903541081488703],
    'flux_dn_sw': [0.0, 0.0014646201694471386, 0.00517383371183306, 0.038600148714628246, 0.008972955845354489, 0.000999213684108301, 0.016839330199399125, 0.0013397522184788772],
    'flux_dn_direct_sw': [0.0, 0.0031023663471891074, 0.01843428963809294, 0.1158412573158844, 0.028582314186045996, 0.0010597839199558542, 0.012317254298769132, 0.0061798681100526665],
    'flux_up_sw_clear': [0.0, 0.0013596679920680193, 0.0005906548691084819, 0.0012588702680132397, 0.0013092029467996003, 0.001998388752903785, 0.0008330093955153106, 0.00037469671433790994],
    'flux_dn_sw_clear': [0.0, 0.0014646201694471386, 0.0005306758808448349, 0.0012295089843974635, 0.0013295200760398656, 0.0009991671833098409, 0.0007966784967105224, 0.0002791155086802254],
    'flux_dn_direct_sw_clear': [0.0, 0.0031023663471891074, 0.0005955626257251675, 0.0013318627013142303, 0.001415400991345381, 0.0010597863120890992, 0.0008447042633375168, 0.00030790601138551286],
    'spectral_flux_dn_sw_surf': [0.0, 0.057038843043944694, 0.11815308698806382, 0.24769740584584707, 0.25768285018624715, 0.07531910699856961, 0.05367792430856638, 0.01858438418644921],
    'spectral_flux_dn_direct_sw_surf': [0.0, 0.020567373477899095, 0.1812737281602904, 0.34178283003715165, 0.3289299632809617, 0.07529453831025018, 0.09811104322357485, 0.0629689545092887],
    'spectral_flux_dn_sw_surf_clear': [0.0, 0.057038843043944694, 0.09419746435213736, 0.08146211496764229, 0.08389949977043759, 0.07531911272229283, 0.05400381408395227, 0.030990954699389905],
    'spectral_flux_dn_direct_sw_surf_clear': [0.0, 0.020567373477899095, 0.11235665421912815, 0.08143869106002753, 0.0838734871197268, 0.07529454695102085, 0.05398866483503893, 0.03098349458613841],
    'canopy_flux_dn_diffuse_sw_surf': [0.0, 0.057038843043944666, 0.09102972874374893, 0.2542132893037136, 0.28019208117773414, 0.07536170716240122, 0.17951320072192006, 0.11334763983613591],
    'canopy_flux_dn_direct_sw_surf': [0.0, 0.008769099825685753, 0.0937768208505738, 0.15747143572422442, 0.0895912906154727, 0.07529453831025015, 0.05302527270619332, 0.054589293948873775],
}


def test_default_sharpness_gap_recorded(z8):
    out, _ = run(z8, 100.0)
    for k in FLUX_KEYS:
        e = rel_err(out[k], z8["flux:" + k])
        got = [float(x) for x in e.reshape(e.shape[0], -1).max(axis=1)]
        print(f"GAP100 {k:40s} max_rel={max(got):.3e}")
        np.testing.assert_allclose(got, DEFAULT_GAPS[k], rtol=1e-6, atol=1e-12, err_msg=k)


# --------------------------------------------------------------------------
# Energy consistency
# --------------------------------------------------------------------------

def test_sw_layers_absorb_nonnegative(hard8):
    """No layer creates SW energy: net (dn - up) flux never increases downward
    (all-sky and clear; same check and bound as the solver test)."""
    out, _ = hard8
    for key in ("", "_clear"):
        fnet = out["flux_dn_sw" + key] - out["flux_up_sw" + key]
        absorbed = fnet[:, :-1] - fnet[:, 1:]
        scale = np.max(np.abs(fnet), axis=1, keepdims=True) + 1.0
        assert np.all(absorbed >= -TOL * scale), float(absorbed.min())


def test_heating_integral_equals_net_flux_divergence(z8, hard8):
    """Column integral of the heating rate = TOA net - surface net (LW, SW, clear).
    This is the telescoping identity of the flux -> heating conversion (it checks
    the conversion, not the solver); the solver energy checks are the absorption
    sign test above and the solver test's continuity/conservative-column tests."""
    out, _ = hard8
    dp = np.diff(inputs(z8)["pressure_hl"], axis=1)
    for up, dn in (("flux_up_lw", "flux_dn_lw"), ("flux_up_sw", "flux_dn_sw"),
                   ("flux_up_lw_clear", "flux_dn_lw_clear"), ("flux_up_sw_clear", "flux_dn_sw_clear")):
        fnet = out[dn] - out[up]
        hr = compute_heating_rate(jnp.asarray(fnet)[:, None, :], None,
                                  dp=jnp.pad(jnp.asarray(dp), ((0, 0), (0, 1)), constant_values=1.0)[:, None, :])
        hr = np.asarray(hr)[:, 0, :-1]
        integral = np.sum(hr * dp, axis=1) * constants.c_pd / constants.g
        expect = fnet[:, 0] - fnet[:, -1]
        scale = np.max(np.abs(fnet), axis=1) + 1.0
        assert np.all(np.abs(integral - expect) <= TOL * scale * dp.shape[1]), up


def test_toa_and_surface_consistency(z8, hard8):
    out, _ = hard8
    inp = inputs(z8)
    mu0, s0 = inp["cos_sza"], inp["solar_irradiance"]
    day = mu0 > 0.0
    # TOA: no downwelling LW; SW down = direct = S0 * mu0 by day, zero at night.
    for k in ("flux_dn_lw", "flux_dn_lw_clear"):
        assert np.all(out[k][:, 0] == 0.0), k
    for k in ("flux_dn_sw", "flux_dn_direct_sw", "flux_dn_sw_clear", "flux_dn_direct_sw_clear"):
        assert rel_err(out[k][day, 0], s0 * mu0[day]).max() <= TOL, k
        assert np.all(out[k][~day] == 0.0), k
    # Surface: band and canopy fluxes sum to the broadband surface flux.
    surf = {"": out["flux_dn_sw"][:, -1], "_direct": out["flux_dn_direct_sw"][:, -1]}
    for sfx, ref in surf.items():
        assert rel_err(out[f"spectral_flux_dn{sfx}_sw_surf"].sum(1), ref).max() <= TOL, sfx
        clr = out[f"flux_dn{sfx}_sw_clear"][:, -1]
        assert rel_err(out[f"spectral_flux_dn{sfx}_sw_surf_clear"].sum(1), clr).max() <= TOL, sfx
    canopy = out["canopy_flux_dn_diffuse_sw_surf"] + out["canopy_flux_dn_direct_sw_surf"]
    assert rel_err(canopy.sum(1), surf[""]).max() <= TOL
    assert rel_err(out["canopy_flux_dn_direct_sw_surf"].sum(1), surf["_direct"]).max() <= TOL
    # lw_derivative is d flux_up / d surface flux_up: exactly 1 at the surface.
    assert np.all(out["lw_derivative"][:, -1] == 1.0)


# --------------------------------------------------------------------------
# Differentiability (default sharpness, production uniforms)
# --------------------------------------------------------------------------

def test_grad_flux_functional(z8):
    """d(TOA + surface net fluxes)/d(T, q, cloud fraction, q_liquid): finite
    everywhere and nonzero for each input, on two cloudy daylight columns."""
    inp = inputs(z8)
    cols = np.flatnonzero((inp["cos_sza"] > 0.0) & (inp["q_liquid"].max(1) > 0.0))[:2]
    assert cols.size == 2
    sub = {k: (v[cols] if isinstance(v, np.ndarray) else v) for k, v in inp.items()}
    iseed = np.asarray(z8["in:iseed"])[cols].astype(np.int64)
    ulw, usw = ecrad.draw_uniforms(iseed, sub["cloud_fraction"].shape[1])

    def functional(t_hl, q, cf, ql):
        gas = jnp.asarray(sub["gas_mmr"]).at[..., prep.IH2O].set(q)
        o = ecrad.radiation(**dict(sub, temperature_hl=t_hl, gas_mmr=gas, cloud_fraction=cf,
                                   q_liquid=ql), uniforms_lw=ulw, uniforms_sw=usw)
        net = lambda u, d: o[d] - o[u]  # noqa: E731
        return (jnp.sum(net("flux_up_lw", "flux_dn_lw")[:, [0, -1]])
                + jnp.sum(net("flux_up_sw", "flux_dn_sw")[:, [0, -1]]))

    args = (sub["temperature_hl"], sub["gas_mmr"][..., prep.IH2O], sub["cloud_fraction"], sub["q_liquid"])
    grads = jax.grad(functional, argnums=(0, 1, 2, 3))(*map(jnp.asarray, args))
    for name, g in zip(("temperature_hl", "q", "cloud_fraction", "q_liquid"), grads):
        g = np.asarray(g)
        print(f"GRAD {name}: max|g|={np.abs(g).max():.3e} nonzero={np.count_nonzero(g)}/{g.size}")
        assert np.all(np.isfinite(g)), name
        assert np.any(g != 0.0), name


# --------------------------------------------------------------------------
# eager vs jit
# --------------------------------------------------------------------------

_JIT_SCRIPT = """
import sys, numpy as np, jax
jax.config.update("jax_enable_x64", True)
from tests.atmosphere import test_ecrad_ifs_end_to_end_oracle as t
out, _ = t.run(t.load(t.FIXTURE), float(sys.argv[2]), jit=True)
np.savez(sys.argv[1], **out)
"""


@pytest.mark.parametrize("sharpness", [HARD, 100.0])
def test_eager_vs_jit(z8, hard8, tmp_path, sharpness):
    """jit (separate process, --xla_cpu_max_isa=AVX so XLA emits no FMA; the
    oracle is built -ffp-contract=off) == eager to 1e-12; the default-ISA
    difference is printed (KNOWN DIFFERENCE, see the solver test)."""
    eager = hard8[0] if sharpness == HARD else run(z8, sharpness)[0]
    for flags, bound in (("--xla_cpu_max_isa=AVX", TOL), ("", None)):
        path = tmp_path / f"jit_{len(flags)}.npz"
        env = dict(os.environ, XLA_FLAGS=flags, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
        subprocess.run([sys.executable, "-c", _JIT_SCRIPT, str(path), repr(sharpness)],
                       env=env, check=True)
        jit = np.load(path)
        diff = {k: float(rel_err(eager[k], jit[k]).max()) for k in ecrad.OUTPUT_KEYS}
        print(f"EAGER_VS_JIT sharpness={sharpness} XLA_FLAGS='{flags}' max_rel={max(diff.values()):.3e} "
              f"worst={max(diff, key=diff.get)}")
        if bound is not None:
            assert all(v <= _V53_JIT_TOL.get(k, bound) for k, v in diff.items()), diff  # V53
