"""ecRad 49r1 McICA cloud generator vs the RUNNING 49r1 oracle (replayed stream).

Fixtures: ``oracle_fixtures/ecrad_ifs49.npz`` (``in:*`` ecRad input file,
``rng:*`` the oracle's uniform stream: one block per
``initialize_random_numbers`` call, identified by its seed) and
``oracle_fixtures/ecrad_ifs49_gen.npz`` (``gen:*`` the generator outputs dumped
after each ``cloud_generator`` call in radiation_mcica_lw/sw.F90).  Set
``ECRAD_IFS49_ALL`` and ``ECRAD_IFS49_GEN_ALL`` to the 32-column npz pair to
run the full-file match.

The replay adaptor maps each block (call order per generate_column_exp_ran:
rand_top(ng), then per g-point rand_cloud(iend+1-itrigger) followed by one
(rand_inhom1(n), rand_inhom2(n)) pair per fill) into the generator's static
layout, and asserts that every draw is placed exactly once.  The cloudy runs
that locate each fill are read off the oracle's own od_scaling > 0 (vote G11);
their lengths must equal the recorded call sizes.

Bar: max ``|a-b| / max(|a|,|b|)`` <= 1e-12 in fp64 at ``sharpness = inf``.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from legoesm.atmosphere.physics.radiation.ecrad_ifs import mcica_generator as mg  # noqa: E402

HERE = Path(__file__).parent / "oracle_fixtures"
TOL = 1.0e-12
# config49r1_rrtmg.nam / radiation_config.F90:214 (the run's values)
FRAC_THRESHOLD = 1.0e-6
DECORR_SCALING = 0.5
NG = {"lw": 140, "sw": 112}
PAIRS = [(HERE / "ecrad_ifs49.npz", HERE / "ecrad_ifs49_gen.npz")]
if os.environ.get("ECRAD_IFS49_ALL") and os.environ.get("ECRAD_IFS49_GEN_ALL"):
    PAIRS.append((Path(os.environ["ECRAD_IFS49_ALL"]), Path(os.environ["ECRAD_IFS49_GEN_ALL"])))
FILL = 0.5  # value of slots the oracle never drew


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    d = np.maximum(np.abs(a), np.abs(b))
    return np.where(d > 0, np.abs(a - b) / np.where(d > 0, d, 1.0), 0.0)


def _inputs(z):
    f = lambda k: np.asarray(z["in:" + k], np.float64)  # noqa: E731
    return f("cloud_fraction"), f("overlap_param"), f("fractional_std")


def _seeds(z, spec):
    iseed = np.asarray(z["in:iseed"])
    assert np.array_equal(iseed, np.round(iseed))
    return iseed.astype(np.int64) + (mg.LW_SEED_OFFSET if spec == "lw" else 0)


def _blocks(z):
    """[(seed, [call arrays])] in stream order; checks the stream partitions."""
    vals, seeds = z["rng:values"], z["rng:seeds"]
    block, sizes = z["rng:block_of_call"], z["rng:call_sizes"]
    assert sizes.sum() == vals.size and np.all(np.diff(block) >= 0)
    start = np.concatenate([[0], np.cumsum(sizes)])
    calls = [[] for _ in seeds]
    for c, b in enumerate(block):
        calls[b].append(vals[start[c]:start[c + 1]])
    return list(zip(seeds.tolist(), calls))


def replay(z, g, spec, fill=FILL):
    """Oracle stream -> (rand_top, rand_cloud, rand_inhom1, rand_inhom2) per column.

    Returns the four arrays (fill where never drawn), a per-column bool "has a
    block", and the number of stream values placed.
    """
    frac, _, _ = _inputs(z)
    ncol, nlev = frac.shape
    ng = NG[spec]
    od = g[f"gen:{spec}_od_scaling"]
    seeds = _seeds(z, spec)
    other = _seeds(z, "sw" if spec == "lw" else "lw")
    assert not np.intersect1d(seeds, other).size and np.unique(seeds).size == ncol
    blocks = {s: c for s, c in _blocks(z)}
    top = np.full((ncol, ng), fill)
    cloud, inh1, inh2 = (np.full((ncol, ng, nlev), fill) for _ in range(3))
    seen = [np.zeros(a.shape[1:], bool) for a in (top, cloud, inh1, inh2)]
    has = np.zeros(ncol, bool)
    placed = 0

    def put(arr, flag, idx, v):
        assert not flag[idx].any(), "slot filled twice"
        flag[idx] = True
        arr[idx] = v

    for c in range(ncol):
        if seeds[c] not in blocks:
            continue
        has[c] = True
        calls = list(blocks[seeds[c]])
        for f in seen:
            f[:] = False
        pos = np.flatnonzero(frac[c] > 0.0)
        iend = pos[-1]
        assert calls[0].size == ng
        put(top[c], seen[0], slice(None), calls.pop(0))
        placed += ng
        for jg in range(ng):
            rc = calls.pop(0)
            itrig = iend + 1 - rc.size
            assert pos[0] <= itrig <= iend
            put(cloud[c], seen[1], (jg, slice(itrig, iend + 1)), rc)
            placed += rc.size
            cl = od[c, jg] > 0.0
            assert not cl[:itrig].any() and not cl[iend + 1:].any() and cl[itrig]
            edges = np.flatnonzero(np.diff(np.concatenate([[0], cl[itrig:iend + 1].astype(int), [0]])))
            runs = [(itrig + a, b - a) for a, b in zip(edges[::2], edges[1::2])]
            if not cl[iend]:
                runs.append((iend + 1, 0))  # final fill of zero layers (jlev = iend+1)
            for a, n in runs:
                r1, r2 = calls.pop(0), calls.pop(0)
                assert r1.size == n and r2.size == n, "fill sizes disagree with oracle cloud runs"
                put(inh1[c], seen[2], (jg, slice(a, a + n)), r1)
                put(inh2[c], seen[3], (jg, slice(a, a + n)), r2)
                placed += 2 * n
        assert not calls, f"column {c}: {len(calls)} calls left over"
    return (top, cloud, inh1, inh2), has, placed


def _run(z, rand, sharpness=math.inf, jit=True):
    frac, op, fsd = _inputs(z)
    fn = lambda *a: mg.cloud_generator(*a, decorrelation_scaling=DECORR_SCALING,  # noqa: E731
                                       frac_threshold=FRAC_THRESHOLD, sharpness=sharpness)
    fn = jax.jit(fn) if jit else fn
    od, tcc = fn(frac, op, fsd, *rand)
    return np.asarray(od), np.asarray(tcc)


def test_stream_fully_consumed():
    """Every stream value lands in exactly one slot of exactly one (spectrum, column)."""
    for zp, gp in PAIRS:
        z, g = np.load(zp), np.load(gp)
        total = 0
        nblocks = 0
        for spec in ("lw", "sw"):
            _, has, placed = replay(z, g, spec)
            total += placed
            nblocks += has.sum()
            present = g[f"gen:{spec}_present"]
            tcc = g[f"gen:{spec}_total_cloud_cover"]
            # a block exists iff the generator ran and found cloud (cloud_generator:171-200)
            np.testing.assert_array_equal(has, present & (np.nan_to_num(tcc) >= FRAC_THRESHOLD))
        assert nblocks == z["rng:seeds"].size
        assert total == z["rng:values"].size, (total, z["rng:values"].size)


@pytest.mark.parametrize("spec", ["lw", "sw"])
@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p[0].name)
def test_matches_oracle(pair, spec):
    z, g = np.load(pair[0]), np.load(pair[1])
    rand, has, _ = replay(z, g, spec)
    od, tcc = _run(z, rand)
    present = g[f"gen:{spec}_present"]
    tcc_ref = g[f"gen:{spec}_total_cloud_cover"]
    rt = _rel(tcc[present], tcc_ref[present]).max()
    ro = _rel(od[has], g[f"gen:{spec}_od_scaling"][has]).max()
    print(f"{pair[0].name} {spec}: cols={present.sum()} cloudy={has.sum()} "
          f"max_rel tcc={rt:.3e} od_scaling={ro:.3e}")
    assert rt <= TOL and ro <= TOL
    # slots the oracle never drew are never read
    rand2, _, _ = replay(z, g, spec, fill=0.123)
    od2, tcc2 = _run(z, rand2)
    assert np.array_equal(od2[has], od[has]) and np.array_equal(tcc2, tcc)


def test_eager_matches_jit():
    z, g = np.load(PAIRS[0][0]), np.load(PAIRS[0][1])
    rand, has, _ = replay(z, g, "lw")
    cols = np.flatnonzero(has)[:2]
    frac, op, fsd = (a[cols] for a in _inputs(z))
    r = [a[cols] for a in rand]
    out = {}
    for name, jit in (("jit", True), ("eager", False)):
        fn = lambda *a: mg.cloud_generator(*a, decorrelation_scaling=DECORR_SCALING,  # noqa: E731
                                           frac_threshold=FRAC_THRESHOLD, sharpness=100.0)
        if jit:
            out[name] = jax.jit(fn)(frac, op, fsd, *r)
        else:
            with jax.disable_jit():
                out[name] = fn(frac, op, fsd, *r)
    d = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(out["jit"], out["eager"]))
    print(f"eager vs jit max abs diff (sharpness 100) = {d:.3e}")
    assert d <= 1e-12


def test_gamma_table_provenance():
    p = mg.gamma_provenance()
    x, fsd1, inv = mg.load_pdf_sampler()
    assert list(x.shape) == p["shapes"]["x"] == [81, 1000]
    assert fsd1 == 0.0 and inv == 1.0 / float(np.float32(0.05))


# --- differentiability -------------------------------------------------------
# Small hand-made column: layer 0 clear, layers 1-4 cloudy, one g-point.
F0 = np.array([0.0, 0.3, 0.5, 0.4, 0.6])
OP0 = np.array([0.5, 0.7, 0.6, 0.8])
FSD0 = np.full(5, 1.0)


def _small(frac, op, rand, sharpness):
    top, cloud, i1, i2 = rand
    od, tcc = mg.cloud_generator(frac[None], op[None], jnp.asarray(FSD0)[None], top[None],
                                 cloud[None], i1[None], i2[None],
                                 decorrelation_scaling=DECORR_SCALING,
                                 frac_threshold=FRAC_THRESHOLD, sharpness=sharpness)
    return jnp.sum(od) + tcc[0]


def _rand(top=0.5, cloud=None, i1=None, i2=None):
    nlev = F0.size
    c = np.full((1, nlev), 0.5) if cloud is None else np.asarray(cloud, float)[None]
    a = np.full((1, nlev), 0.37) if i1 is None else np.asarray(i1, float)[None]
    b = np.full((1, nlev), 0.9) if i2 is None else np.asarray(i2, float)[None]
    return (jnp.array([top]), jnp.asarray(c), jnp.asarray(a), jnp.asarray(b))


def _boundary_cases():
    cum, pair = (np.asarray(v) for v in mg.cum_cloud_cover_exp_ran(jnp.asarray(F0), jnp.asarray(OP0)))
    tcc = cum[-1]
    over = np.diff(cum)
    opi = OP0 ** (1.0 / DECORR_SCALING)
    cases = {}
    # cloud-top trigger exactly at cum(layer 2): rand_top*tcc = cum[2]
    cases["trigger"] = _rand(top=cum[2] / tcc)
    # cloud -> cloud transition into layer 2 (trigger at layer 1): r*f1 = f2+f1-pair1
    cc = np.full(5, 0.5)
    cc[1] = (F0[2] + F0[1] - pair[1]) / F0[1]
    cases["cloud_to_cloud"] = _rand(top=1e-3, cloud=cc)
    # clear -> cloud into layer 3: layer 2 clear (r>=..), then r*(cum2-f2) = pair2-over2-f2
    cl = np.full(5, 0.5)
    cl[1] = 0.999
    cl[2] = (pair[2] - over[2] - F0[2]) / (cum[2] - F0[2])
    cases["clear_to_cloud"] = _rand(top=1e-3, cloud=cl)
    # inhomogeneity repeat into layer 2: r2 = op_inhom(1), layers 1-2 cloudy
    i2 = np.full(5, 0.9)
    i2[2] = opi[1]
    cc2 = np.full(5, 0.01)
    cases["inhom_repeat"] = _rand(top=1e-3, cloud=cc2, i1=[0.1, 0.2, 0.8, 0.4, 0.5], i2=i2)
    return cases


# (rand array index, slot) of the uniform that feeds ONLY the named switch.
# The McICA sampling is exact at every sharpness (user rule 2026-11-04, vote
# T4): at each boundary case the uniform's gradient is exactly 0 and the
# output equals the hard form, while frac/op keep finite nonzero gradients.
SLOT = {"trigger": (0, (0,)), "cloud_to_cloud": (1, (0, 1)),
        "clear_to_cloud": (1, (0, 2)), "inhom_repeat": (3, (0, 2))}


@pytest.mark.parametrize("case", list(SLOT))
def test_sampling_exact_across_transition(case):
    rand = _boundary_cases()[case]
    f, op = jnp.asarray(F0), jnp.asarray(OP0)
    gf, gop = jax.grad(_small, argnums=(0, 1))(f, op, rand, 100.0)
    assert np.all(np.isfinite(np.concatenate([gf, gop]))) and np.abs(np.concatenate([gf, gop])).max() > 0
    which, slot = SLOT[case]
    g_r = jax.grad(_small, argnums=2)(f, op, rand, 100.0)[which][slot]
    v100, vinf = float(_small(f, op, rand, 100.0)), float(_small(f, op, rand, math.inf))
    print(f"{case}: d/d frac,op max={np.abs(np.concatenate([gf, gop])).max():.3e} "
          f"d/d uniform smooth={float(g_r):.3e} value 100={v100!r} inf={vinf!r}")
    assert float(g_r) == 0.0
    # the boundary case's draw decides the same way at 100 as at inf
    for d in (-1e-9, 1e-9):
        r = list(rand)
        r[which] = r[which].at[slot].add(d)
        assert ((float(_small(f, op, tuple(r), 100.0)) != v100)
                == (float(_small(f, op, tuple(r), math.inf)) != vinf)), (case, d)


def test_strict_compare_and_single_layer():
    assert float(mg._lt(jnp.float64(1.0), jnp.float64(1.0))) == 0.0   # oracle "<"
    od, tcc = mg.cloud_generator(jnp.array([[0.5]]), jnp.zeros((1, 0)), jnp.ones((1, 1)),
                                 jnp.array([[0.2]]), jnp.full((1, 1, 1), 0.5),
                                 jnp.full((1, 1, 1), 0.37), jnp.full((1, 1, 1), 0.9),
                                 decorrelation_scaling=DECORR_SCALING,
                                 frac_threshold=FRAC_THRESHOLD, sharpness=math.inf)
    assert float(tcc[0]) == 0.5
    assert float(od[0, 0, 0]) == float(mg.sample_from_pdf(jnp.float64(1.0), jnp.float64(0.37),
                                                            sharpness=math.inf))


def test_gradient_tcc_gate():
    def h(f):
        cf = jnp.zeros(5).at[2].set(f)
        top, cloud, i1, i2 = _rand()
        _, tcc = mg.cloud_generator(cf[None], jnp.asarray(OP0)[None], jnp.asarray(FSD0)[None],
                                    top[None], cloud[None], i1[None], i2[None],
                                    decorrelation_scaling=DECORR_SCALING,
                                    frac_threshold=FRAC_THRESHOLD, sharpness=100.0)
        return tcc[0]
    d = jax.grad(h)(FRAC_THRESHOLD)
    assert np.isfinite(d) and d > 1.0  # gate slope adds to d tcc/d f = 1


def test_check_grads_interior():
    rand = _rand(top=0.4, cloud=[0.3, 0.6, 0.2, 0.7, 0.5], i1=[0.1, 0.2, 0.8, 0.4, 0.5],
                 i2=[0.3, 0.9, 0.2, 0.6, 0.4])
    fn = lambda f, op: _small(f, op, rand, 100.0)  # noqa: E731
    check_grads(fn, (jnp.asarray(F0) + 0.01, jnp.asarray(OP0)), order=1,
                modes=("fwd", "rev"), eps=1e-6)


def test_draw_uniforms_layout():
    top, cloud, i1, i2 = mg.draw_uniforms(254477991 + mg.LW_SEED_OFFSET, 140, 137)
    assert top.shape == (140,) and cloud.shape == i1.shape == i2.shape == (140, 137)
    assert top.dtype == jnp.float64
    for a in (top, cloud, i1, i2):
        a = np.asarray(a)
        assert a.min() >= 0.0 and a.max() < 1.0
    again = mg.draw_uniforms(254477991 + mg.LW_SEED_OFFSET, 140, 137)
    assert all(np.array_equal(a, b) for a, b in zip((top, cloud, i1, i2), again))


@pytest.mark.parametrize("f_thin", [1e-6, 1e-4, 3e-3])
def test_smooth_cover_bounds(f_thin):
    """Smooth pair-cover max keeps thin cloud (G12''): a lone layer's cover is its
    fraction; two overcast layers never exceed cover 1."""
    frac = jnp.array([0.0, f_thin, 0.0, 0.0])
    cum, _ = mg.cum_cloud_cover_exp_ran(frac, jnp.ones(3), 100.0)
    cum_h, _ = mg.cum_cloud_cover_exp_ran(frac, jnp.ones(3), math.inf)
    # equal to the hard form (whose 1 - cum_product carries its own rounding)
    assert float(cum[-1]) == float(cum_h[-1]) and abs(float(cum[-1]) - f_thin) <= 1e-9 * f_thin
    cum1, _ = mg.cum_cloud_cover_exp_ran(jnp.array([1.0, 1.0, 0.2]), jnp.array([0.7, 0.4]), 100.0)
    assert float(jnp.max(cum1)) <= 1.0


def test_smooth_cover_near_overcast_and_clear():
    """Review round 2: no cover collapse next to an overcast layer, no jump at
    MaxCloudFrac, finite gradients through adjacent clear layers."""
    f0 = np.nextafter(mg._max_cloud_frac(np.float64), 0.0)
    covers = [float(mg.cum_cloud_cover_exp_ran(jnp.array([f, 0.999]), jnp.ones(1), 100.0)[0][-1])
              for f in (f0, mg._max_cloud_frac(np.float64))]
    assert covers[0] >= f0 and abs(covers[0] - covers[1]) <= 1e-13
    g = jax.grad(lambda f: mg.cum_cloud_cover_exp_ran(f, jnp.full(2, 0.5), 100.0)[0][-1])(
        jnp.array([0.0, 0.0, 0.5]))
    assert np.all(np.isfinite(np.asarray(g)))
