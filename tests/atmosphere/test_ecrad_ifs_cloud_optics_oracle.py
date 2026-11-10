"""ecRad 49r1 cloud optics vs the RUNNING 49r1 oracle (saved radiative properties).

Fixture ``oracle_fixtures/ecrad_ifs49.npz`` (vote V38): ``in:*`` is the ecRad
input file (float32, read by the driver into fp64 unchanged: no scaling, no
``do_correct_unphysical_inputs``), ``props:*`` the arrays
``radiation_save.F90:save_radiative_properties`` wrote right after
``cloud_optics`` (in-cloud, delta-scaled, cropped ``cloud_fraction``).
Set ``ECRAD_IFS49_ALL`` to the 32-column npz to run the full-file match.

Bar: max ``|a-b| / max(|a|,|b|)`` <= 1e-12 in fp64 at ``sharpness = inf``.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from legoesm.atmosphere.physics.radiation.ecrad_ifs import cloud_optics as co  # noqa: E402

FIXTURE = Path(__file__).parent / "oracle_fixtures" / "ecrad_ifs49.npz"
TOL = 1.0e-12
PAIRS = {"od_lw": "od_lw_cloud", "ssa_lw": "ssa_lw_cloud", "g_lw": "asymmetry_lw_cloud",
         "od_sw": "od_sw_cloud", "ssa_sw": "ssa_sw_cloud", "g_sw": "asymmetry_sw_cloud",
         "cloud_fraction": "cloud_fraction"}
FILES = [FIXTURE] + ([Path(os.environ["ECRAD_IFS49_ALL"])] if os.environ.get("ECRAD_IFS49_ALL") else [])


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    d = np.maximum(np.abs(a), np.abs(b))
    return np.where(d > 0, np.abs(a - b) / np.where(d > 0, d, 1.0), 0.0)


def _inputs(z):
    f64 = lambda k: np.asarray(z["in:" + k], dtype=np.float64)  # noqa: E731
    return dict(pressure_hl=f64("pressure_hl"), cloud_fraction=f64("cloud_fraction"),
                q_liq=f64("q_liquid"), q_ice=f64("q_ice"),
                re_liq=f64("re_liquid"), re_ice=f64("re_ice"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_matches_oracle(path):
    z = np.load(path)
    inp = _inputs(z)
    np.testing.assert_array_equal(inp["pressure_hl"], z["props:pressure_hl"])
    out = co.cloud_optics(**inp, sharpness=math.inf)
    worst = {}
    for name, key in PAIRS.items():
        ref = z["props:" + key]
        got = np.asarray(getattr(out, name))
        assert got.shape == ref.shape, (name, got.shape, ref.shape)
        worst[name] = float(_rel(got, ref).max())
    print(f"\n{path.name} max rel:", {k: f"{v:.3e}" for k, v in worst.items()})
    assert all(v <= TOL for v in worst.values()), worst
    # non-vacuous: cloudy cells and both phases present
    assert (z["props:od_sw_cloud"] > 0).any() and (np.asarray(inp["q_ice"]) > 0).any()


def test_crop_hard_form_is_oracle_rule():
    f = np.array([0.0, 5e-7, 1e-6, 1e-6, 2e-6, 0.3, 0.3, 1.0])
    ql = np.array([1e-5, 1e-5, 1e-9, 4e-10, 0.0, 0.0, 9e-10, 1e-4])
    qi = np.array([0.0, 0.0, 0.0, 5e-10, 0.0, 1e-9, 0.0, 0.0])
    fc, _ = co.crop_cloud_fraction(f, ql, qi, sharpness=math.inf)
    ref = np.where((f < 1e-6) | ((0.0 + ql) + qi < 1e-9), 0.0, f)
    np.testing.assert_array_equal(np.asarray(fc), ref)


def test_tables_match_provenance():
    h = hashlib.sha256(Path(co.TABLE_FILE).read_bytes()).hexdigest()
    prov = co.provenance()
    assert h == prov["npz_sha256"]
    c = co.load_coefficients()
    assert {k: list(v.shape) for k, v in c.items()} == prov["shapes"]


def _column(nlev=6):
    p = np.linspace(2.0e4, 9.0e4, nlev + 1)
    return dict(pressure_hl=p, cloud_fraction=np.full(nlev, 0.4),
                q_liq=np.full(nlev, 2e-5), q_ice=np.full(nlev, 1e-5),
                re_liq=np.full(nlev, 1e-5), re_ice=np.full(nlev, 4e-5))


def _loss(name, x, base, sharpness=100.0):
    inp = dict(base)
    inp[name] = jnp.full_like(jnp.asarray(base[name]), x)
    out = co.cloud_optics(**inp, sharpness=sharpness)
    return sum(jnp.sum(v) for v in out)


# (input, points spanning the smoothed transition)
TRANSITIONS = [
    ("cloud_fraction", 1e-6 * np.array([0.97, 0.99, 1.0, 1.01, 1.03])),   # crop on fraction
    ("q_liq", 1e-9 * np.array([0.97, 0.99, 1.0, 1.01, 1.03])),            # crop on q sum (q_ice=0 below)
    ("q_liq", 1e-11 * np.array([0.1, 0.5, 0.9, 1.5])),                    # liquid presence C1 ramp
    ("re_liq", 1.2e-6 * np.array([0.97, 0.99, 1.0, 1.01, 1.03])),         # SOCRATES lower clamp
    ("re_liq", 50e-6 * np.array([0.97, 0.99, 1.0, 1.01, 1.03])),          # SOCRATES upper clamp
    ("re_ice", 100e-6 * np.array([0.97, 0.99, 1.0, 1.01, 1.03])),         # Fu radius cap
]


@pytest.mark.parametrize("name,points", TRANSITIONS, ids=lambda v: v if isinstance(v, str) else "")
def test_gradient_across_transition(name, points):
    base = _column()
    if name == "q_liq" and points[0] > 1e-10:   # crop on the q sum: liquid only
        base["q_ice"] = np.zeros_like(base["q_ice"])
    grad = jax.grad(lambda x: _loss(name, x, base))
    for x in points:
        gv = float(grad(float(x)))
        assert np.isfinite(gv) and gv != 0.0, (name, x, gv)


def test_gradient_across_g_cap():
    g = jnp.asarray(co._fu_max_g(np.float64)) * jnp.array([0.997, 0.999, 1.0, 1.001, 1.003])
    dg = jax.vmap(jax.grad(lambda v: co._g_cap(v, 100.0)))(g)
    assert np.all(np.isfinite(dg)) and np.all(np.asarray(dg) != 0.0)


def test_check_grads_interior():
    base = _column()
    keys = ("cloud_fraction", "q_liq", "q_ice", "re_liq", "re_ice")

    def f(x):
        inp = dict(base)
        for i, k in enumerate(keys):
            inp[k] = jnp.asarray(base[k]) * (1.0 + x[i])
        out = co.cloud_optics(**inp, sharpness=100.0)
        return jnp.stack([jnp.sum(v) for v in out])

    check_grads(f, (jnp.zeros(len(keys)),), order=1, modes=["fwd", "rev"], eps=1e-4)


@pytest.mark.parametrize("sharpness", [math.inf, 100.0])
def test_eager_vs_jit(sharpness):
    z = np.load(FIXTURE)
    inp = _inputs(z)
    eager = co.cloud_optics(**inp, sharpness=sharpness)
    jitted = jax.jit(lambda *a: co.cloud_optics(*a, sharpness=sharpness))(*inp.values())
    diff = max(float(_rel(a, b).max()) for a, b in zip(eager, jitted))
    print(f"\neager vs jit sharpness={sharpness}: max rel diff {diff:.3e}")
    assert diff <= TOL
    if sharpness == math.inf:   # jit also meets the oracle bar
        worst = max(float(_rel(getattr(jitted, n), z["props:" + k]).max()) for n, k in PAIRS.items())
        print(f"jit vs oracle max rel {worst:.3e}")
        assert worst <= TOL
    else:                       # production smoothing gap, measured (no bar)
        cloudy = z["props:cloud_fraction"] > 0
        for n, k in PAIRS.items():
            a, b = np.asarray(getattr(eager, n)), z["props:" + k]
            m = cloudy if a.ndim == 2 else cloudy[..., None] & np.ones(a.shape, bool)
            print(f"smooth(100) vs oracle {n}: cloudy max rel {_rel(a, b)[m].max():.2e}, "
                  f"clear max abs {np.abs(a - b)[~m].max() if (~m).any() else 0.0:.2e}")


def test_hard_ssa_unguarded_like_oracle():
    """Fu LW od can be negative at small re (Fortran divides unguarded, :474-475)."""
    base = _column()
    base["q_liq"] = np.zeros_like(base["q_liq"])
    base["re_ice"] = np.full_like(base["re_ice"], 4e-6)
    out = co.cloud_optics(**base, sharpness=math.inf)
    neg = np.asarray(out.od_lw) < 0
    assert neg.any()
    assert np.all(np.asarray(out.ssa_lw)[neg] != 0.0)


def test_hard_clear_cell_skipped_like_oracle():
    """Cropped cells are skipped (Fortran :298), so singular re there gives zeros."""
    base = dict(_column(), cloud_fraction=np.zeros(6), re_ice=np.zeros(6), re_liq=np.zeros(6))
    out = co.cloud_optics(**base, sharpness=math.inf)
    for v in out:
        np.testing.assert_array_equal(np.asarray(v), 0.0)


def test_gradient_finite_for_tiny_water():
    """CO8: q ~ 1e-90 must not give NaN reverse gradients in the smooth form."""
    base = _column()
    for name in ("q_liq", "re_liq", "q_ice"):
        b = dict(base, q_liq=np.full(6, 1e-90), q_ice=np.zeros(6))
        gv = float(jax.grad(lambda x: _loss(name, x, b))(float(b[name][0])))
        assert np.isfinite(gv), (name, gv)
    b = dict(base, cloud_fraction=np.full(6, 1e-155))      # CO8 floor on the fraction divide
    gv = float(jax.grad(lambda x: _loss("cloud_fraction", x, b))(1e-155))
    assert np.isfinite(gv), gv
