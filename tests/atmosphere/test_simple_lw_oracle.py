"""Stevens (2005) simple longwave vs the RUNNING gSAM 1.8.8 rad_simple oracle.

Fixture ``tests/atmosphere/oracle_fixtures/simple_lw_gsam.npz`` was generated
by RUNNING gSAM 1.8.8 ``SRC/rad_simple.f90`` (release tarball
http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz, sha256 in the npz
``meta``), compiled unmodified (gfortran 8.5.0, fp64), with ``python
scripts/validate/oracle/simple_lw/make_fixture.py --tgz <tarball> --build
<dir> --out simple_lw_gsam.npz``.

gSAM's form is selected by ``cp_j_kg_k=DYCOMS_SPEC_CP_J_KG_K``,
``density_at_inversion=False`` and gSAM's face density ``rhow`` passed as
``rho_half``; the hard limit of the smooth condensate guard (RULE 5) is
``condensate_smooth_width_kg_kg = 1e-30``.  At the default width the recorded
per-column gap is pinned in ``test_default_width_gap_per_column``.

Tolerance (user decision 2026-10-19): max relative error <= 1e-10 in fp64.
Relative error is ``|a-b| / max(|a|, |b|)`` (0 where both are exactly 0); no
floor -- the metric of the merged tier-1 oracle tests. Compared quantities are
the oracle's own outputs: the net upward flux at faces 1..nzm (``radlwdn``)
and the heating rate (``qrad``).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.radiation.simple_lw import (  # noqa: E402
    DYCOMS_SPEC_CP_J_KG_K,
    SimpleLWConfig,
    simple_lw_net_upward_flux,
    simple_lw_temperature_tendency,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/atmosphere/oracle_fixtures/simple_lw_gsam.npz"
GENERATOR = ROOT / "scripts/validate/oracle/simple_lw/make_fixture.py"
TOL = 1.0e-10
_HARD_W = 1.0e-30   # condensate guard width at the hard (gSAM) limit
GSAM = SimpleLWConfig(cp_j_kg_k=DYCOMS_SPEC_CP_J_KG_K, density_at_inversion=False,
                      condensate_smooth_width_kg_kg=_HARD_W)
GSAM_DEFAULT_W = GSAM._replace(
    condensate_smooth_width_kg_kg=SimpleLWConfig().condensate_smooth_width_kg_kg)
GSAM_TARBALL_SHA256 = "0e9ea4a37f4daca5051cc4018cc95bbc5583159c7c27d7d05d1104fae4ed270c"


def _fx():
    return dict(np.load(FIXTURE))


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    d = np.maximum(np.abs(a), np.abs(b))
    return np.where(d > 0, np.abs(a - b) / np.where(d > 0, d, 1.0), 0.0)


def _args(fx, name, dtype=jnp.float64):
    g = {k: fx[f"{name}__{k}"] for k in ("zi", "adz", "zc", "rho", "rhow", "qv", "qcl", "qci")}
    cond = g["qcl"] + g["qci"]
    qt = g["qv"] + g["qcl"] + g["qci"]
    dzk = g["adz"] * fx[f"{name}__dz"]
    a = [jnp.asarray(x, dtype=dtype) for x in (cond, qt, g["rho"], dzk, g["zi"])]
    kw = dict(z_full=jnp.asarray(g["zc"], dtype=dtype), rho_half=jnp.asarray(g["rhow"], dtype=dtype))
    return a, kw


def test_fixture_provenance_and_inputs_reproduce():
    """The committed generator rebuilds the fixture's inputs bit-for-bit."""
    fx = _fx()
    meta = json.loads(str(fx["meta"]))
    assert meta["oracle"] == "gSAM" and meta["tag"] == "1.8.8"
    assert meta["file"] == "SRC/rad_simple.f90"
    assert meta["tarball_sha256"] == GSAM_TARBALL_SHA256
    spec = importlib.util.spec_from_file_location("simple_lw_make_fixture", GENERATOR)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    cols = gen.build_columns()
    assert list(cols) == [str(n) for n in fx["col_names"]]
    for name, c in cols.items():
        for k in gen.COL_FIELDS:
            np.testing.assert_array_equal(np.asarray(c[k]), fx[f"{name}__{k}"])


def test_matches_gsam_rad_simple():
    fx = _fx()
    worst = {}
    for name in fx["col_names"]:
        a, kw = _args(fx, str(name))
        F = np.asarray(simple_lw_net_upward_flux(*a, GSAM, **kw))
        H = np.asarray(simple_lw_temperature_tendency(*a, GSAM, **kw))
        worst[f"{name}:flux"] = float(_rel(F[:-1], fx[f"{name}__ora_flux"]).max())
        worst[f"{name}:heating"] = float(_rel(H, fx[f"{name}__ora_hr"]).max())
    print("max rel err", worst)
    bad = {k: v for k, v in worst.items() if v > TOL}
    assert not bad, bad


def test_edge_columns_exercise_the_guards():
    """Non-vacuity of the fixture: the q_t == 8 g/kg cell and the negative
    condensate cell sit where the guards change the answer."""
    fx = _fx()
    qv = fx["edge_eq_threshold_only__qv"]
    qcl = fx["edge_eq_threshold_only__qcl"]
    assert np.any(qv + qcl == 0.008)
    assert np.any(fx["edge_neg_cond_only__qcl"] < 0.0)
    # the oracle's flux on the threshold column equals the unperturbed one
    np.testing.assert_array_equal(fx["edge_eq_threshold_only__ora_flux"],
                                  fx["rf01_5m__ora_flux"])


def test_gsam_form_refuses_to_run_without_face_density():
    fx = _fx()
    a, kw = _args(fx, "rf01_5m")
    kw.pop("rho_half")
    with pytest.raises(ValueError, match="rho_half"):
        simple_lw_net_upward_flux(*a, GSAM, **kw)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("cfg", [GSAM, SimpleLWConfig()], ids=["gsam", "default"])
def test_gradient_finite_on_fixture_columns(dtype, cfg):
    fx = _fx()
    for name in fx["col_names"]:
        (cond, qt, rho, dzk, zi), kw = _args(fx, str(name), dtype)
        if cfg.density_at_inversion:
            kw.pop("rho_half")

        def loss(cond, rho):
            return jnp.sum(simple_lw_temperature_tendency(cond, qt, rho, dzk, zi, cfg, **kw) ** 2)

        g_c, g_r = jax.grad(loss, argnums=(0, 1))(cond, rho)
        assert np.all(np.isfinite(np.asarray(g_c))), name
        assert np.all(np.isfinite(np.asarray(g_r))), name


def test_eager_jit_parity():
    fx = _fx()
    a, kw = _args(fx, "rf01_stretched")
    f = lambda *x: simple_lw_temperature_tendency(*x, GSAM, **kw)  # noqa: E731
    assert _rel(f(*a), jax.jit(f)(*a)).max() <= TOL


# Per-column gap of the DEFAULT condensate-guard width (1e-6 kg/kg, smooth
# q*sigmoid(q/w)) vs the gSAM oracle, gSAM's other knobs (cp, face density),
# metric = per-column max of _rel.  Measured 2026-10-29 (#2222); a change to
# the smooth form or its default width re-measures these (CLAUDE.md RULE 5).
_DEFAULT_W_GAP_FLUX = {'rf01_5m': 1.3889235762494833e-06, 'astex209_25m': 5.933915958550329e-09, 'rf01_stretched': 5.5063926923207045e-05, 'rf01_ice_split': 1.3889235762494833e-06, 'rf01_clear': 1.2751576863915922e-16, 'dry_no_inversion_cloudy': 1.1073693100701172e-06, 'edge_eq_threshold_neg_cond': 0.00013642562549729529, 'edge_eq_threshold_only': 1.3889235762494833e-06, 'edge_neg_cond_only': 0.00013642562549729529}
_DEFAULT_W_GAP_HEAT = {'rf01_5m': 0.00036394819399193027, 'astex209_25m': 1.4460475907403964e-07, 'rf01_stretched': 0.011425048613804736, 'rf01_ice_split': 0.00036394819399193027, 'rf01_clear': 1.2373080453361e-13, 'dry_no_inversion_cloudy': 0.0012653413343336048, 'edge_eq_threshold_neg_cond': 1.0, 'edge_eq_threshold_only': 0.00036394819399193027, 'edge_neg_cond_only': 1.0}


def test_default_width_gap_per_column():
    """RULE 5: the default smoothing departs from gSAM by the recorded gap."""
    fx = _fx()
    for name in fx["col_names"]:
        name = str(name)
        a, kw = _args(fx, name)
        F = np.asarray(simple_lw_net_upward_flux(*a, GSAM_DEFAULT_W, **kw))
        H = np.asarray(simple_lw_temperature_tendency(*a, GSAM_DEFAULT_W, **kw))
        np.testing.assert_allclose(_rel(F[:-1], fx[f"{name}__ora_flux"]).max(),
                                   _DEFAULT_W_GAP_FLUX[name], rtol=1e-6, atol=1e-12,
                                   err_msg=name)
        np.testing.assert_allclose(_rel(H, fx[f"{name}__ora_hr"]).max(),
                                   _DEFAULT_W_GAP_HEAT[name], rtol=1e-6, atol=1e-12,
                                   err_msg=name)


def test_check_grads_through_the_condensate_guard_at_default():
    """RULE 5 gradient test: finite, non-zero derivative through the guard.

    The negative-condensate cell is shifted by s * w (w the default width),
    so s = 0 sits on the switch's own scale; the hard guard has zero
    derivative there."""
    fx = _fx()
    (cond, qt, rho, dzk, zi), kw = _args(fx, "edge_neg_cond_only")
    k = int(np.argmin(np.asarray(cond)))
    assert float(cond[k]) < 0.0
    w = GSAM_DEFAULT_W.condensate_smooth_width_kg_kg
    e = jnp.zeros_like(cond).at[k].set(w)

    def f(s, cfg=GSAM_DEFAULT_W):
        return simple_lw_net_upward_flux(cond + s * e, qt, rho, dzk, zi, cfg, **kw)

    s0 = jnp.asarray(0.0)
    check_grads(f, (s0,), order=2, modes=("fwd", "rev"))
    assert float(jnp.max(jnp.abs(jax.jacfwd(f)(s0)))) > 0.0
    hard = jax.jacfwd(lambda s: f(s, GSAM))(s0)
    assert float(jnp.max(jnp.abs(hard))) == 0.0
