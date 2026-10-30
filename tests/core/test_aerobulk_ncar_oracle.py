"""Oracle fixture: ``compute_most_fluxes(scheme="large_yeager")`` reproduces the
COMPILED AeroBulk 1.0.0 ``TURB_NCAR`` (Large & Yeager 2004, 2009).

Oracle: brodeau/aerobulk tag 1.0.0 (SHA ca40928872eb9c00f96a51cf3a6eed0bb4a70580),
mod_blk_ncar / mod_phymbl compiled unmodified, mod_const
with legoESM's grav and R_vap (still default-real literals, as AeroBulk writes
every constant), no cool-skin / warm-layer, nb_iter = 5.  The fixture
``oracle_fixtures/aerobulk_ncar.npz`` holds the oracle's inputs and outputs
for 1549 states (ERA5 open ocean incl. seven named regimes, the frozen
aerobulk-python baseline, ten synthetic edges) at zt/zu = 10/10 and 2/10 m; its
``meta`` string records the provenance.  Regenerate with::

    bash scripts/validate/oracle/aerobulk/build.sh <aerobulk-1.0.0> <build>
    python scripts/validate/oracle/aerobulk/make_fixture.py --drv <build>/drv \\
        --scheme ncar --ic <ic_2001.nc> --fc <forcing_2001.nc> \\
        --out tests/core/oracle_fixtures/aerobulk_ncar.npz

The oracle fluxes are AeroBulk's BULK_FORMULA structure (tau = rho Cd Ub |U|,
Q_H = rho cp Ch Ub (t_zu - T_s), E = rho Ce Ub (q_zu - q_s)) built from its
Cd, Ch, Ce, Ub, t_zu, q_zu with the model's rho, cp = c_pd and L; AeroBulk heat
fluxes are positive DOWN and its stress lies along the wind, so the model's are
compared with a sign flip.  Pass: max relative
error <= 1e-10 (fp64), exact agreement where the oracle value is exactly zero.

RULE 5 (#2241; form and width decided by codex and GLM separately): the
in-loop ``zeta_u >= 0`` switch of the neutral Stanton number (18 vs 32.7) is a
compact quintic blend of half-width ``ncar_neutral_width`` (default 1e-3).
The oracle match above is the hard limit (width 0).  At the default the signed
per-state gap (ours - oracle) of every output is pinned against
``oracle_fixtures/aerobulk_ncar_default_gap.npz`` (1e-6 relative, 1e-12
absolute).  Regenerate it after a change to the smooth form or its width with
``python tests/core/test_aerobulk_ncar_oracle.py``.  Still hard (pending user
decisions, #2241): the first-guess ``dthv >= 0`` switch, the 33 m/s drag
cutoff and the dt/dq sign floors.
"""
from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.test_util import check_grads  # noqa: E402
from legoesm.core.bulk_flux import (  # noqa: E402
    NCAR_NEUTRAL_WIDTH,
    _ncar_stable_weight,
    compute_most_fluxes,
)

from legoesm import constants  # noqa: E402

_FIXTURE = Path(__file__).parent / "oracle_fixtures" / "aerobulk_ncar.npz"
_GAP_FILE = Path(__file__).parent / "oracle_fixtures" / "aerobulk_ncar_default_gap.npz"
_RTOL = 1e-10
_HEIGHTS = {"z10": 10.0, "z2": 2.0}


def _fixture():
    return np.load(_FIXTURE)


def _inputs(d, h, dtype=jnp.float64):
    return {k: jnp.asarray(d[f"{h}_in_{k}"], dtype=dtype)
            for k in ("u", "v", "th_a", "q", "th_s", "qs", "rho", "L")}


def _ours(x, zt, **kw):
    tx, ty, sh, lh, us = compute_most_fluxes(
        x["u"], x["v"], x["th_a"], x["q"], x["th_s"], x["qs"], x["rho"],
        z_ref=10.0, z_t=zt, z_q=zt, scheme="large_yeager", n_iter=5,
        L_latent=x["L"], **kw)
    return {"taux": tx, "tauy": ty, "sen": sh, "lat": lh, "ustar": us}


def _oracle(d, h):
    g = lambda k: d[f"{h}_{k}"]  # noqa: E731
    rho, cd, ub = g("in_rho"), g("oracle_Cd"), g("oracle_Ub")
    return {"taux": -rho * cd * ub * g("in_u"), "tauy": -rho * cd * ub * g("in_v"),
            "sen": -rho * constants.c_pd * g("oracle_Ch") * ub * (g("oracle_t_zu") - g("in_th_s")),
            "lat": -rho * g("in_L") * g("oracle_Ce") * ub * (g("oracle_q_zu") - g("in_qs")),
            "ustar": g("oracle_us")}


def max_rel_errors(d, h, **kw):
    ours, ref = _ours(_inputs(d, h), _HEIGHTS[h], **kw), _oracle(d, h)
    out = {}
    for k, b in ref.items():
        a = np.asarray(ours[k])
        zero = b == 0.0
        assert np.all(a[zero] == 0.0), (h, k)
        out[k] = float(np.max(np.abs(a[~zero] - b[~zero]) / np.abs(b[~zero])))
    return out


def test_fixture_provenance_and_coverage():
    d = _fixture()
    meta = str(d["meta"])
    assert "ca40928872eb9c00f96a51cf3a6eed0bb4a70580" in meta and "TURB_NCAR" in meta
    for h in _HEIGHTS:
        dt = d[f"{h}_in_th_a"] - d[f"{h}_in_th_s"]
        assert (dt > 0).sum() >= 50 and (dt < 0).sum() >= 50  # both stabilities
        assert np.any(d[f"{h}_in_W"] == 0.0) and np.any(d[f"{h}_in_W"] > 33.0)


@pytest.mark.parametrize("h", sorted(_HEIGHTS))
def test_large_yeager_matches_compiled_aerobulk_turb_ncar(h):
    worst = max_rel_errors(_fixture(), h, ncar_neutral_width=0.0)
    bad = {k: v for k, v in worst.items() if not v <= _RTOL}
    assert not bad, (h, worst)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_large_yeager_gradient_finite_on_fixture(dtype):
    d = _fixture()
    for h, zt in _HEIGHTS.items():
        x = _inputs(d, h, dtype)

        def loss(u, v, th_a, q, th_s, qs):
            y = x | dict(u=u, v=v, th_a=th_a, q=q, th_s=th_s, qs=qs)
            r = _ours(y, zt)
            return jnp.sum(r["taux"] + r["tauy"] + r["sen"] + r["lat"] + r["ustar"])

        args = [x[k] for k in ("u", "v", "th_a", "q", "th_s", "qs")]
        for g in jax.grad(loss, argnums=tuple(range(6)))(*args):
            assert np.all(np.isfinite(np.asarray(g))), (h, dtype)


def test_large_yeager_jit_matches_eager():
    d = _fixture()
    x = _inputs(d, "z2")
    eager = _ours(x, 2.0)
    jitted = jax.jit(lambda y: _ours(y, 2.0))(x)
    for k in eager:
        np.testing.assert_allclose(np.asarray(jitted[k]), np.asarray(eager[k]),
                                   rtol=_RTOL, atol=0.0, err_msg=k)


@pytest.mark.parametrize("h", sorted(_HEIGHTS))
def test_large_yeager_2m_diagnostic_closes_at_measurement_height(h):
    """return_2m uses the ported solver's profile (AeroBulk TURB_NCAR's height
    shift from z_t): at z_diag = z_t it returns the air temperature."""
    d = _fixture()
    x, zt = _inputs(d, h), _HEIGHTS[h]
    out = compute_most_fluxes(x["u"], x["v"], x["th_a"], x["q"], x["th_s"], x["qs"], x["rho"],
                              z_ref=10.0, z_t=zt, z_q=zt, scheme="large_yeager", n_iter=5,
                              L_latent=x["L"], return_2m=True,
                              z_diag=zt)
    t, ta = np.asarray(out[5]), np.asarray(x["th_a"])
    assert np.max(np.abs(t - ta) / ta) <= _RTOL


def test_large_yeager_gradient_finite_calm_stable_column():
    """Exact calm wind over a colder sea - the bulk wind sits on AeroBulk's
    0.5 m/s floor; the gradient must stay finite."""
    def loss(u, v, ts):
        r = compute_most_fluxes(u, v, jnp.array([285.0]), jnp.array([6e-3]), ts,
                                jnp.array([7e-3]), jnp.array([1.2]), scheme="large_yeager")
        return jnp.sum(r[0] + r[2] + r[3] + r[4])
    for g in jax.grad(loss, argnums=(0, 1, 2))(jnp.array([0.0]), jnp.array([0.0]),
                                                jnp.array([280.0])):
        assert np.all(np.isfinite(np.asarray(g)))


def _default_gaps(d, **kw):
    """Signed per-state gap (ours at the public defaults - oracle)."""
    out = {}
    for h, zt in _HEIGHTS.items():
        ours, ref = _ours(_inputs(d, h), zt, **kw), _oracle(d, h)
        for k, b in ref.items():
            out[f"{h}_{k}"] = np.asarray(ours[k]) - b
    return out


def test_default_width_is_the_recorded_one():
    import inspect
    assert NCAR_NEUTRAL_WIDTH == 1.0e-3
    sig = inspect.signature(compute_most_fluxes).parameters
    assert sig["ncar_neutral_width"].default == NCAR_NEUTRAL_WIDTH


def test_default_gap_per_state():
    """RULE 5 KNOWN DIFFERENCE: at the default neutral width every state's
    signed gap to AeroBulk equals the stored one."""
    stored = np.load(_GAP_FILE)
    gaps = _default_gaps(_fixture())
    assert sorted(gaps) == sorted(k for k in stored.files if k != "width")
    for k, g in gaps.items():
        np.testing.assert_allclose(g, stored[k], rtol=1e-6, atol=1e-12, err_msg=k)
    assert max(np.abs(stored[k]).max() for k in gaps) > 1e-6


def test_check_grads_ncar_stable_weight_at_default():
    """zeta = 0, where the hard step jumps (fails at width 0); the default
    quintic blend is smooth there with a non-zero slope.  eps 1e-7 (N6)."""
    def f(z):
        return _ncar_stable_weight(z, NCAR_NEUTRAL_WIDTH)
    z0 = jnp.asarray(0.0)
    check_grads(f, (z0,), order=2, modes=("fwd", "rev"), eps=1e-7)
    assert float(jax.grad(f)(z0)) != 0.0



def test_check_grads_ncar_neutral_scheme_level_at_default():
    """Scheme-level: u = 6 m/s, air 290 K / 10 g/kg over a 12 g/kg sea, T_s =
    289.331297 K, where the hard loop's zeta_u sign switch makes the fluxes
    jump (found 2026-10-30 by a value-jump scan of the hard form over T_s in
    [289, 290.3]; the first-guess dthv = 0 switch, still hard, sits at 289.63 K).
    check_grads fails there at width 0."""
    def f(ts):
        r = compute_most_fluxes(jnp.atleast_1d(6.0), jnp.atleast_1d(0.0), jnp.atleast_1d(290.0),
                                jnp.atleast_1d(0.010), jnp.atleast_1d(ts), jnp.atleast_1d(0.012),
                                jnp.atleast_1d(1.2), z_ref=10.0, scheme="large_yeager")
        return jnp.sum(r[0] + r[2] + r[3] + r[4])
    t0 = jnp.asarray(289.3312971145683)
    check_grads(f, (t0,), order=2, modes=("fwd", "rev"), eps=1e-5)
    assert float(jax.grad(f)(t0)) != 0.0


if __name__ == "__main__":
    np.savez_compressed(_GAP_FILE, width=NCAR_NEUTRAL_WIDTH, **_default_gaps(_fixture()))
    print("wrote", _GAP_FILE)
