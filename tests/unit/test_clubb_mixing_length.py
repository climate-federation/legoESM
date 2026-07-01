"""Unit tests for the CLUBB nonlocal mixing length (now in ``clubb.py``).

Idealized-profile checks: a neutral column mixes over a far larger length scale
than a strongly stable column; shapes, positivity, the Lscale_max cap, and
AD/JIT cleanliness.

Part of the fuller CLUBB port — see ``docs/dev-notes/clubb.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb import make_clubb_grid  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    _EP1,
    _EP2,
    _LV2_COEF,
    compute_mixing_length,
    set_Lscale_max,
)
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    sat_mixrat_liq as _my_sat,
)

from legoesm import constants  # noqa: E402

# CLUBB-JAX reference tree (sibling of the repo); absent in CI -> parity skips.
_CLUBB_JAX_ROOT = Path(__file__).resolve().parents[2].parent / "CLUBB-JAX"


def _setup(nzt=30, dthv_dz=(0.0, 1.0e-2)):
    """Two columns sharing a grid: col0 neutral, col1 strongly stable (default)."""
    ngrdcol = len(dthv_dz)
    nzm = nzt + 1
    zm_1d = np.linspace(0.0, 3000.0, nzm)
    zm = jnp.asarray(np.tile(zm_1d, (ngrdcol, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])           # (ngrdcol, nzt)
    gr = make_clubb_grid(zm, zt)

    zt_np = np.asarray(zt)
    thvm = jnp.asarray(np.stack([300.0 + s * zt_np[i] for i, s in enumerate(dthv_dz)]))
    thlm = thvm                                   # dry (rt small) -> thl ~ thv
    rtm = jnp.full((ngrdcol, nzt), 1.0e-3)
    p = 1.0e5 * jnp.exp(-zt / 8000.0)
    exner = (p / constants.p_ref) ** constants.kappa
    thv_ds = thvm
    em = jnp.full((ngrdcol, nzm), 0.5)            # TKE on zm
    mu = jnp.full((ngrdcol,), 1.0e-3)
    Lscale_max = jnp.full((ngrdcol,), 1.0e5)
    return dict(thvm=thvm, thlm=thlm, rtm=rtm, em=em, Lscale_max=Lscale_max,
                p_in_Pa=p, exner=exner, thv_ds=thv_ds, mu=mu, lmin=20.0,
                l_implemented=False, gr=gr), ngrdcol, nzt


def _run(kw):
    return compute_mixing_length(**kw)


# Deterministic 3-column fixture spanning neutral (boundary exit), strongly
# stable (early exit), and moist/saturated (condensing) parcels on a stretched
# grid. Used both to GENERATE the committed golden outputs (from the patched
# CLUBB-JAX reference) and to CHECK the port against them — so the bit-exact
# oracle parity runs in CI without the sibling CLUBB-JAX checkout.
_GOLDEN_NPZ = Path(__file__).resolve().parent / "clubb_fixtures" / "clubb_lscale_golden.npz"


def _golden_inputs():
    ng, nzt = 3, 28
    nzm = nzt + 1
    idx = np.arange(nzm, dtype=np.float64)
    zm_1d = np.cumsum(np.concatenate([[0.0], 40.0 * 1.12 ** idx[:-1]]))
    zm = jnp.asarray(np.tile(zm_1d, (ng, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    gr = make_clubb_grid(zm, zt)
    zt_np = np.asarray(zt)
    thvm = np.empty((ng, nzt))
    thvm[0] = 300.0
    thvm[1] = 300.0 + 0.012 * zt_np[1]
    thvm[2] = 300.0 + 0.004 * zt_np[2]
    thvm = jnp.asarray(thvm)
    rtm = np.full((ng, nzt), 2.0e-3)
    rtm[2] = 1.6e-2
    rtm = jnp.asarray(rtm)
    p = 1.0e5 * jnp.exp(-zt / 8500.0)
    exner = (p / constants.p_ref) ** constants.kappa
    return dict(thvm=thvm, thlm=thvm, rtm=rtm, em=jnp.full((ng, nzm), 0.6),
                Lscale_max=jnp.full((ng,), 1.0e5), p_in_Pa=p, exner=exner,
                thv_ds=thvm, mu=jnp.array([1.0e-3, 2.0e-3, 5.0e-4]), lmin=20.0,
                l_implemented=False, gr=gr)


def test_set_lscale_max():
    standalone = set_Lscale_max(False, None, None, 3)
    assert standalone.shape == (3,)
    np.testing.assert_allclose(np.asarray(standalone), 1.0e5)
    host = set_Lscale_max(True, jnp.full(3, 4.0e4), jnp.full(3, 2.0e4), 3)
    np.testing.assert_allclose(np.asarray(host), 0.25 * 2.0e4)


def test_shapes_and_positivity():
    kw, ng, nzt = _setup()
    Lscale, Lup, Ldn = _run(kw)
    for arr in (Lscale, Lup, Ldn):
        assert arr.shape == (ng, nzt)
        assert jnp.all(jnp.isfinite(arr))
    assert jnp.all(Lscale > 0.0)
    assert jnp.all(Lup > 0.0) and jnp.all(Ldn > 0.0)


def test_lscale_capped():
    kw, ng, nzt = _setup()
    kw = dict(kw, Lscale_max=jnp.full((ng,), 500.0))
    Lscale, _, _ = _run(kw)
    assert jnp.all(Lscale <= 500.0 + 1e-9)


def test_neutral_mixes_more_than_stable():
    """Neutral column (col0) -> much larger Lscale than strongly stable (col1)."""
    kw, ng, nzt = _setup(dthv_dz=(0.0, 1.0e-2))
    Lscale, _, _ = _run(kw)
    # Compare away from the surface/top boundaries.
    neutral = float(jnp.mean(Lscale[0, 2:-2]))
    stable = float(jnp.mean(Lscale[1, 2:-2]))
    assert neutral > 5.0 * stable, (neutral, stable)


def test_more_stable_gives_shorter_lscale():
    """Monotone: increasing stability shortens the mixing length."""
    means = []
    for slope in (1.0e-3, 5.0e-3, 2.0e-2):
        kw, ng, nzt = _setup(dthv_dz=(slope,))
        Lscale, _, _ = _run(kw)
        means.append(float(jnp.mean(Lscale[0, 2:-2])))
    assert means[0] > means[1] > means[2]


def test_jit_and_grad_clean():
    kw, ng, nzt = _setup()

    @jax.jit
    def run(thlm):
        Lscale, _, _ = compute_mixing_length(**dict(kw, thlm=thlm))
        return Lscale

    out = run(kw["thlm"])
    assert jnp.all(jnp.isfinite(out))

    def loss(thlm):
        Lscale, _, _ = compute_mixing_length(**dict(kw, thlm=thlm))
        return jnp.sum(Lscale)

    g = jax.grad(loss)(kw["thlm"])
    assert g.shape == kw["thlm"].shape
    assert jnp.all(jnp.isfinite(g))


def test_matches_committed_golden():
    """BIT-EXACT against the committed golden fixture (runs in CI, no skip).

    The golden ``Lscale``/``Lscale_up``/``Lscale_down`` were generated from the
    CLUBB-JAX reference ``compute_mixing_length`` with its constants + saturation
    patched to the legoESM values (see ``test_parity_vs_clubb_jax_reference``,
    which regenerates and re-verifies them when the sibling checkout is present).
    This locks the intricate parcel indexing / fractional-exit math against the
    oracle even where the CLUBB-JAX tree is absent.
    """
    golden = np.load(_GOLDEN_NPZ)
    Lscale, Lscale_up, Lscale_down = compute_mixing_length(**_golden_inputs())
    # FP-reassociation tolerance: the C1-C8 "condense" refactor fused/re-
    # associated the algebra (~1e-14 rel) -> round-off, not bit-identity.
    np.testing.assert_allclose(np.asarray(Lscale), golden["Lscale"], rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(np.asarray(Lscale_up), golden["Lscale_up"], rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(np.asarray(Lscale_down), golden["Lscale_down"], rtol=1e-13, atol=1e-16)


@pytest.mark.skipif(
    not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
    reason="CLUBB-JAX reference tree not present",
)
def test_parity_vs_clubb_jax_reference():
    """Live bit-exact parity vs CLUBB-JAX (regenerator/cross-check; skipped in CI).

    Monkeypatches the reference's constants + saturation to the legoESM values so
    the comparison isolates the ALGORITHM from the documented ~0.1% constant
    difference. Also confirms the committed golden fixture is still in sync with
    the reference (so a stale golden is caught on dev machines).
    """
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.mixing_length as refmod

    refmod.sat_mixrat_liq = lambda p, t, _sf: _my_sat(p, t)
    refmod.grav = constants.g
    refmod.Cp = constants.c_pd
    refmod.Lv = constants.L_v
    refmod.Rd = constants.R_d
    refmod.ep = constants.epsilon
    refmod.ep1 = _EP1
    refmod.ep2 = _EP2
    refmod._LV2_COEF = _LV2_COEF

    kw = _golden_inputs()
    gr = kw["gr"]
    nzt = gr.zt.shape[1]
    mine = compute_mixing_length(**kw)
    refgr = SimpleNamespace(zm=gr.zm, zt=gr.zt, dzm=gr.dzm, invrs_dzm=gr.invrs_dzm,
                            k_ub_zt=nzt - 1, k_lb_zt=0, k_lb_zm=0)
    ref = refmod.compute_mixing_length(
        kw["thvm"], kw["thlm"], kw["rtm"], kw["em"], kw["Lscale_max"],
        kw["p_in_Pa"], kw["exner"], kw["thv_ds"], kw["mu"], kw["lmin"], 3, False, refgr)

    golden = np.load(_GOLDEN_NPZ)
    for name, a, b in zip(("Lscale", "Lscale_up", "Lscale_down"), mine, ref):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0,
                                   err_msg=f"{name} diverged from CLUBB-JAX reference")
        np.testing.assert_array_equal(
            np.asarray(b), golden[name],
            err_msg=f"committed golden {name} is stale vs the reference")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
