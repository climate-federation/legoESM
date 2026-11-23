"""CLUBB-MF as a model option: CAM's Poisson draws, the forcing CAM hands
CLUBB, its Courant limiter (RULE 5), and the switch on the prognostic path.

Oracle for the draws: ``clubb_mf.npz`` holds the integer draws CAM's own
``poisson`` (CESM_share KISS) produced on each fixture column (see
test_clubb_mf_oracle.py).  Re-measure the stored limiter gaps with
``python test_clubb_mf_coupling.py`` when the smooth form or its default changes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm.atmosphere.physics.turbulence.clubb_mf import (
    CFL_SHARPNESS,
    CFLVAL,
    KSEED,
    L0,
    NUP,
    clubb_mf_forcing,
    entrainment_draws,
)

jax.config.update("jax_enable_x64", True)

FIXTURE = Path(__file__).resolve().parents[2] / "oracle_fixtures" / "clubb_mf.npz"
CFL_GAPS = FIXTURE.with_name("clubb_mf_cfl_gaps.json")
DT_ON_LIMITER = 600.0  # the fixture's dt; columns whose Courant number reaches 1 bind
C_KEYS = ("zm", "dzt", "u", "v", "thl", "qt", "thv", "p_zm", "iex_zm",
          "wthl_sfc", "wqt_sfc", "ths", "pblh", "lts")


@pytest.fixture(scope="module")
def fx():
    return dict(np.load(FIXTURE, allow_pickle=True))


def _col(fx, i):
    return {k: jnp.asarray(fx[f"c_{k}"][i]) for k in C_KEYS}


def test_draws_match_cam_generator(fx):
    """Every draw CAM's poisson made on each triggered fixture column, bit for bit
    (both the Knuth and the Hormann branch are exercised)."""
    meta = json.loads(str(fx["meta"]))
    assert meta["init_ints"][0] == 0 and meta["init_ints"][3] == KSEED  # Lopt 0, kseed
    assert meta["init_reals"][2] == L0
    n_hormann = 0
    for step in (1, 2):
        for i in range(fx["c_dzt"].shape[0]):
            if not np.any(fx[f"ora{step}_dL0"][i] > 0.0):
                continue
            dzt, u = fx["c_dzt"][i], fx["c_u"][i]
            lam = np.zeros((dzt.shape[0], NUP))
            lam[:-1] = (dzt[:-1] / L0)[:, None]
            n_hormann += int(np.sum(lam >= 10.0))
            got = entrainment_draws(jnp.asarray(lam), jnp.asarray(
                [u[0], u[KSEED], u[1 + KSEED], u[2 + KSEED]]))
            np.testing.assert_array_equal(np.asarray(got), fx[f"ora{step}_enti"][i],
                                          err_msg=f"step {step} column {i}")
    assert n_hormann > 0


def _forcing(fx, i, dt, cfl_sharpness=CFL_SHARPNESS):
    c = _col(fx, i)
    rzm = jnp.asarray(fx["c_rho_zm"][i])
    irzt = 1.0 / jnp.asarray(fx["c_rho_zt"][i])
    return clubb_mf_forcing(c, dt, rzm, irzt, 1.0 / c["dzt"], cfl_sharpness=cfl_sharpness)


def test_forcing_is_flux_form_and_conserves_the_column(fx):
    """Mass-weighted column integral of each forcing is zero (fluxes vanish at
    the surface and the top), and some column carries a nonzero forcing."""
    any_on = False
    for i in range(fx["c_dzt"].shape[0]):
        fthl, fqt, _ = _forcing(fx, i, DT_ON_LIMITER)
        w = jnp.asarray(fx["c_rho_zt"][i]) * jnp.asarray(fx["c_dzt"][i])
        for f in (fthl, fqt):
            scale = float(jnp.sum(jnp.abs(f) * w)) or 1.0
            assert abs(float(jnp.sum(f * w))) <= 1e-12 * scale, i
        any_on |= bool(jnp.any(fqt != 0.0))
    assert any_on


def _max_cfl(fx, i, dt):
    _, _, out = _forcing(fx, i, dt, math.inf)
    aw = np.sum(np.asarray(out["upa"]) * np.asarray(out["upw"]), axis=1)
    aw[0] = 0.0
    # CAM: dtime*invrs_dz_g(k)*max(|s_aw(k)|, |s_aw(k+1)|), invrs_dz_g = 1/dz
    return float(np.max(dt * (1.0 / fx["c_dzt"][i]) * np.maximum(np.abs(aw[:-1]), np.abs(aw[1:]))))


def test_courant_limiter_hard_limit_is_cams(fx):
    """At sharpness inf, cflfac = cflval/max_cfl where max_cfl > cflval, else 1."""
    for i in range(fx["c_dzt"].shape[0]):
        _, _, out = _forcing(fx, i, DT_ON_LIMITER, math.inf)
        m = _max_cfl(fx, i, DT_ON_LIMITER)
        # rtol: the plume sum's reduction order differs from CAM's loop (1 ulp)
        np.testing.assert_allclose(float(out["cflfac"]), CFLVAL / m if m > CFLVAL else 1.0,
                                   rtol=1e-12, atol=0.0, err_msg=str(i))


def _cfl_gaps(fx):
    out = []
    for i in range(fx["c_dzt"].shape[0]):
        hard = _forcing(fx, i, DT_ON_LIMITER, math.inf)
        soft = _forcing(fx, i, DT_ON_LIMITER)
        out.append({k: float(np.max(np.abs(np.asarray(s) - np.asarray(h))))
                    for k, s, h in (("thlm_forcing", soft[0], hard[0]),
                                    ("rtm_forcing", soft[1], hard[1]))})
    return out


def test_courant_limiter_default_gap_per_case(fx):
    """RULE 5 KNOWN DIFFERENCE: the smooth limiter at its default vs CAM's hard
    one, per fixture column, to the stored measurement."""
    stored = json.loads(CFL_GAPS.read_text())
    got = _cfl_gaps(fx)
    assert len(got) == len(stored)
    for i, (g, s) in enumerate(zip(got, stored)):
        for k in g:
            np.testing.assert_allclose(g[k], s[k], rtol=1e-6, atol=1e-12, err_msg=f"{i}/{k}")


def test_check_grads_through_courant_limiter_at_default(fx):
    """dt placed so the column sits exactly ON the limiter (max_cfl = cflval):
    check_grads through cflfac, nonzero and different from the hard form."""
    i = int(np.argmax([_max_cfl(fx, j, DT_ON_LIMITER) for j in range(fx["c_dzt"].shape[0])]))
    dt_on = DT_ON_LIMITER * CFLVAL / _max_cfl(fx, i, DT_ON_LIMITER)

    def f(dt, s=CFL_SHARPNESS):
        return _forcing(fx, i, dt, s)[2]["cflfac"]

    check_grads(f, (jnp.asarray(dt_on),), order=1, modes=("fwd", "rev"), eps=1e-3)
    g = float(jax.grad(f)(jnp.asarray(dt_on)))
    assert np.isfinite(g) and g != 0.0
    assert g != float(jax.grad(lambda d: f(d, math.inf))(jnp.asarray(dt_on)))


def test_prognostic_clubb_takes_the_mass_flux_in_the_scm():
    """The switch reaches the prognostic closure: on, the SCM's thl/rt change
    differs from off on a convective column; both finite."""
    from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from tests.atmosphere.hydrostatic.unit.test_scm import (
        _RCE_DT, _RCE_NLEV, _baseline_cfg, _tropical_qv_profile, _tropical_T_profile)

    out = []
    for on in (False, True):
        cfg = _baseline_cfg(turbulence=TurbulenceConfig(
            scheme="clubb", clubb=CLUBBConfig(prognostic=True), clubb_mf=on))
        scm = SingleColumnModel.create(
            physics_config=cfg, nlev=_RCE_NLEV, dt=_RCE_DT,
            T_profile=_tropical_T_profile(), q_v_profile=_tropical_qv_profile(),
            latitude_deg=0.0)
        final, _ = scm.run(nsteps=4)
        out.append(np.asarray(final.T.data))
    assert np.all(np.isfinite(out[0])) and np.all(np.isfinite(out[1]))
    assert np.max(np.abs(out[1] - out[0])) > 0.0


if __name__ == "__main__":
    d = dict(np.load(FIXTURE, allow_pickle=True))
    CFL_GAPS.write_text(json.dumps(_cfl_gaps(d), indent=1))
    print("wrote", CFL_GAPS, "max_cfl per column:",
          [round(_max_cfl(d, i, DT_ON_LIMITER), 3) for i in range(d["c_dzt"].shape[0])])
