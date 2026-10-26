"""Oracle fixture: ``compute_sam_oceflx_fluxes(variant="cesm")`` reproduces the
COMPILED CESM2.1 ocean surface-flux law.

Oracle: CIME tag cime5.6.47 (SHA d8e9b894d7eb96fe96f8f8a6943b421147f94606),
``src/share/util/shr_flux_mod.F90`` (sha1 40295186e9fcd7d5b099c9451534ad42dfcc8a58)
subroutine ``shr_flux_atmOcn``, compiled unmodified and run with legoESM's
constants and the CESM2.1 f09 driver settings (flux_max_iteration = 5,
flux_convergence = 0, coldair_outbreak_mod = .true., minwind 0.5, gust_fac 0).
The fixture ``oracle_fixtures/shr_flux_atmocn_cesm21.npz`` holds the oracle's
inputs and outputs (1054 states: ERA5 open ocean incl. 42 cold-air-outbreak
points, six named regimes, eight synthetic edges incl. one where the
flux_convergence = 0 early exit binds); its ``meta`` string records
the provenance.  Regenerate with::

    bash scripts/validate/oracle/shr_flux_atmocn/build.sh <cime5.6.47> <build>
    python scripts/validate/oracle/shr_flux_atmocn/make_fixture.py \\
        --drv <build>/drv --out tests/core/oracle_fixtures/shr_flux_atmocn_cesm21.npz

The oracle's own sea-surface humidity ``ssq`` is fed in as ``q_sfc`` (CESM's
private qsat fit is not the model's curve).  CESM fluxes are positive DOWN and
its stress is the stress on the ocean along the wind, so the model's fluxes and
stress are compared with a sign flip.  Pass: max relative error <= 1e-10
(fp64), and exact agreement where the oracle output is exactly zero.
"""
from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from legoesm.core.bulk_flux import compute_sam_oceflx_fluxes  # noqa: E402

_FIXTURE = Path(__file__).parent / "oracle_fixtures" / "shr_flux_atmocn_cesm21.npz"
_RTOL = 1e-10


def _fixture():
    return np.load(_FIXTURE)


def _ours(d):
    j = {k: jnp.asarray(d[k]) for k in d.files if k.startswith(("in_", "oracle_"))}
    tx, ty, sh, lh, us, tref = compute_sam_oceflx_fluxes(
        j["in_ubot"], j["in_vbot"], j["in_thbot"], j["in_qbot"], j["in_ts"],
        j["oracle_ssq"], j["in_rbot"], z_bot=j["in_zbot"], t_bot=j["in_tbot"],
        variant="cesm", return_2m=True)
    return {"taux": -tx, "tauy": -ty, "sen": -sh, "lat": -lh, "ustar": us, "tref": tref}


def test_fixture_covers_the_cold_air_outbreak_and_both_stabilities():
    d = _fixture()
    assert "cime5.6.47" in str(d["meta"]) and "coldair_outbreak_mod=.true." in str(d["meta"])
    assert int(np.sum(d["in_tbot"] - d["in_ts"] < -10.0)) >= 20
    delt = d["in_thbot"] - d["in_ts"]
    assert (delt > 0).sum() >= 50 and (delt < 0).sum() >= 50
    # a state where the Fortran exact-ustar early exit changes the answer
    assert "edge_exact_ustar_repeat_early_exit" in set(d["labels"])


def test_cesm_variant_matches_compiled_shr_flux_atmocn():
    d = _fixture()
    ours = _ours(d)
    for k, a in ours.items():
        a = np.asarray(a)
        b = d[f"oracle_{k}"]
        zero = b == 0.0
        np.testing.assert_array_equal(a[zero], 0.0, err_msg=k)
        rel = np.abs(a[~zero] - b[~zero]) / np.abs(b[~zero])
        assert rel.max() <= _RTOL, (k, float(rel.max()))


def test_cesm_variant_gradient_finite_on_fixture():
    d = _fixture()
    J = lambda k: jnp.asarray(d[k])  # noqa: E731

    def loss(u, th, q, ts, tb):
        r = compute_sam_oceflx_fluxes(
            u, J("in_vbot"), th, q, ts, J("oracle_ssq"), J("in_rbot"),
            z_bot=J("in_zbot"), t_bot=tb, variant="cesm")
        return jnp.sum(r[0] + r[2] + r[3] + r[4])

    g = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(
        J("in_ubot"), J("in_thbot"), J("in_qbot"), J("in_ts"), J("in_tbot"))
    for x in g:
        assert np.all(np.isfinite(np.asarray(x)))
    # t_bot carries gradient exactly where the cold-air-outbreak block is active
    coa = np.asarray(d["in_tbot"] - d["in_ts"] < -10.0)
    assert np.all(np.asarray(g[4])[coa] != 0.0) and np.all(np.asarray(g[4])[~coa] == 0.0)


def test_generator_driver_constants_are_legoesm_constants():
    """The oracle driver hard-codes legoESM's constants (driver.F90); the
    generator's mirror of them must equal both the Fortran and
    ``legoesm.constants``, or a regenerated fixture tests other constants."""
    import re

    from legoesm import constants
    from scripts.validate.oracle.shr_flux_atmocn import make_fixture as mf

    for k, v in mf.DRIVER_LEGO.items():
        assert getattr(constants, k) == v, k
    src = (Path(mf.__file__).parent / "driver.F90").read_text()
    fort = {m[0]: float(m[1]) for m in re.findall(r"L_(\w+)=([0-9.e]+)_R8", src)}
    name = {"RD": "R_d", "RV": "R_v", "CPD": "c_pd", "CPV": "c_pv", "G": "g",
            "KARMAN": "kappa_von_karman", "LV": "L_v"}
    assert set(fort) == set(name)
    for f, k in name.items():
        assert fort[f] == mf.DRIVER_LEGO[k], f
