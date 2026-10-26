"""Oracle fixture: the MOST stability functions ``psi_m`` / ``psi_h`` reproduce
CliMA SurfaceFluxes.jl's universal functions.

Oracle: SurfaceFluxes.jl v1.5.1 (fd42723427b5264b3e5f8ee9f622cc7b2e94fdae;
Businger, Gryanik, Grachev) and v0.12.0 (167c5144c0171cc425432ad26d6569552a73ee11;
Beljaars-Holtslag, removed later), run on Julia 1.10.10 with parameter sets
equal to legoESM's coefficients.  The fixture
``oracle_fixtures/surfacefluxes_psi.npz`` holds the zeta grid and the oracle
values; regenerate with scripts/validate/oracle/surfacefluxes_psi/{build.sh,
make_fixture.py}.

Domain: |zeta| <= 10, where legoESM's psi functions are defined (they clip
zeta to +-10; the solvers clip before calling them).  The Beljaars-Holtslag
comparison is the stable branch only: legoESM pairs every stable fit with the
Businger-Dyer gamma = 16 unstable branch, while SurfaceFluxes v0.12 pairs BH91
with its own unstable form (a documented legoESM design, not part of this port).
Pass: field-normalised max error (the tier-1 CLUBB oracle metric) <= 1e-10
(fp64), and exact zero where the oracle is zero (zeta = 0).
"""
from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.core.bulk_flux import psi_h, psi_m  # noqa: E402

_FIXTURE = Path(__file__).parent / "oracle_fixtures" / "surfacefluxes_psi.npz"
_RTOL = 1e-10
_SCHEMES = ("dyer1974", "gryanik2020", "grachev2007_sheba", "beljaars_holtslag1991")


def _fixture():
    return np.load(_FIXTURE)


def _domain(z, scheme):
    m = np.abs(z) <= 10.0
    if scheme == "beljaars_holtslag1991":
        m &= z >= 0.0
    return m


@pytest.mark.parametrize("scheme", _SCHEMES)
@pytest.mark.parametrize("fn", [psi_m, psi_h], ids=["psi_m", "psi_h"])
def test_psi_matches_surfacefluxes(scheme, fn):
    d = _fixture()
    z = d["zeta"]
    m = _domain(z, scheme)
    ours = np.asarray(fn(jnp.asarray(z[m]), scheme))
    ref = d[f"{scheme}_{fn.__name__}"][m]
    zero = ref == 0.0
    assert np.any(zero)  # zeta = 0 is on the grid
    np.testing.assert_array_equal(ours[zero], 0.0)
    # Field-normalised error (tests/atmosphere/test_clubb_core_oracle.py
    # max_rel_errors): psi crosses zero at zeta = 0, so a pointwise relative
    # error at |zeta| ~ 1e-6 only measures the ulp difference of the cbrt/exp
    # implementations amplified by the cancellation in cbrt(1 + b zeta) - 1.
    err = np.max(np.abs(ours - ref)) / max(np.max(np.abs(ref)), 1e-300)
    assert err <= _RTOL, (scheme, fn.__name__, float(err))


@pytest.mark.parametrize("scheme", _SCHEMES)
def test_psi_gradient_finite_including_zero(scheme):
    z = jnp.asarray(_fixture()["zeta"])
    for fn in (psi_m, psi_h):
        g = jax.vmap(jax.grad(lambda x: fn(x, scheme)))(z)
        assert np.all(np.isfinite(np.asarray(g))), (scheme, fn.__name__)


@pytest.mark.parametrize("scheme", _SCHEMES)
def test_psi_gradient_at_neutral_is_the_unstable_branch_slope(scheme):
    """At zeta = 0 SurfaceFluxes selects the unstable form; its full one-sided
    slope must come through (a min/max tie would halve it): psi_m'(0) =
    -gamma/4, psi_h'(0) = -Pr0 gamma/2 with gamma = 16."""
    pr0 = 0.98 if scheme == "gryanik2020" else 1.0
    gm = float(jax.grad(lambda x: psi_m(x, scheme))(0.0))
    gh = float(jax.grad(lambda x: psi_h(x, scheme))(0.0))
    np.testing.assert_allclose([gm, gh], [-16.0 / 4.0, -pr0 * 16.0 / 2.0], rtol=_RTOL)
