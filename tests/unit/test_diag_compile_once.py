"""The host-side diagnostics that run every interval compile once.

Eager ``lax.scan`` / ``fori_loop`` calls with a fresh closure recompiled at every
diagnostic interval and kept each executable's code mapping, which ended a long
MPAS run with LLVM "Cannot allocate memory".  The 2 m MOST diagnostic and the
column static energies now go through cached jits; these tests pin that (cache
size stays 1 across calls with new values) and that the values are unchanged.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.core.bulk_flux import compute_most_fluxes  # noqa: E402
from legoesm.diagnostics.energy_budget import (  # noqa: E402
    column_dry_static_energy, column_moist_static_energy)
from legoesm.driver import diagnostics as D  # noqa: E402


def _surface(seed, n=64):
    r = np.random.default_rng(seed)
    T = 295.0 + 3.0 * r.random(n)
    return (jnp.asarray(5.0 * r.standard_normal(n)), jnp.asarray(5.0 * r.standard_normal(n)),
            jnp.asarray(T), jnp.asarray(0.015 + 0.002 * r.random(n)),
            jnp.asarray(T + 1.0 + r.random(n)), jnp.asarray(0.02 + 0.001 * r.random(n)),
            jnp.asarray(1.15 + 0.02 * r.random(n)))


def test_most_2m_compiles_once_and_matches_the_eager_call():
    D._most_2m_jit.clear_cache()
    for seed in range(3):
        args = _surface(seed)
        out = D._most_2m_jit(*args, scheme="coare3", stability_scheme="dyer1974")
        ref = compute_most_fluxes(*args, scheme="coare3", return_2m=True,
                                  stability_scheme="dyer1974")
        np.testing.assert_allclose(np.asarray(out[-1]), np.asarray(ref[-1]), rtol=1e-12, atol=0)
    assert D._most_2m_jit._cache_size() == 1


def _column(seed, n=16, nlev=8):
    r = np.random.default_rng(seed)
    sig_half = np.linspace(0.0, 1.0, nlev + 1)
    sig = 0.5 * (sig_half[1:] + sig_half[:-1])
    T = 200.0 + 90.0 * sig[None, :] + r.random((n, nlev))
    return dict(T=jnp.asarray(T), q_v=jnp.asarray(0.01 * sig[None, :] * r.random((n, nlev))),
                u=jnp.asarray(r.standard_normal((n, nlev))), v=jnp.asarray(r.standard_normal((n, nlev))),
                phis=jnp.asarray(100.0 * r.random(n)), p_s=jnp.asarray(1.0e5 - 500.0 * r.random(n)),
                dsigma=jnp.asarray(np.diff(sig_half)), sigma_full=jnp.asarray(sig))


def test_column_static_energies_compile_once():
    for f in (column_moist_static_energy, column_dry_static_energy):
        f.clear_cache()
        vals = []
        for seed in range(3):
            kw = _column(seed)
            if f is column_dry_static_energy:
                for k in ("q_v", "u", "v"):
                    kw.pop(k)
            vals.append(np.asarray(f(**kw)))
        assert f._cache_size() == 1
        assert all(np.isfinite(v).all() for v in vals)
