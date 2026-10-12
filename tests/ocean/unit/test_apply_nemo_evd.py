"""NEMO zdfevd slot for the unstructured lanes (MPAS, FESOM): apply_nemo_evd.

zdfevd.F90:107-110 overwrites avt with rn_evd where bn2 fires; the closure is
left untouched elsewhere.  The fired set is checked against an independent
bn2 evaluation (compute_buoyancy_frequency_nemo_bn2), not against the kernel
the helper calls.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy

set_policy(PrecisionPolicy.fp64())

import jax.numpy as jnp  # noqa: E402

from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2  # noqa: E402
from legoesm.ocean.physics.convection.config import (  # noqa: E402
    EnhancedDiffusionConfig,
)
from legoesm.ocean.physics.convection.enhanced_diffusion import (  # noqa: E402
    apply_nemo_evd,
)

NLEV = 6
THR = -1.0e-12          # zdfevd.F90:108
RN_EVD = 100.0          # ORCA1 namzdf rn_evd
K_CLOSURE = 1.0e-3


def _cfg(**kw):
    base = dict(K_conv=RN_EVD, K_bg=0.0, nu_conv=0.0, nu_bg=0.0,
                n2_mode="nemo_bn2", n2_eos_form="teos10",
                smooth_transition=False, n2_threshold=THR,
                evd_composition="nemo_replace")
    base.update(kw)
    return EnhancedDiffusionConfig(**base)


def _columns():
    dz = jnp.full((NLEV,), 10.0)
    t_depth = jnp.broadcast_to(5.0 + 10.0 * jnp.arange(NLEV), (2, NLEV))
    w_depth = jnp.broadcast_to(10.0 + 10.0 * jnp.arange(NLEV - 1), (2, NLEV - 1))
    e3w = jnp.full((2, NLEV - 1), 10.0)
    T = jnp.array([[25.0, 24.0, 23.0, 22.0, 21.0, 20.0],     # stable
                   [25.0, 24.0, 22.0, 23.5, 21.0, 20.0]])    # inversion at iface 2
    S = jnp.full((2, NLEV), 35.0)
    J = jnp.ones((2,))
    return dict(T=T, S=S, dz_ref=dz, jacobian=J, t_depth=t_depth,
                w_depth=w_depth, e3w_int=e3w)


def test_replaces_exactly_where_bn2_fires_and_nowhere_else():
    c = _columns()
    K = jnp.full((2, NLEV - 1), K_CLOSURE)
    out = np.asarray(apply_nemo_evd(K, _cfg(), active=jnp.ones_like(K, bool), **c))
    n2 = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        c["T"], c["S"], c["t_depth"], c["w_depth"], None, eos_form="teos10",
        e3w_int=c["e3w_int"]))
    fired = n2 < THR
    assert fired.sum() == 1 and fired[1, 2]                 # the planted inversion
    np.testing.assert_array_equal(out[fired], RN_EVD)       # rn_evd ALONE
    np.testing.assert_array_equal(out[~fired], K_CLOSURE)   # closure untouched


def test_dry_interface_never_fires():
    c = _columns()
    K = jnp.full((2, NLEV - 1), K_CLOSURE)
    active = jnp.ones_like(K, bool).at[1, 2].set(False)
    out = np.asarray(apply_nemo_evd(K, _cfg(), active=active, **c))
    np.testing.assert_array_equal(out, K_CLOSURE)


@pytest.mark.parametrize("bad", [dict(n2_mode="insitu"),
                                 dict(evd_composition="additive"),
                                 dict(nu_conv=RN_EVD)])
def test_only_nemos_own_card_is_accepted(bad):
    c = _columns()
    K = jnp.full((2, NLEV - 1), K_CLOSURE)
    with pytest.raises(ValueError):
        apply_nemo_evd(K, _cfg(**bad), active=jnp.ones_like(K, bool), **c)
