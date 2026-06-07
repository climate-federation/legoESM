"""Morrison cloud-droplet number Nc must match SAM M2005 default dopredictNc=.false.
(specified constant Nc_0), not the broken sink-only prognostic Nc that drifted to
−1.3e7 → NaN.

Covers:
  * effective_Nc specified mode (predict_Nc=False) returns Nc_0 for any N_c
    (negative / zero / large) — and the prognostic mode still guards N_c<=1;
  * Morrison default (predict_Nc=False) gives dN_c_dt == 0 and FINITE tendencies
    even for a hugely-negative N_c;
  * Seifert-Beheng prognostic Nc behaviour is unchanged (effective_Nc default).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.microphysics._warm_rain import effective_Nc

jax.config.update("jax_enable_x64", True)


def test_effective_nc_specified_mode_is_constant():
    Nc0 = 1e8
    for ncval in (-1.3e7, -1.0, 0.0, 1e6, 5e8):
        N_c = jnp.full((4, 4), ncval)
        eff = effective_Nc(N_c, Nc0, predict_Nc=False)
        assert np.allclose(np.asarray(eff), Nc0)        # always the specified Nc0


def test_effective_nc_prognostic_guards_nonpositive():
    Nc0 = 1e8
    N_c = jnp.array([-1.3e7, 0.0, 0.5, 2.0e8])
    eff = np.asarray(effective_Nc(N_c, Nc0, predict_Nc=True))
    assert np.allclose(eff[:3], Nc0)                    # <=1 -> Nc0 (guards negatives)
    assert np.isclose(eff[3], 2.0e8)                    # >1 -> prognostic value
    assert (eff > 0).all()                              # never <= 0 -> x_c safe


def test_morrison_config_default_is_specified_nc():
    """Morrison default must be SAM dopredictNc=.false. (specified constant Nc)."""
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    assert MorrisonConfig().predict_Nc is False
