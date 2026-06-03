"""SAM/SB2001 rain self-collection + breakup NRAGG tests (iter-22).

gSAM ``MICRO_M2005`` (module_mp_graupel.f90:1980-1986) evolves rain number by
the Seifert-Beheng (2001) self-collection + breakup::

    dum   = 1                              if 1/LAMR < 300µm   (self-collection)
          = 2 − exp(2300·(1/LAMR − 300µm)) if 1/LAMR ≥ 300µm   (→ breakup)
    NRAGG = −5.78·dum·q_r·N_r·ρ

legoESM's legacy form (``k_sc=1e-3`` sigmoid) self-collected ~5580× too WEAKLY,
leaving too many small rain drops (which then fall too slowly + evaporate too
fast through the PSD fall-speed / evaporation coupling).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    self_collection_breakup_sb2001,
)
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio


jax.config.update("jax_enable_x64", True)
_CFG = MorrisonConfig()


def _nragg(q_r, N_r, rho=1.0, dt=None):
    return float(self_collection_breakup_sb2001(
        jnp.asarray(N_r), jnp.asarray(q_r), jnp.asarray(rho), _CFG, dt=dt))


def _sam(q_r, N_r, rho=1.0):
    lamr = (math.pi * constants.rho_water * N_r / (rho * q_r)) ** (1.0 / 3.0)
    lamr = min(max(lamr, _CFG.lamr_min), _CFG.lamr_max)
    inv = 1.0 / lamr
    dum = 1.0 if inv < 300e-6 else 2.0 - math.exp(2300.0 * (inv - 300e-6))
    return -5.78 * dum * q_r * N_r * rho


def test_nragg_matches_sam_small_drops():
    """Many small drops (1/LAMR < 300µm) ⇒ pure self-collection = SAM NRAGG."""
    assert _nragg(1.0e-3, 1.0e4) == pytest.approx(_sam(1.0e-3, 1.0e4), rel=1e-6)
    assert _nragg(1.0e-3, 1.0e4) < 0.0          # N_r decreases


def test_nragg_breakup_increases_number_for_big_drops():
    """Few large drops (1/LAMR ≫ 300µm) ⇒ dum<0 ⇒ NRAGG>0 = breakup."""
    assert _nragg(5.0e-3, 1.0e2) > 0.0


def test_nragg_much_stronger_than_legacy():
    """The faithful self-collection is ~thousands× the legacy k_sc=1e-3 rate."""
    from legoesm.atmosphere.physics.microphysics._warm_rain import (
        self_collection_breakup,
    )
    dsc, dbr = self_collection_breakup(
        jnp.asarray(1.0e4), jnp.asarray(1.0e-3), jnp.asarray(1.0),
        _CFG.k_sc, _CFG.breakup_sharpness, _CFG.D_eq)
    legacy = float(dsc + dbr)
    assert abs(_nragg(1.0e-3, 1.0e4)) > 1000.0 * abs(legacy)


def test_nragg_relaxes_to_equilibrium_without_overshoot():
    """The explicit-Euler guard relaxes N_r toward the SB equilibrium drop
    size (1/LAMR≈601µm) WITHOUT overshooting it (codex iter-22 — better than a
    ±N_r/dt cap that would throttle breakup)."""
    d0, steep = _CFG.rain_breakup_d0, _CFG.rain_breakup_steepness
    lamr_eq = 1.0 / (d0 + math.log(2.0) / steep)

    def nr_eq(q_r, rho=1.0):
        return lamr_eq ** 3 * rho * q_r / (math.pi * constants.rho_water)

    dt = 20.0
    # Big drops (breakup, N_r below equilibrium): N_r rises toward but not
    # past nr_eq.
    q_r, N_r = 5.0e-3, 1.0e2
    N_r_new = N_r + _nragg(q_r, N_r, dt=dt) * dt
    assert N_r < N_r_new <= nr_eq(q_r) + 1.0
    # Small drops (self-collection, N_r above equilibrium): N_r falls toward
    # nr_eq, never below it, always ≥ 0.
    q_r2, N_r2 = 1.0e-3, 1.0e5
    N_r2_new = N_r2 + _nragg(q_r2, N_r2, dt=dt) * dt
    assert nr_eq(q_r2) - 1.0 <= N_r2_new < N_r2
    assert N_r2_new >= 0.0


def test_nragg_zero_without_rain():
    assert _nragg(0.0, 0.0) == 0.0
    assert _nragg(1.0e-9, 1.0e4) == 0.0         # below SAM QR≥1e-8 gate


def test_nragg_ad_safe():
    def loss(args):
        qr, nr = args
        return self_collection_breakup_sb2001(
            nr.reshape(1, 1), qr.reshape(1, 1), jnp.asarray(1.0),
            _CFG, dt=20.0).sum()

    for qr0, nr0 in [(0.0, 0.0), (1.0e-3, 1.0e4), (0.0, 1.0e4)]:
        g = jax.grad(loss)((jnp.asarray(qr0), jnp.asarray(nr0)))
        assert bool(jnp.all(jnp.isfinite(jnp.asarray([g[0], g[1]]))))


def test_unknown_rain_selfcoll_scheme_raises():
    qsat = float(saturation_mixing_ratio(jnp.asarray(290.0), jnp.asarray(9.0e4)))
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=jnp.full((1, 1), 1.0e-3), q_i=z, q_s=z, q_g=z,
        N_c=z, N_r=jnp.full((1, 1), 1.0e4), N_i=z,
    )
    with pytest.raises(ValueError, match="Unknown rain_selfcoll_scheme"):
        morrison_microphysics(
            jnp.full((1, 1), 290.0), jnp.full((1, 1), 0.9 * qsat), hm,
            jnp.full((1, 1), 9.0e4), jnp.full((1, 2), 9.0e4),
            jnp.full((1, 1), 1.08), jnp.full((1, 1), 300.0), 20.0,
            MorrisonConfig(rain_selfcoll_scheme="sb_typo"),
        )
