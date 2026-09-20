"""CLUBB rad-heating ROUTING (MPAS turbulence wrapper path).

Covers the routing contract that hands the cached ``PhysicsState.rad_heating``
to the prognostic-CLUBB turbulence kernel as ``rad_dT_dt``:

  * ``_prognostic_clubb_rad_kwargs`` returns ``{"rad_dT_dt": ...}`` ONLY for
    prognostic CLUBB (``carry_field == "clubb_moments"``) with a populated
    cache — reshaped to top-down ``(ncol, nlev)``, cast to the state dtype;
  * diagnostic CLUBB (carry ``"tke"``) and every non-clubb scheme get ``{}``
    (no kwarg at all — their kernels do not accept it, byte-identical path);
  * prognostic CLUBB with an absent/None ``rad_heating`` (or no phys_state)
    also gets ``{}``.

The PHYSICAL effect of the radiative thlp2 source is covered by the CLUBB
budget tests; here we pin the plumbing/dispatch, which is factory/trace-time
(no JAX trace of a column) so it stays cheap, in the style of
test_clubb_cf_routing.py.
"""

from __future__ import annotations

import types

import jax.numpy as jnp

from legoesm.atmosphere.physics.turbulence.integration import (
    _prognostic_clubb_rad_kwargs,
)


class _FakePhysState:
    """Minimal PhysicsState-like namespace: only ``rad_heating`` matters."""

    def __init__(self, rad_heating):
        self.rad_heating = rad_heating


class TestRadHeatingRouting:
    def test_prognostic_clubb_with_rad_heating(self):
        """(a) prognostic CLUBB + finite cached rad_heating => kwarg passed,
        reshaped to (ncol, nlev) equal to rad_heating."""
        nCells, nlev = 4, 7
        rad = jnp.arange(nCells * nlev, dtype=jnp.float32).reshape(nCells, nlev)
        kw = _prognostic_clubb_rad_kwargs(
            "clubb_moments", _FakePhysState(rad), nCells, nlev, jnp.float32)
        assert set(kw) == {"rad_dT_dt"}
        got = kw["rad_dT_dt"]
        assert got.shape == (nCells, nlev)
        assert got.dtype == jnp.float32
        assert bool(jnp.array_equal(got, rad))

    def test_prognostic_clubb_reshapes_flat_cache(self):
        """The cache may arrive flattened; the kwarg is reshaped top-down."""
        nCells, nlev = 3, 5
        flat = jnp.arange(nCells * nlev, dtype=jnp.float32)
        kw = _prognostic_clubb_rad_kwargs(
            "clubb_moments", _FakePhysState(flat), nCells, nlev, jnp.float32)
        assert set(kw) == {"rad_dT_dt"}
        assert kw["rad_dT_dt"].shape == (nCells, nlev)
        assert bool(jnp.array_equal(kw["rad_dT_dt"], flat.reshape(nCells, nlev)))

    def test_diagnostic_clubb_no_kwarg(self):
        """(b) diagnostic CLUBB (carry ``"tke"``) => NO kwarg at all."""
        rad = jnp.ones((4, 7), dtype=jnp.float32)
        kw = _prognostic_clubb_rad_kwargs(
            "tke", _FakePhysState(rad), 4, 7, jnp.float32)
        assert kw == {}

    def test_prognostic_clubb_rad_heating_none_no_kwarg(self):
        """(c) prognostic CLUBB but rad_heating None => NO kwarg."""
        kw = _prognostic_clubb_rad_kwargs(
            "clubb_moments", _FakePhysState(None), 4, 7, jnp.float32)
        assert kw == {}

    def test_prognostic_clubb_missing_attr_no_kwarg(self):
        """A phys_state without the attribute (getattr default None) => {}."""
        bare = types.SimpleNamespace()  # no rad_heating
        kw = _prognostic_clubb_rad_kwargs(
            "clubb_moments", bare, 4, 7, jnp.float32)
        assert kw == {}

    def test_no_phys_state_no_kwarg(self):
        """No phys_state at all (one-off physics_fn call) => {}."""
        kw = _prognostic_clubb_rad_kwargs("clubb_moments", None, 4, 7, jnp.float32)
        assert kw == {}

    def test_other_schemes_no_kwarg(self):
        """Every non-clubb carry field stays byte-identical: no kwarg."""
        rad = jnp.ones((4, 7), dtype=jnp.float32)
        for carry in ("tke", "qke"):
            kw = _prognostic_clubb_rad_kwargs(
                carry, _FakePhysState(rad), 4, 7, jnp.float32)
            assert kw == {}
