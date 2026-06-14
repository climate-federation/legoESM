"""Unit tests for the prescribed sea-ice -> ocean salt-mass flux apply
(``apply_ice_salt_flux_step``) and its canonical PSU closure.

legoESM has no interactive sea-ice model, so the brine-rejection / melt
salt exchange is prescribed from a NEMO ORCA1 icemod ``sfxice`` field and
injected as a top-cell salinity SOURCE.  These tests pin:

(a) SIGN -- positive salt flux (brine rejection) SALTENS the top cell;
(b) MAGNITUDE -- the Arctic-balance sanity (sfxice=1.37e-7, h1=1 m,
    dt=1 yr -> dS ~ +4.2 PSU, the right order to counter the rivers);
(c) thin vs thick top cell -- thinner h_top -> larger dS (same salt mass);
(d) land-masked cells unchanged;
(e) the underlying JAX closure ``salt_flux_salinity_tendency`` is JIT +
    autodiff safe (the host wrapper is NumPy, like the sibling
    sss/runoff/geothermal applicators);
(f) Field metadata + dtype preserved, and a ``None`` flux is an exact
    no-op.

Convention confirmed against NEMO ``src/OCE/TRA/trasbc.F90`` line ~137
(``sbc_tsc(jp_sal) = r1_rho0 * sfx``) + lines ~152-153 (tendency
``/ e3t1``): ``dS/dt|salt = sfx/(rho0*h_top)``, x1000 because ``sfx`` is
a salt MASS flux while NEMO salinity is PSU (g/kg).
"""
from __future__ import annotations

import collections

import jax

jax.config.update("jax_enable_x64", True)  # sci test: float64 PSU identities

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.core.field import Field  # noqa: E402
from legoesm.ocean.coupler.ice_salt_apply import apply_ice_salt_flux_step  # noqa: E402
from legoesm.ocean.freshwater import salt_flux_salinity_tendency  # noqa: E402

RHO = float(constants.rho_ocean)
_SEC_PER_YEAR = 365.0 * 86400.0


def _state(S, land_mask, dtype=np.float64, dims=("y", "x", "z")):
    """Minimal C-grid-like ocean state: ``S`` (..., nlev) + ``land_mask`` (...)."""
    _State = collections.namedtuple("_State", ["S", "land_mask"])
    S = np.asarray(S, dtype=dtype)
    lm = np.asarray(land_mask, dtype=dtype)
    return _State(
        S=Field(jnp.asarray(S), name="S", dims=dims, units="psu"),
        land_mask=Field(jnp.asarray(lm), name="land_mask",
                        dims=dims[:-1], units="1"),
    )


def test_positive_salt_flux_saltens_top_cell():
    # Brine rejection (sfxice > 0) raises the surface salinity; deeper layer
    # untouched.
    S0 = np.array([[[34.0, 35.0]]])               # (1,1,nlev=2)
    st = _state(S0, land_mask=[[1.0]])
    out = apply_ice_salt_flux_step(
        st, salt_flux_kg_m2_s=np.array([[1.0e-6]]),
        h_top=np.array([[1.0]]), dt=600.0)
    Sn = np.asarray(out.S.data)
    assert Sn[0, 0, 0] > S0[0, 0, 0]              # top saltened
    np.testing.assert_array_equal(Sn[0, 0, 1], S0[0, 0, 1])  # below unchanged


def test_negative_salt_flux_freshens_top_cell():
    # Melt / dilution (sfxice < 0) lowers the surface salinity.
    S0 = np.array([[[34.0, 35.0]]])
    st = _state(S0, land_mask=[[1.0]])
    out = apply_ice_salt_flux_step(
        st, salt_flux_kg_m2_s=np.array([[-1.0e-6]]),
        h_top=np.array([[1.0]]), dt=600.0)
    assert np.asarray(out.S.data)[0, 0, 0] < S0[0, 0, 0]


def test_arctic_balance_magnitude():
    # The key sanity: NEMO Arctic-mean sfxice ~ +1.37e-7 kg/m^2/s over a 1 m
    # top cell for one year gives dS ~ +4.2 PSU -- the right order to balance
    # the river freshening (which collapses the SSS without this term).
    S0 = np.array([[[30.0]]])                      # single-level column
    st = _state(S0, land_mask=[[1.0]], dims=("y", "x", "z"))
    out = apply_ice_salt_flux_step(
        st, salt_flux_kg_m2_s=np.array([[1.37e-7]]),
        h_top=np.array([[1.0]]), dt=_SEC_PER_YEAR, rho_0=RHO)
    dS = float(np.asarray(out.S.data)[0, 0, 0] - S0[0, 0, 0])
    # Closed form: dS = dt * 1000 * sfx / (rho0 * h1).
    dS_expected = _SEC_PER_YEAR * 1.0e3 * 1.37e-7 / (RHO * 1.0)
    np.testing.assert_allclose(dS, dS_expected, rtol=1e-12)
    assert 3.0 < dS < 6.0                          # Arctic-balance band


def test_thinner_top_cell_gives_larger_ds():
    # Same salt MASS flux: a thinner top cell concentrates it into less water
    # -> larger salinity change (dS ~ 1/h_top).
    S0 = np.array([[[34.0]], [[34.0]]])            # two columns (2,1,1)
    st = _state(S0, land_mask=[[1.0], [1.0]])
    out = apply_ice_salt_flux_step(
        st, salt_flux_kg_m2_s=np.array([[1.0e-6], [1.0e-6]]),
        h_top=np.array([[1.0], [10.0]]), dt=600.0)
    Sn = np.asarray(out.S.data)
    dS_thin = Sn[0, 0, 0] - S0[0, 0, 0]            # h_top = 1 m
    dS_thick = Sn[1, 0, 0] - S0[1, 0, 0]           # h_top = 10 m
    assert dS_thin > dS_thick > 0
    # dS ~ 1/h_top exactly; rtol=1e-9 absorbs the subtraction-cancellation
    # rounding floor of the (Sn - S0) PSU difference.
    np.testing.assert_allclose(dS_thin, 10.0 * dS_thick, rtol=1e-9)


def test_land_cell_unchanged():
    # land_mask = 0 -> the salt flux must not touch the cell (no flux through
    # walls / dry land), even with a nonzero prescribed flux there.
    S0 = np.array([[[34.0]], [[34.0]]])
    st = _state(S0, land_mask=[[1.0], [0.0]])      # cell 1 is land
    out = apply_ice_salt_flux_step(
        st, salt_flux_kg_m2_s=np.array([[1.0e-6], [1.0e-6]]),
        h_top=np.array([[1.0], [1.0]]), dt=600.0)
    Sn = np.asarray(out.S.data)
    assert Sn[0, 0, 0] > S0[0, 0, 0]               # ocean cell changed
    np.testing.assert_array_equal(Sn[1, 0, 0], S0[1, 0, 0])  # land unchanged


def test_thin_dry_top_cell_zeroed():
    # The canonical closure zeroes h_top <= 1 mm (thin/dry partial top cell):
    # a vanishing top cell must not produce a huge 1/h salinity blow-up.
    S0 = np.array([[[34.0]]])
    st = _state(S0, land_mask=[[1.0]])
    out = apply_ice_salt_flux_step(
        st, salt_flux_kg_m2_s=np.array([[1.0e-6]]),
        h_top=np.array([[1.0e-6]]), dt=600.0)      # 1 micron -> below guard
    np.testing.assert_array_equal(np.asarray(out.S.data)[0, 0, 0], S0[0, 0, 0])


def test_none_flux_is_exact_noop():
    # salt_flux=None lets the caller gate the apply without a call-site branch.
    S0 = np.array([[[34.0, 35.0]]])
    st = _state(S0, land_mask=[[1.0]])
    out = apply_ice_salt_flux_step(
        st, salt_flux_kg_m2_s=None, h_top=np.array([[1.0]]), dt=600.0)
    assert out is st                               # original object returned
    np.testing.assert_array_equal(np.asarray(out.S.data), S0)


def test_field_metadata_and_dtype_preserved():
    # name/dims/units survive the update, and the salinity dtype is NOT silently
    # widened (f32 stays f32 even under x64).
    S0 = np.array([[[34.0, 35.0]]], dtype=np.float32)
    st = _state(S0, land_mask=[[1.0]], dtype=np.float32,
                dims=("yc", "xc", "lev"))
    out = apply_ice_salt_flux_step(
        st, salt_flux_kg_m2_s=np.array([[1.0e-6]]),
        h_top=np.array([[1.0]]), dt=600.0)
    assert out.S.name == "S"
    assert out.S.dims == ("yc", "xc", "lev")
    assert out.S.units == "psu"
    assert np.asarray(out.S.data).dtype == np.float32


def test_mpas_shape_top_cell_only():
    # Grid-agnostic: an MPAS-style (nCells, nlev) state updates only the
    # surface layer via the same [..., 0] indexing.
    S0 = np.array([[34.0, 35.0, 36.0],             # (nCells=3, nlev=3)
                   [33.0, 34.0, 35.0],
                   [32.0, 33.0, 34.0]])
    st = _state(S0, land_mask=[1.0, 1.0, 0.0], dims=("nCells", "z"))
    out = apply_ice_salt_flux_step(
        st, salt_flux_kg_m2_s=np.array([1.0e-6, 1.0e-6, 1.0e-6]),
        h_top=np.array([1.0, 1.0, 1.0]), dt=600.0)
    Sn = np.asarray(out.S.data)
    assert Sn[0, 0] > S0[0, 0] and Sn[1, 0] > S0[1, 0]   # ocean saltened
    np.testing.assert_array_equal(Sn[2, 0], S0[2, 0])    # land unchanged
    np.testing.assert_array_equal(Sn[:, 1:], S0[:, 1:])  # sub-surface untouched


def test_closure_jit_and_grad_safe():
    # The host wrapper is NumPy (like sss/runoff/geothermal apply); the
    # DIFFERENTIABLE numerics it wraps is salt_flux_salinity_tendency -- assert
    # that is JIT + autodiff safe with the analytic gradient.
    h_top = jnp.asarray(np.array([[2.0]]))
    f = jax.jit(lambda sfx: jnp.sum(
        salt_flux_salinity_tendency(sfx, h_top, RHO)))
    val = float(f(1.0e-6))
    assert np.isfinite(val) and val > 0
    # d(dS/dt)/d(sfx) = 1e3 / (rho0 * h_top) (linear in the flux).
    g = float(jax.grad(f)(1.0e-6))
    np.testing.assert_allclose(g, 1.0e3 / (RHO * 2.0), rtol=1e-9)


def test_closure_matches_nemo_trasbc_formula():
    # Pin the exact NEMO trasbc.F90 form: dS/dt = 1000 * sfx / (rho0 * h_top).
    sfx = 1.37e-7
    h_top = 1.0
    dS_dt = float(salt_flux_salinity_tendency(
        np.array([sfx]), np.array([h_top]), RHO)[0])
    np.testing.assert_allclose(
        dS_dt, 1.0e3 * sfx / (RHO * h_top), rtol=1e-12)
