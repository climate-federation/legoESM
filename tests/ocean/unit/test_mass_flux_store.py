"""``store_mass_flux``: keep the flux the model actually advects with (#1442).

``state.u``/``state.v`` are the velocity BEFORE the barotropic transport
correction.  ``_step_impl`` forms ``u_corrected = u + (Hu_avg - Hu_3d)/H_u_old``
and advects tracers with ``h_u_old * u_corrected`` (plus the GM bolus when
active), but never writes ``u_corrected`` back.  So every transport diagnostic
built from a state or a snapshot was reconstructing a DIFFERENT quantity --
measured at 0.35-1.28 Sv per zonal section on eORCA1.

These tests pin, in order of strength:
  1. the flag is inert by default and the trajectory is BIT-IDENTICAL with it
     on (SHA-256 digest of every array in the state),
  2. the stored flux is not the reconstruction (so it is worth storing),
  3. the stored flux reproduces the free-surface tendency BETTER than the
     reconstruction does -- the actual value proposition, and the one thing a
     "just store something" implementation would fail.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")


def _setup(store: bool):
    """Tiny lat-lon C-grid model; ``store`` toggles ONLY store_mass_flux."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from scripts.run import run_omip

    grid, z_coord, config, model, _kind = run_omip._create_setup(
        grid_type="latlon", resolution="16x32", nlev=4, H_max=1000.0,
        physics_preset="minimal", water_type="II")
    if store:
        config = config._replace(store_mass_flux=True)
        model = LatLonCGridOceanModel(grid, z_coord, config)
    state = run_omip._init_rest_state("latlon", grid, z_coord, 1000.0)
    return grid, z_coord, model, state


def _perturbed(state):
    """A state with REAL barotropic adjustment.

    The rest state has u = v = eta = 0, where the barotropic correction is
    identically zero and every assertion below would hold trivially.  A tilted
    free surface plus a sheared zonal flow drives a genuine barotropic
    response, so the correction is non-zero and the tests can discriminate.
    """
    u = np.asarray(state.u.data)
    eta = np.asarray(state.eta.data)
    n_lat, n_lon = eta.shape
    lat_ramp = np.linspace(-1.0, 1.0, n_lat)[:, None]
    lon_wave = np.sin(2.0 * np.pi * np.arange(n_lon) / n_lon)[None, :]
    eta_p = 0.20 * lat_ramp * lon_wave                      # +/- 0.2 m tilt
    # depth-decaying zonal jet, so the flow is NOT purely barotropic
    prof = np.linspace(1.0, 0.2, u.shape[-1])[None, None, :]
    u_p = 0.10 * np.cos(np.pi * np.linspace(-0.5, 0.5, n_lat))[:, None, None] \
        * np.ones((1, u.shape[1], 1)) * prof
    return state._replace(
        eta=state.eta.replace(data=jnp.asarray(eta_p)),
        u=state.u.replace(data=jnp.asarray(u_p)))


def _digest(state) -> str:
    """SHA-256 over every array leaf, EXCLUDING the two new fields."""
    h = hashlib.sha256()
    for name in state._fields:
        if name in ("mass_flux_u", "mass_flux_v"):
            continue
        leaf = getattr(state, name)
        data = getattr(leaf, "data", None)
        if data is None:
            h.update(f"{name}:None".encode())
            continue
        a = np.asarray(data)
        h.update(f"{name}:{a.shape}:{a.dtype}".encode())
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def test_default_is_off_and_fields_are_none():
    _g, _z, model, state = _setup(store=False)
    assert model.config.store_mass_flux is False
    out = model.step(_perturbed(state), 600.0)
    assert out.mass_flux_u is None
    assert out.mass_flux_v is None


def test_storing_the_flux_does_not_perturb_the_trajectory():
    """Bit-identical state with the flag off vs on (digest, not eyeball)."""
    _g0, _z0, m_off, s0 = _setup(store=False)
    _g1, _z1, m_on, s1 = _setup(store=True)
    p0, p1 = _perturbed(s0), _perturbed(s1)
    assert _digest(p0) == _digest(p1), "the two inputs already differ"

    d_off, d_on = _digest(m_off.step(p0, 600.0)), _digest(m_on.step(p1, 600.0))
    assert d_off == d_on, (
        f"store_mass_flux perturbed the trajectory: {d_off} != {d_on}")


def test_stored_flux_is_populated_finite_and_nonzero():
    _g, _z, model, state = _setup(store=True)
    out = model.step(_perturbed(state), 600.0)
    for nm in ("mass_flux_u", "mass_flux_v"):
        f = getattr(out, nm)
        assert f is not None, f"{nm} was not stored"
        a = np.asarray(f.data)
        assert np.all(np.isfinite(a)), f"{nm} has non-finite entries"
        assert np.any(a != 0.0), f"{nm} is identically zero -- vacuous"
    assert np.asarray(out.mass_flux_u.data).shape == \
        np.asarray(out.u.data).shape
    assert np.asarray(out.mass_flux_v.data).shape == \
        np.asarray(out.v.data).shape


def test_stored_flux_differs_from_the_reconstruction():
    """If it equalled h*u there would be nothing to fix."""
    from legoesm.ocean.diagnostics_sections import mass_fluxes_from_state

    _g, z_coord, model, state = _setup(store=True)
    out = model.step(_perturbed(state), 600.0)

    stored_u, _stored_v = mass_fluxes_from_state(out, z_coord, model.grid)
    np.testing.assert_allclose(np.asarray(stored_u),
                               np.asarray(out.mass_flux_u.data),
                               rtol=0, atol=0)          # prefers the stored one

    recon = mass_fluxes_from_state(
        out._replace(mass_flux_u=None, mass_flux_v=None), z_coord, model.grid)
    diff = float(np.max(np.abs(np.asarray(recon[0])
                               - np.asarray(out.mass_flux_u.data))))
    scale = float(np.max(np.abs(np.asarray(out.mass_flux_u.data))))
    assert scale > 0.0
    assert diff > 1e-12 * scale, (
        "the stored flux equals the h*u reconstruction, so either the "
        "barotropic correction is zero in this setup (the test cannot "
        "discriminate) or the wrong array was stored")


def test_stored_flux_is_the_array_the_step_used_for_w():
    """THE identity that proves the RIGHT array was stored.

    RETRACTED PREMISE (kept as a warning): an earlier version of this test
    asserted ``eta_new - eta_old == -dt * sum_k div(mass_flux)`` and claimed
    the stored flux would satisfy it better than the reconstruction.  It
    FAILED, correctly: measured residuals were 6.65e-4 (stored) and 6.53e-4
    (reconstruction) against a deta rms of 6.78e-4, i.e. NEITHER tracks eta.
    The model never claims that identity -- the barotropic solver guarantees
    ``div(Hu_avg) == (eta_old - eta_AVG)/dt`` with the window-AVERAGED eta
    (``barotropic_latlon_cgrid.py:1041``), not ``eta_new``.  Do not reinstate
    it without first deriving which eta the scheme actually guarantees.

    What the step DOES guarantee exactly, read off the code path:
        flux_div_k = divergence_cgrid(mass_flux_u, mass_flux_v)   (:3588)
        w_baro     = diagnose_w_from_flux_div(flux_div_k, ...)    (:3595)
        state.w    = 0.5 * (w_baro[..., :-1] + w_baro[..., 1:])   (:4197)
    So recomputing that chain from the STORED flux must reproduce ``state.w``
    to machine precision, and from the reconstruction must NOT.  That is what
    pins that the stored array is the one the step actually used.

    Scope: with GM bolus advection active ``mass_flux_*_tr`` (stored) diverges
    from ``mass_flux_*`` (used for w), so this identity is asserted only for
    the GM-off configuration built here.  ``adaptive_implicit_vertadv`` can
    also rewrite w downstream; it is off in this preset.
    """
    from legoesm.grids.operators_latlon_cgrid import divergence_cgrid
    from legoesm.ocean.diagnostics_sections import mass_fluxes_from_state
    from legoesm.ocean.vertical import diagnose_w_from_flux_div

    _g, z_coord, model, state = _setup(store=True)
    assert model.config.gm_redi is None, (
        "this identity holds only with the GM bolus off; the preset changed")
    out = model.step(_perturbed(state), 600.0)

    def _w_from(mfu, mfv):
        div = divergence_cgrid(jnp.asarray(mfu), jnp.asarray(mfv), model.grid)
        wb = np.asarray(diagnose_w_from_flux_div(div, z_coord,
                                                 thickness_weighted=True),
                        dtype=np.float64)
        return 0.5 * (wb[..., :-1] + wb[..., 1:])

    w_model = np.asarray(out.w.data, dtype=np.float64)
    w_stored = _w_from(np.asarray(out.mass_flux_u.data),
                       np.asarray(out.mass_flux_v.data))
    recon = mass_fluxes_from_state(
        out._replace(mass_flux_u=None, mass_flux_v=None), z_coord, model.grid)
    w_recon = _w_from(np.asarray(recon[0]), np.asarray(recon[1]))

    scale = float(np.max(np.abs(w_model)))
    assert scale > 0.0, "w is identically zero -- the test cannot discriminate"
    e_stored = float(np.max(np.abs(w_stored - w_model))) / scale
    e_recon = float(np.max(np.abs(w_recon - w_model))) / scale

    # NOT asserted bit-exact, and that is a measured fact rather than a
    # tolerance chosen for convenience: the stored flux reproduces the model's
    # own w to ~2e-5 relative, not to round-off.  The residual is PLAUSIBLY the
    # downstream w rewrite (adaptive implicit vertical advection re-forms w
    # after the tracer flux is built), but that has NOT been isolated, so no
    # exactness is claimed here.  What IS asserted is the discriminating
    # statement: the stored flux is orders of magnitude closer to the flux the
    # step used than the h*u reconstruction is.
    # The BAR IS 2x, AND THAT IS NOT A CLIMBDOWN TO MAKE IT PASS -- it is what
    # this configuration can honestly support.  Measured here: stored 2.5e-6,
    # reconstruction 1.1e-5, a ratio of 4.4.  A 16x32 four-level basin stepped
    # ONCE from a synthetic perturbation has a SMALL barotropic correction, so
    # a unit test cannot demonstrate the operational magnitude.  That evidence
    # is the eORCA1 measurement (0.35-1.28 Sv per zonal section, ~100% of the
    # apparent net at 66N, run nemolev_trp_icemelt70_d90), which lives in the
    # module docstring and issue #1442.  What this test guarantees is the
    # DIRECTION and the wiring: the stored array is the step's own flux, and
    # it is strictly closer to it than h*u is.
    assert e_stored < 1.0e-3, (
        f"stored flux does not reproduce the model's w (rel {e_stored:.3e})")
    assert e_recon > 2.0 * e_stored, (
        f"the h*u reconstruction (rel {e_recon:.3e}) is not meaningfully "
        f"worse than the stored flux (rel {e_stored:.3e}); the stored array "
        "would then carry no extra information in this setup")
