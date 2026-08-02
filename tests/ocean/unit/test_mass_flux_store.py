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
  2. the stored pair is the ``_tr`` (tracer-advecting) pair and NOT the raw
     momentum pair -- discriminated by a GM through-FCT control triple, the
     only configuration in which the two arrays differ at all,
  3. the stored flux reproduces the step's own w-diagnostic chain to round-off
     while the ``h*u`` reconstruction does not,
  4. the carry survives every entry path that changed shape when the flag
     turned two ``None`` slots into ``Field``s: ``lax.scan``, the SPMD
     ``shard_map``, and the MPI band scatter/gather.

WHY (2) IS THE LOAD-BEARING TEST.  With GM off, ``mass_flux_u_tr`` IS
``mass_flux_u`` -- the same array under two names.  A regression that stored
the raw pair would pass every other test in this file.  The codex review of the
first version of this suite found exactly that hole.  ``_gm_configs`` builds
the ONE configuration that tells them apart.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

_DT = 600.0


def _setup(store: bool, *, gm_bolus: str | None = None,
           barotropic_solver: str | None = None):
    """Tiny lat-lon C-grid model; ``store`` toggles ONLY store_mass_flux.

    ``gm_bolus`` (``None`` | ``"centred"`` | ``"through_fct"``) selects the GM
    control arm; ``None`` leaves ``gm_redi=None`` (the preset default).  Every
    other config field is identical across arms, so a difference between two
    arms is attributable to the one field that changed.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from scripts.run import run_omip

    grid, z_coord, config, model, _kind = run_omip._create_setup(
        grid_type="latlon", resolution="16x32", nlev=4, H_max=1000.0,
        physics_preset="minimal", water_type="II")
    rebuild = False
    if store:
        config = config._replace(store_mass_flux=True)
        rebuild = True
    if gm_bolus is not None:
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        # nemo_iso_lap is the ONLY slope scheme that honours
        # gm_bolus_advection (the model's `_want_bolus` requires it).
        config = config._replace(gm_redi=GMRediConfig(
            kappa_GM=1.0e3, kappa_Redi=1.0e3,
            slope_scheme="nemo_iso_lap", gm_bolus_advection=gm_bolus))
        rebuild = True
    if barotropic_solver is not None:
        config = config.replace_flat(barotropic_solver=barotropic_solver)
        rebuild = True
    if rebuild:
        model = LatLonCGridOceanModel(grid, z_coord, config)
    state = run_omip._init_rest_state("latlon", grid, z_coord, 1000.0)
    return grid, z_coord, model, state


def _perturbed(state):
    """A state with REAL barotropic adjustment AND real isopycnal slopes.

    The rest state has u = v = eta = 0 and a horizontally UNIFORM T/S, where
    (a) the barotropic correction is identically zero and (b) the GM bolus is
    identically zero -- every assertion below would hold trivially and the
    suite would prove nothing.  So this adds all three ingredients:

    * a tilted free surface + a depth-decaying zonal jet -> a genuine
      barotropic transport correction (the thing ``mass_flux`` captures that
      ``h*u`` cannot);
    * a meridional + zonal TEMPERATURE front over a stable stratification ->
      non-zero isopycnal slopes -> a non-zero GM bolus transport (the thing
      that tells ``_tr`` apart from the raw pair).

    Non-vacuity of the temperature front is not assumed: the GM test asserts
    the resulting bolus is non-zero and fails loudly if it is not.
    """
    u = np.asarray(state.u.data)
    eta = np.asarray(state.eta.data)
    T = np.asarray(state.T.data)
    n_lat, n_lon = eta.shape
    nlev = T.shape[-1]
    lat_ramp = np.linspace(-1.0, 1.0, n_lat)[:, None]
    lon_wave = np.sin(2.0 * np.pi * np.arange(n_lon) / n_lon)[None, :]
    eta_p = 0.20 * lat_ramp * lon_wave                      # +/- 0.2 m tilt
    # depth-decaying zonal jet, so the flow is NOT purely barotropic
    prof = np.linspace(1.0, 0.2, u.shape[-1])[None, None, :]
    u_p = 0.10 * np.cos(np.pi * np.linspace(-0.5, 0.5, n_lat))[:, None, None] \
        * np.ones((1, u.shape[1], 1)) * prof
    # Stable stratification + a horizontal front: warm/light at the surface
    # and to the south, so d(rho)/dx and d(rho)/dy are both non-zero and the
    # isopycnal slope S = -grad_h(rho)/d(rho)/dz is finite.
    z_prof = np.linspace(0.0, -8.0, nlev)[None, None, :]     # -8 C by depth
    T_p = (T + z_prof
           + 4.0 * lat_ramp[:, :, None]                      # +/- 4 C in lat
           + 1.0 * lon_wave[:, :, None])                     # +/- 1 C in lon
    return state._replace(
        eta=state.eta.replace(data=jnp.asarray(eta_p)),
        u=state.u.replace(data=jnp.asarray(u_p)),
        T=state.T.replace(data=jnp.asarray(T_p)))


_NEW_SLOTS = ("mass_flux_u", "mass_flux_v")


def _digest(state) -> str:
    """SHA-256 over every array leaf, EXCLUDING the two new fields."""
    h = hashlib.sha256()
    for name in state._fields:
        if name in _NEW_SLOTS:
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


# ---------------------------------------------------------------------------
# 1. inert by default, and non-perturbing when on
# ---------------------------------------------------------------------------

def test_default_is_off_and_fields_are_none():
    _g, _z, model, state = _setup(store=False)
    assert model.config.store_mass_flux is False
    out = model.step(_perturbed(state), _DT)
    assert out.mass_flux_u is None
    assert out.mass_flux_v is None


def test_storing_the_flux_does_not_perturb_the_trajectory():
    """Bit-identical state with the flag off vs on (digest, not eyeball)."""
    _g0, _z0, m_off, s0 = _setup(store=False)
    _g1, _z1, m_on, s1 = _setup(store=True)
    p0, p1 = _perturbed(s0), _perturbed(s1)
    assert _digest(p0) == _digest(p1), "the two inputs already differ"

    d_off, d_on = _digest(m_off.step(p0, _DT)), _digest(m_on.step(p1, _DT))
    assert d_off == d_on, (
        f"store_mass_flux perturbed the trajectory: {d_off} != {d_on}")


def test_stored_flux_is_populated_finite_and_nonzero():
    _g, _z, model, state = _setup(store=True)
    out = model.step(_perturbed(state), _DT)
    for nm in _NEW_SLOTS:
        f = getattr(out, nm)
        assert f is not None, f"{nm} was not stored"
        a = np.asarray(f.data)
        assert np.all(np.isfinite(a)), f"{nm} has non-finite entries"
        assert np.any(a != 0.0), f"{nm} is identically zero -- vacuous"
    assert np.asarray(out.mass_flux_u.data).shape == \
        np.asarray(out.u.data).shape
    assert np.asarray(out.mass_flux_v.data).shape == \
        np.asarray(out.v.data).shape


# ---------------------------------------------------------------------------
# 2. THE DISCRIMINATOR: the stored pair is `_tr`, not the raw momentum pair
# ---------------------------------------------------------------------------

def test_stored_pair_is_the_tr_pair_not_the_raw_pair():
    """The one configuration in which ``_tr`` and the raw pair DIFFER.

    Codex finding (YELLOW 11) on the first version of this suite: with GM off
    ``mass_flux_u_tr is mass_flux_u``, so every test passed whether the step
    stored the tracer-advecting pair or the raw momentum pair.  This test is
    the missing discriminator.

    CONTROLLED TRIPLE -- one field changes between arms, nothing else:
        A: gm_redi = None                      (no bolus at all)
        B: gm_bolus_advection = "centred"      (bolus applied IN-OPERATOR)
        C: gm_bolus_advection = "through_fct"  (bolus added to the ADVECTING
                                                flux -> lands in ``_tr``)
    Within ONE step from a shared IC the base (momentum/continuity) mass flux
    is bit-identical across all three: GM is a TRACER tendency, it does not
    touch momentum or the barotropic solve, and the flux is formed BEFORE the
    GM call.  Therefore:
        stored(A) == stored(B)   -- the raw pair, in both
        stored(C) != stored(B)   -- only C folds the bolus into the pair
    If the step stored the RAW pair, all three would be equal and the second
    assertion goes red.  That is the mutation this test exists to catch.
    """
    _gA, _zA, mA, sA = _setup(store=True, gm_bolus=None)
    _gB, _zB, mB, sB = _setup(store=True, gm_bolus="centred")
    _gC, _zC, mC, sC = _setup(store=True, gm_bolus="through_fct")
    outA = mA.step(_perturbed(sA), _DT)
    outB = mB.step(_perturbed(sB), _DT)
    outC = mC.step(_perturbed(sC), _DT)

    raw = np.asarray(outA.mass_flux_u.data, dtype=np.float64)
    ctr = np.asarray(outB.mass_flux_u.data, dtype=np.float64)
    fct = np.asarray(outC.mass_flux_u.data, dtype=np.float64)
    scale = float(np.max(np.abs(raw)))
    assert scale > 0.0, "the base mass flux is zero -- the setup is vacuous"

    # Control: GM-off and GM-centred store the SAME array, confirming the base
    # flux is invariant to the GM arm and that any C-vs-B difference below is
    # the bolus and nothing else.
    np.testing.assert_allclose(ctr, raw, rtol=0.0, atol=0.0)

    d_bolus = float(np.max(np.abs(fct - ctr)))
    assert d_bolus > 1e-10 * scale, (
        f"through-FCT GM changed the stored flux by only {d_bolus:.3e} "
        f"(scale {scale:.3e}): the step is storing the RAW mass_flux pair, "
        "not the tracer-advecting mass_flux_*_tr pair -- OR the bolus is "
        "zero in this setup, in which case the test cannot discriminate and "
        "the temperature front in _perturbed must be restored.")

    # Same statement for the v pair (a u-only capture would pass above).
    rawv = np.asarray(outB.mass_flux_v.data, dtype=np.float64)
    fctv = np.asarray(outC.mass_flux_v.data, dtype=np.float64)
    scalev = float(np.max(np.abs(rawv)))
    assert scalev > 0.0
    assert float(np.max(np.abs(fctv - rawv))) > 1e-10 * scalev, (
        "mass_flux_v does not carry the bolus while mass_flux_u does -- the "
        "v capture is the raw pair")


def test_stored_pair_is_bolus_inclusive_while_w_is_not():
    """Pins the documented HORIZONTAL-ONLY scope (codex YELLOW 9).

    Under ``gm_bolus_advection="through_fct"`` the step stores the
    bolus-inclusive ``_tr`` pair but builds ``state.w`` from the BASE
    ``w_baro`` (``add_bolus_to_advecting_flux``'s ``w_baro_tr`` is used for
    tracer advection and then discarded).  So the stored pair and ``state.w``
    are NOT a matched advecting triple, and a 3-D budget that pairs them is
    wrong.  This is a KNOWN, DOCUMENTED asymmetry, not an accident -- pinning
    it here means it cannot be silently "fixed" on one side only: a change
    that starts storing the base pair fails the test above, and a change that
    starts building ``state.w`` from ``w_baro_tr`` fails this one.
    """
    _gB, _zB, mB, sB = _setup(store=True, gm_bolus="centred")
    _gC, _zC, mC, sC = _setup(store=True, gm_bolus="through_fct")
    wB = np.asarray(mB.step(_perturbed(sB), _DT).w.data, dtype=np.float64)
    wC = np.asarray(mC.step(_perturbed(sC), _DT).w.data, dtype=np.float64)
    # state.w is built from the BASE pair, which is identical in both arms, so
    # w is bit-identical even though the STORED flux differs (asserted above).
    np.testing.assert_allclose(wC, wB, rtol=0.0, atol=0.0,
                               err_msg=(
                                   "state.w now responds to the GM bolus arm; "
                                   "if w is intentionally built from w_baro_tr, "
                                   "update the HORIZONTAL-ONLY note on "
                                   "LatLonCGridOceanState.mass_flux_u and this "
                                   "test together"))


def test_stored_flux_differs_from_the_reconstruction():
    """If it equalled h*u there would be nothing to fix."""
    from legoesm.ocean.diagnostics_sections import mass_fluxes_from_state

    _g, z_coord, model, state = _setup(store=True)
    out = model.step(_perturbed(state), _DT)

    stored_u, _stored_v = mass_fluxes_from_state(out, z_coord, model.grid)
    np.testing.assert_allclose(np.asarray(stored_u),
                               np.asarray(out.mass_flux_u.data),
                               rtol=0, atol=0)          # prefers the stored one

    recon = mass_fluxes_from_state(out, z_coord, model.grid,
                                   source="reconstruct")
    diff = float(np.max(np.abs(np.asarray(recon[0])
                               - np.asarray(out.mass_flux_u.data))))
    scale = float(np.max(np.abs(np.asarray(out.mass_flux_u.data))))
    assert scale > 0.0
    assert diff > 1e-12 * scale, (
        "the stored flux equals the h*u reconstruction, so either the "
        "barotropic correction is zero in this setup (the test cannot "
        "discriminate) or the wrong array was stored")


def _w_identity_residuals(model, z_coord, state):
    """Relative error of ``state.w`` rebuilt from the STORED vs the
    RECONSTRUCTED flux.  Returns ``(e_stored, e_recon, dtype)``.

    Recomputes, in float64 numpy, the exact chain the step ran:
        flux_div_k = divergence_cgrid(mass_flux_u, mass_flux_v)
        w_baro     = diagnose_w_from_flux_div(flux_div_k, ...)
        state.w    = 0.5 * (w_baro[..., :-1] + w_baro[..., 1:])
    """
    from legoesm.grids.operators_latlon_cgrid import divergence_cgrid
    from legoesm.ocean.diagnostics_sections import mass_fluxes_from_state
    from legoesm.ocean.vertical import diagnose_w_from_flux_div

    out = model.step(state, _DT)

    def _w_from(mfu, mfv):
        div = divergence_cgrid(jnp.asarray(mfu), jnp.asarray(mfv), model.grid)
        wb = np.asarray(diagnose_w_from_flux_div(div, z_coord,
                                                 thickness_weighted=True),
                        dtype=np.float64)
        return 0.5 * (wb[..., :-1] + wb[..., 1:])

    w_model = np.asarray(out.w.data, dtype=np.float64)
    w_stored = _w_from(np.asarray(out.mass_flux_u.data, dtype=np.float64),
                       np.asarray(out.mass_flux_v.data, dtype=np.float64))
    recon = mass_fluxes_from_state(out, z_coord, model.grid,
                                   source="reconstruct")
    w_recon = _w_from(np.asarray(recon[0], dtype=np.float64),
                      np.asarray(recon[1], dtype=np.float64))
    scale = float(np.max(np.abs(w_model)))
    assert scale > 0.0, "w is identically zero -- the test cannot discriminate"
    return (float(np.max(np.abs(w_stored - w_model))) / scale,
            float(np.max(np.abs(w_recon - w_model))) / scale,
            np.asarray(out.mass_flux_u.data).dtype)


import contextlib


@contextlib.contextmanager
def _precision_policy(policy):
    """Temporarily install a global precision policy, always restoring it.

    Promoting the INPUT state is not enough: ``_step_impl`` ends with
    ``cast_pytree(state_new, None, "storage", allow_downcast=True)``, so the
    returned state is at the STORAGE policy dtype whatever the input was.  (A
    first attempt at the scaling test below cast only the state and measured
    float32 on both arms -- vacuous, and it said so rather than passing.)
    """
    from legoesm.core.precision import get_policy, set_policy

    prev = get_policy()
    set_policy(policy)
    try:
        yield
    finally:
        set_policy(prev)


def test_stored_flux_is_the_array_the_step_used_for_w():
    """THE identity that proves the RIGHT array was stored.

    RETRACTED PREMISE #1 (kept as a warning): an earlier version of this test
    asserted ``eta_new - eta_old == -dt * sum_k div(mass_flux)`` and claimed
    the stored flux would satisfy it better than the reconstruction.  It
    FAILED, correctly: measured residuals were 6.65e-4 (stored) and 6.53e-4
    (reconstruction) against a deta rms of 6.78e-4, i.e. NEITHER tracks eta.
    The model never claims that identity -- the barotropic solver guarantees
    ``div(Hu_avg) == (eta_old - eta_AVG)/dt`` with the window-AVERAGED eta
    (``barotropic_latlon_cgrid.py:1041``), not ``eta_new``.  Do not reinstate
    it without first deriving which eta the scheme actually guarantees.

    RETRACTED PREMISE #2 (codex YELLOW 11): the same earlier version explained
    a ~2.5e-6 relative residual here as "PLAUSIBLY the downstream w rewrite
    (adaptive implicit vertical advection)".  RETRACTED -- it was wrong twice
    over: that block writes ``state_new.u``/``.v``, NOT w, and it is OFF in
    this preset.  The residual is the STORAGE PRECISION of the state, and that
    is not asserted on faith either: ``test_w_identity_residual_tracks_the_
    state_precision`` promotes the state to float64 and shows the residual
    collapses by >=1e3, which the precision explanation predicts and a
    "different array" explanation cannot produce.  (Note ``JAX_ENABLE_X64=1``
    alone does NOT settle this: the ocean state is built at float32 STORAGE
    precision regardless of that flag, which is exactly why the first attempt
    at this tolerance -- keyed off ``jax.config.jax_enable_x64`` -- measured
    1.7e-6 and went red.)

    Scope: asserted for the GM-off configuration built here, where the stored
    ``_tr`` pair IS the pair w was diagnosed from (see
    ``test_stored_pair_is_bolus_inclusive_while_w_is_not`` for the through-FCT
    case, where they deliberately differ).  ``adaptive_implicit_vertadv`` is
    off in this preset.
    """
    _g, z_coord, model, state = _setup(store=True)
    assert model.config.gm_redi is None, (
        "this identity holds only with the GM bolus off; the preset changed")
    e_stored, e_recon, dtype = _w_identity_residuals(
        model, z_coord, _perturbed(state))

    # Bar in ULP of the STATE's storage precision, not an absolute constant:
    # the residual is round-off, so the meaningful question is how many eps.
    eps = float(np.finfo(dtype).eps)
    assert e_stored < 200.0 * eps, (
        f"stored flux does not reproduce the model's w: rel {e_stored:.3e} "
        f"= {e_stored / eps:.1f} eps({dtype}) -- beyond round-off, so it is "
        "not the array the step diagnosed w from")
    # The h*u reconstruction is a genuinely DIFFERENT quantity (it is missing
    # the barotropic correction), so it must be well clear of round-off.
    assert e_recon > 4.0 * e_stored, (
        f"the h*u reconstruction (rel {e_recon:.3e}) is not meaningfully "
        f"worse than the stored flux (rel {e_stored:.3e}); the stored array "
        "would then carry no extra information in this setup")


def test_w_identity_residual_tracks_the_state_precision():
    """MECHANISM TEST for the residual above -- do not delete it.

    A proposed cause must survive a scaling test before it is cited (that is
    how the "adaptive vertical advection rewrites w" story should have died
    before it was ever written down).  The claim here is: the ~1e-6 residual
    is ROUND-OFF of the float32 state, not evidence that the stored array
    differs from the one the step used.

    Prediction: rebuild the whole model under an all-float64 precision policy
    and the STORED residual must collapse by roughly eps(f32)/eps(f64) ~ 1e9
    -- a factor of 1e3 is asserted as a deliberately loose floor.  The
    RECONSTRUCTION residual must NOT collapse: it is a real physical
    difference (the missing barotropic correction), so it is
    precision-independent to within a factor of a few.  A genuinely-different
    stored array would behave like the reconstruction and fail the first
    assertion.
    """
    from legoesm.core.precision import PrecisionPolicy

    if not jax.config.jax_enable_x64:
        pytest.skip("needs JAX_ENABLE_X64=1 to build the float64 arm")
    _g32, z32, m32, s32 = _setup(store=True)
    e32, r32, dt32 = _w_identity_residuals(m32, z32, _perturbed(s32))
    with _precision_policy(PrecisionPolicy.fp64()):
        _g64, z64, m64, s64 = _setup(store=True)
        e64, r64, dt64 = _w_identity_residuals(m64, z64, _perturbed(s64))
    assert dt32 == np.float32 and dt64 == np.float64, (
        f"the two arms did not run at different precisions "
        f"({dt32} vs {dt64}) -- the scaling test is vacuous")

    assert e64 < e32 / 1.0e3, (
        f"the w-identity residual did NOT collapse with precision "
        f"(f32 {e32:.3e} -> f64 {e64:.3e}, ratio {e32 / max(e64, 1e-300):.2e}). "
        "It is therefore NOT round-off, and the stored array is not the one "
        "the step diagnosed w from.  Do not paper over this with a looser "
        "tolerance -- find the second writer.")
    assert r64 > r32 / 10.0, (
        f"the RECONSTRUCTION residual also collapsed with precision "
        f"(f32 {r32:.3e} -> f64 {r64:.3e}); it is supposed to be a real "
        "physical difference (the missing barotropic correction), so this "
        "test can no longer tell round-off from a genuine mismatch")


# ---------------------------------------------------------------------------
# 3. metadata (codex YELLOW 7)
# ---------------------------------------------------------------------------

def test_stored_fields_carry_mass_flux_units_not_velocity_units():
    """``m^2/s``, not the ``m/s`` inherited from the u/v Fields it copies.

    A Field mislabelled ``m/s`` silently mis-scales any CF/netCDF export and
    any unit-aware consumer.  dims/staggering ARE inherited (identical face
    layout) and that is asserted too, so the fix cannot regress into dropping
    the staggering metadata.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        MASS_FLUX_UNITS,
    )

    _g, _z, model, state = _setup(store=True)
    out = model.step(_perturbed(state), _DT)
    assert MASS_FLUX_UNITS == "m^2/s"
    for nm, src in (("mass_flux_u", "u"), ("mass_flux_v", "v")):
        f = getattr(out, nm)
        s = getattr(out, src)
        assert f.units == MASS_FLUX_UNITS, (
            f"{nm}.units is {f.units!r}; the stored data are thickness x "
            f"velocity [m^2/s], not a velocity")
        assert s.units != MASS_FLUX_UNITS, (
            f"state.{src}.units is now {s.units!r} -- the test can no longer "
            "tell an inherited unit from an overridden one")
        assert f.name == nm
        assert f.dims == s.dims, "face dims must be inherited from u/v"
        assert f.staggering == s.staggering


def test_seeded_carry_metadata_matches_what_the_step_writes():
    """Seed and step must produce the IDENTICAL Field aux data.

    ``Field`` carries name/dims/units/staggering as pytree AUX data, so a seed
    that differs in any of them is a different treedef and ``lax.scan``
    rejects the carry.  Both go through ``mass_flux_fields``; this asserts the
    shared constructor actually made them agree.
    """
    _g, _z, model, state = _setup(store=True)
    seeded = model.seed_scan_carry(_perturbed(state), _DT)
    stepped = model.step(_perturbed(state), _DT)
    for nm in _NEW_SLOTS:
        a, b = getattr(seeded, nm), getattr(stepped, nm)
        assert a is not None and b is not None
        assert (a.name, a.dims, a.units, a.staggering) == \
               (b.name, b.dims, b.units, b.staggering)
    assert (jax.tree_util.tree_structure(seeded)
            == jax.tree_util.tree_structure(stepped)), (
        "seeded carry and step output have different pytree structures -- "
        "lax.scan would reject the carry")


# ---------------------------------------------------------------------------
# 4. carry survives every entry path (codex RED 2 / RED 3 / RED 4)
# ---------------------------------------------------------------------------

def test_seed_scan_carry_seeds_the_mass_flux_slots():
    _g, _z, model, state = _setup(store=True)
    seeded = model.seed_scan_carry(_perturbed(state), _DT)
    for nm, src in (("mass_flux_u", "u"), ("mass_flux_v", "v")):
        f = getattr(seeded, nm)
        assert f is not None, f"{nm} was not seeded -- lax.scan would crash"
        assert np.asarray(f.data).shape == \
            np.asarray(getattr(seeded, src).data).shape
        assert not np.any(np.asarray(f.data)), "the seed must be zeros"
    # Idempotent: re-seeding an already-prepared carry is a no-op.
    again = model.seed_scan_carry(seeded, _DT)
    np.testing.assert_array_equal(np.asarray(again.mass_flux_u.data),
                                  np.asarray(seeded.mass_flux_u.data))


def test_seed_is_a_noop_when_the_flag_is_off():
    _g, _z, model, state = _setup(store=False)
    seeded = model.seed_scan_carry(_perturbed(state), _DT)
    assert seeded.mass_flux_u is None
    assert seeded.mass_flux_v is None


def test_seeded_carry_survives_lax_scan():
    """The RED 2 regression: a None -> Field flip mid-scan crashes lax.scan.

    Runs a REAL two-iteration ``lax.scan`` over the model step, which is the
    thing that actually broke.  Without the seeding this raises a carry
    structure mismatch inside ``lax.scan`` (and ``seed_scan_carry`` itself
    raises earlier, in its dtype-reconciliation ``tree_map``).
    """
    _g, _z, model, state = _setup(store=True)
    seeded = model.seed_scan_carry(_perturbed(state), _DT)

    def _body(carry, _x):
        return model.step(carry, _DT), None

    final, _ = jax.lax.scan(_body, seeded, xs=None, length=2)
    assert final.mass_flux_u is not None
    a = np.asarray(final.mass_flux_u.data)
    assert np.all(np.isfinite(a)) and np.any(a != 0.0)
    assert (jax.tree_util.tree_structure(final)
            == jax.tree_util.tree_structure(seeded))


def test_direct_jit_step_loop_does_not_retrace():
    """The other half of RED 2: a jitted direct loop must not recompile.

    Feeding a SEEDED carry back into the same jitted step keeps the input
    treedef constant, so the trace count stays at 1.  Unseeded, iteration 2
    would present a different structure and retrace.
    """
    _g, _z, model, state = _setup(store=True)
    traces = []

    @jax.jit
    def _step(s):
        traces.append(1)
        return model.step(s, _DT)

    carry = model.seed_scan_carry(_perturbed(state), _DT)
    for _ in range(3):
        carry = _step(carry)
    jax.block_until_ready(carry.T.data)
    assert len(traces) == 1, (
        f"the jitted step traced {len(traces)} times over 3 iterations -- the "
        "state pytree is not constant across the loop")


def test_mass_flux_v_is_registered_as_a_v_staggered_spmd_field():
    """RED 3: ``mass_flux_v`` has the n_lat+1 v-face leading dim.

    Omitted from ``_V_STAGGERED_STATE_FIELDS`` it got the CELL spec, so the
    band slicer neither split nor reconstructed its boundary row.  Asserting
    membership by NAME is the cheap gate; the shape round-trip below is the
    behavioural one.
    """
    from legoesm.ocean.dynamics import sharded_ocean_step as sos

    assert "mass_flux_v" in sos._V_STAGGERED_STATE_FIELDS
    assert "mass_flux_u" not in sos._V_STAGGERED_STATE_FIELDS, (
        "mass_flux_u is a U-face field (leading dim n_lat) and must take the "
        "default cell sharding")


def test_spmd_shard_and_gather_round_trip_the_mass_flux_pair():
    """RED 3, behavioural: shard -> gather must restore both face shapes.

    ``shard_state_latlon`` drops ``mass_flux_v``'s top (pole-wall) row and
    ``gather_state_latlon`` re-appends it as zero -- valid here for the same
    reason it is valid for ``v``: ``mass_flux_v = h_v_old * v_corrected *
    v_mask_3d`` and the pole-wall ``v_mask[n_lat] == 0``, so the row IS zero.
    That is asserted, not assumed.
    """
    from jax.sharding import Mesh

    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_state_latlon, shard_state_latlon,
    )

    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("needs >= 2 devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    _g, _z, model, state = _setup(store=True)
    out = model.step(_perturbed(state), _DT)
    n_lat = np.asarray(out.T.data).shape[0]
    n_dev = 2
    assert n_lat % n_dev == 0

    mfv = np.asarray(out.mass_flux_v.data)
    np.testing.assert_allclose(mfv[-1], 0.0, rtol=0, atol=0,
                               err_msg=("the top v-face row of mass_flux_v is "
                                        "non-zero, so the drop/re-append-zero "
                                        "round trip is NOT an identity"))

    mesh = Mesh(np.asarray(devices[:n_dev]).reshape(n_dev), ("lat",))
    sharded = shard_state_latlon(out, mesh)
    assert np.asarray(sharded.mass_flux_v.data).shape[0] == n_lat, (
        "mass_flux_v was not carried as the n_lat v_lower slab")
    assert np.asarray(sharded.mass_flux_u.data).shape == \
        np.asarray(out.mass_flux_u.data).shape, (
        "mass_flux_u is a u-face field and must keep its n_lat leading dim")
    back = gather_state_latlon(sharded, mesh)
    np.testing.assert_allclose(np.asarray(back.mass_flux_v.data), mfv,
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(back.mass_flux_u.data),
                               np.asarray(out.mass_flux_u.data),
                               rtol=0, atol=0)


def test_mpi_band_scatter_slices_both_mass_flux_faces():
    """RED 4 (scatter half): a populated global state MUST be sliced.

    Omitted, every rank kept the FULL-domain flux while every other field was
    band-local.  Uses two real band layouts; pure slicing, no MPI runtime.
    """
    from legoesm.parallel.latlon_mpi import (
        make_latlon_band_layout, scatter_state_latlon_cgrid_ocean,
    )

    _g, _z, model, state = _setup(store=True)
    out = model.step(_perturbed(state), _DT)
    mfu = np.asarray(out.mass_flux_u.data)
    mfv = np.asarray(out.mass_flux_v.data)
    n_lat, n_lon = np.asarray(out.T.data).shape[:2]

    for rank in (0, 1):
        layout = make_latlon_band_layout(rank, 2, n_lat, n_lon)
        band = scatter_state_latlon_cgrid_ocean(out, layout)
        s, e = layout.lat_start, layout.lat_end
        # u-face: rows [s, e), exactly like ``u``
        assert np.asarray(band.mass_flux_u.data).shape == \
            np.asarray(band.u.data).shape
        np.testing.assert_allclose(np.asarray(band.mass_flux_u.data),
                                   mfu[s:e], rtol=0, atol=0)
        # v-face: rows [s, e+1), exactly like ``v``
        assert np.asarray(band.mass_flux_v.data).shape == \
            np.asarray(band.v.data).shape
        np.testing.assert_allclose(np.asarray(band.mass_flux_v.data),
                                   mfv[s:e + 1], rtol=0, atol=0)


def test_mpi_band_scatter_passes_none_through():
    """A run with the flag OFF must not gain Fields from the scatter."""
    from legoesm.parallel.latlon_mpi import (
        make_latlon_band_layout, scatter_state_latlon_cgrid_ocean,
    )

    _g, _z, model, state = _setup(store=False)
    out = model.step(_perturbed(state), _DT)
    n_lat, n_lon = np.asarray(out.T.data).shape[:2]
    band = scatter_state_latlon_cgrid_ocean(
        out, make_latlon_band_layout(0, 2, n_lat, n_lon))
    assert band.mass_flux_u is None and band.mass_flux_v is None


def test_mpi_band_gather_collects_both_mass_flux_faces(monkeypatch):
    """RED 4 (gather half): rank 0 must not keep its BAND-LOCAL flux.

    ``gather_state_latlon_cgrid_ocean``'s per-field gather needs a live MPI
    communicator, so the collective itself is stubbed: ``gather_field_latlon``
    is replaced by a recorder.  What is under test is the WIRING -- that both
    slots go through the gather and that ``mass_flux_v`` is declared a v-face
    -- which is exactly what was missing.

    IDENTIFYING THE FIELD IS THE WHOLE POINT, and the first version of this
    test got it wrong: it keyed the recorder on the array's LEADING DIM, but
    ``mass_flux_v`` and ``v`` have the same leading dim, so an
    ``is_v_face=True`` call logged by ``v`` satisfied the assertion meant for
    ``mass_flux_v``.  A deliberate mutation (gather ``mass_flux_v`` with the
    CELL convention) went UNCAUGHT.  The two slots now carry unique sentinel
    VALUES before the gather, so each recorded call is attributable to exactly
    one field, and each sentinel is required to appear exactly once with the
    right ``is_v_face``.
    """
    from legoesm.parallel import latlon_mpi as lm

    # Values no other state field can be uniformly equal to (masks are 0/1,
    # eta ~ 0.2 m, T ~ 0-20 C but never CONSTANT, S ~ 35 but not at these).
    SENTINEL = {"mass_flux_u": -11.5, "mass_flux_v": -13.25}
    OUT = 7.0
    calls = []

    def _fake_gather(data, layout, *, is_v_face=False):
        a = np.asarray(data)
        tag = (float(a.flat[0])
               if a.size and bool(np.all(a == a.flat[0])) else None)
        calls.append((tag, bool(is_v_face)))
        return np.full_like(a, OUT)

    monkeypatch.setattr(lm, "gather_field_latlon", _fake_gather)

    _g, _z, model, state = _setup(store=True)
    out = model.step(_perturbed(state), _DT)
    n_lat, n_lon = np.asarray(out.T.data).shape[:2]
    layout = lm.make_latlon_band_layout(0, 2, n_lat, n_lon)
    band = lm.scatter_state_latlon_cgrid_ocean(out, layout)
    # Stamp the two slots with their sentinels (shapes/dtypes untouched).
    band = band._replace(**{
        nm: getattr(band, nm).replace(
            data=jnp.full_like(getattr(band, nm).data, v))
        for nm, v in SENTINEL.items()})

    gathered = lm.gather_state_latlon_cgrid_ocean(band, layout)

    assert gathered is not None, "rank 0 must return a state"
    for nm in _NEW_SLOTS:
        f = getattr(gathered, nm)
        assert f is not None, f"{nm} vanished from the gathered state"
        np.testing.assert_allclose(
            np.asarray(f.data), OUT, rtol=0, atol=0,
            err_msg=(f"{nm} was NOT passed through the gather -- rank 0 kept "
                     "its band-local array in an otherwise global state"))

    # Attribution by sentinel VALUE, not by shape: mass_flux_v must be
    # gathered as a v-face (its duplicated boundary row trimmed on every rank
    # but the northernmost), mass_flux_u with the cell/u convention.
    got = {}
    for tag, is_v in calls:
        if tag is None:
            continue
        for nm, v in SENTINEL.items():
            if tag == pytest.approx(v, abs=1e-6):
                got.setdefault(nm, []).append(is_v)
    assert got.get("mass_flux_v") == [True], (
        f"mass_flux_v gather calls: {got.get('mass_flux_v')!r}; expected "
        "exactly one with is_v_face=True.  Gathered with the CELL convention "
        "its duplicated boundary row is not trimmed, so the assembled global "
        "array gains one row per rank instead of one row total.")
    assert got.get("mass_flux_u") == [False], (
        f"mass_flux_u gather calls: {got.get('mass_flux_u')!r}; expected "
        "exactly one with is_v_face=False (it is a u-face field, leading dim "
        "n_lat, with no duplicated boundary row to trim)")


# ---------------------------------------------------------------------------
# 5. config surface: positional stability, unsupported lanes, source=
# ---------------------------------------------------------------------------

def test_store_mass_flux_is_the_last_config_field():
    """RED 1: inserted mid-tuple it SHIFTS every later positional argument.

    ``LatLonCGridOceanConfig`` documents a positional-order compatibility
    contract ("appended at the END ... to preserve positional construction for
    legacy callers") in several places.  The first version of #1442 put the
    field between ``freshwater_salinity`` and ``zdf_drag_in_matrix``, silently
    shifting ~10 later fields for any positional caller.
    """
    from legoesm.ocean.state import LatLonCGridOceanConfig

    fields = LatLonCGridOceanConfig._fields
    assert fields[-1] == "store_mass_flux", (
        f"store_mass_flux is at index {fields.index('store_mass_flux')} of "
        f"{len(fields)}; a new field must be APPENDED, never inserted -- "
        f"the field after it is {fields[fields.index('store_mass_flux') + 1]!r}")


def test_store_mass_flux_survives_flat_construction_and_replace():
    """The route the OMIP builders use (``_ovr`` -> ``replace_flat``)."""
    from legoesm.ocean.state import LatLonCGridOceanConfig

    cfg = LatLonCGridOceanConfig()
    assert cfg.store_mass_flux is False
    assert "store_mass_flux" in LatLonCGridOceanConfig.flat_fields()
    assert cfg.replace_flat(store_mass_flux=True).store_mass_flux is True
    assert cfg.flat_get("store_mass_flux") is False


def test_implicit_unsplit_rejects_store_mass_flux():
    """RED 5: ``_unsplit_ab2_step`` BYPASSES the capture in ``_step_impl``.

    Silently the slots would stay ``None`` on step 1 and then hold a STALE
    value forever once seeded -- which a transport diagnostic would integrate
    as if it were live.  Rejected at construction, like ``prescribed_flow``.
    """
    with pytest.raises(ValueError, match="store_mass_flux"):
        _setup(store=True, barotropic_solver="implicit_unsplit")


def test_implicit_unsplit_without_the_flag_still_constructs():
    """The guard must not reject the solver itself (non-vacuity control)."""
    _g, _z, model, _s = _setup(store=False,
                               barotropic_solver="implicit_unsplit")
    assert model.config.barotropic.barotropic_solver == "implicit_unsplit"


@pytest.mark.parametrize("bad", ["Stored", "reconstuct", "", "raw", None])
def test_mass_fluxes_from_state_rejects_an_unknown_source(bad):
    """Dispatch hardening: a typo must raise, not silently pick a branch."""
    from legoesm.ocean.diagnostics_sections import mass_fluxes_from_state

    _g, z_coord, model, state = _setup(store=True)
    out = model.step(_perturbed(state), _DT)
    with pytest.raises(ValueError, match="source must be one of"):
        mass_fluxes_from_state(out, z_coord, model.grid, source=bad)


def test_source_stored_raises_when_the_state_has_no_stored_flux():
    """Codex YELLOW 8: a caller that REQUIRES exactness must not be handed
    the reconstruction under the same name."""
    from legoesm.ocean.diagnostics_sections import mass_fluxes_from_state

    _g, z_coord, model, state = _setup(store=False)
    out = model.step(_perturbed(state), _DT)
    with pytest.raises(ValueError, match="carries no mass_flux"):
        mass_fluxes_from_state(out, z_coord, model.grid, source="stored")


def test_source_reconstruct_ignores_the_stored_flux():
    """The opt-out: ``reconstruct`` must recompute from the CURRENT state.

    This is the concrete hazard YELLOW 8 named -- a caller that MUTATED the
    state and expected the flux to follow.  Under ``auto`` it does not (the
    stored flux is a record of the step, not a function of the state); under
    ``reconstruct`` it does.  Both behaviours are asserted so neither can
    change silently.
    """
    from legoesm.ocean.diagnostics_sections import mass_fluxes_from_state

    _g, z_coord, model, state = _setup(store=True)
    out = model.step(_perturbed(state), _DT)
    # Double every velocity: the reconstruction is linear in u, the stored
    # flux cannot see the edit at all.
    edited = out._replace(u=out.u.replace(data=out.u.data * 2.0))

    auto_u, _ = mass_fluxes_from_state(edited, z_coord, model.grid,
                                       source="auto")
    np.testing.assert_allclose(np.asarray(auto_u),
                               np.asarray(out.mass_flux_u.data),
                               rtol=0, atol=0)

    rec_before, _ = mass_fluxes_from_state(out, z_coord, model.grid,
                                           source="reconstruct")
    rec_after, _ = mass_fluxes_from_state(edited, z_coord, model.grid,
                                          source="reconstruct")
    np.testing.assert_allclose(np.asarray(rec_after),
                               2.0 * np.asarray(rec_before),
                               rtol=1e-10, atol=0)


def test_min_water_column_m_reaches_the_reconstruction():
    """``source="reconstruct"`` must honour the floor the caller passed.

    The floor is ``water_col = max(eta + H_bathy, min_water_column_m)``, so it
    only bites above the deepest column: this basin is H_max = 1000 m, and an
    earlier version of this test used 500 m -- inert, and it went red for that
    reason rather than for a wiring defect.  5000 m forces every column to the
    floor, giving J = 5 and a 5x thickness.
    """
    from legoesm.ocean.diagnostics_sections import mass_fluxes_from_state

    _g, z_coord, model, state = _setup(store=True)
    out = model.step(_perturbed(state), _DT)
    a, _ = mass_fluxes_from_state(out, z_coord, model.grid,
                                  source="reconstruct",
                                  min_water_column_m=None)
    b, _ = mass_fluxes_from_state(out, z_coord, model.grid,
                                  source="reconstruct",
                                  min_water_column_m=5000.0)
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert np.max(np.abs(a)) > 0.0
    assert not np.allclose(a, b), (
        "min_water_column_m had no effect on the reconstruction -- either it "
        "is not forwarded or the chosen floor is inactive on this bathymetry")
    # It bites through the z* Jacobian, which is linear in the water column:
    # J = water_col / H_max, so a 5000 m floor over a 1000 m basin is exactly
    # 5x the thickness (eta is O(0.2 m), i.e. 2e-4 of the column).
    np.testing.assert_allclose(b, 5.0 * a, rtol=2e-3, atol=0)


# ---------------------------------------------------------------------------
# 6. driver wiring (codex RED 6)
# ---------------------------------------------------------------------------

def test_both_supported_gateway_builders_accept_store_mass_flux():
    """RED 6: ``--gateway-transports`` supports latlon AND tripole."""
    import inspect

    from scripts.run import run_omip_core2

    for fn_name in ("build_tripole", "build_latlon_bathy"):
        fn = getattr(run_omip_core2, fn_name)
        params = inspect.signature(fn).parameters
        assert "store_mass_flux" in params, (
            f"{fn_name} has no store_mass_flux parameter, so a "
            f"--gateway-transports run on that grid silently falls back to "
            f"the h*u reconstruction")
        assert params["store_mass_flux"].default is False


def test_every_supported_gateway_grid_branch_passes_store_mass_flux():
    """Each ``app_grid_type`` the guard accepts must have a wired builder.

    Reads the DRIVER SOURCE and, for every ``app_grid_type = "<supported>"``
    assignment, walks BACK to the builder call that produced it and requires a
    ``store_mass_flux=`` argument in between.  Source inspection is the only
    cheap way to pin wiring inside a monolithic ``main()``; it names the
    symbols that actually run and goes red if the kwarg is dropped from either
    branch.
    """
    import re
    from pathlib import Path

    from legoesm.ocean.diagnostics_sections import SUPPORTED_APP_GRIDS

    root = Path(__file__).resolve().parents[3]
    lines = (root / "scripts/run/run_omip_core2.py").read_text().splitlines()
    seen = set()
    for i, line in enumerate(lines):
        m = re.match(r'\s*app_grid_type = "([a-z_]+)"\s*$', line)
        if not m or m.group(1) not in SUPPORTED_APP_GRIDS:
            continue
        seen.add(m.group(1))
        start = next(j for j in range(i, -1, -1)
                     if re.search(r"=\s*build_\w+\($", lines[j]))
        block = "\n".join(lines[start:i])
        assert "store_mass_flux=" in block, (
            f"the builder call for app_grid_type={m.group(1)!r} "
            f"(line {start + 1}) does not pass store_mass_flux; "
            "--gateway-transports would silently reconstruct on that grid")
    assert seen == set(SUPPORTED_APP_GRIDS), (
        f"only found builder branches for {sorted(seen)}, expected "
        f"{sorted(SUPPORTED_APP_GRIDS)}")


def test_run_omip_restart_does_not_persist_the_diagnostic_flux():
    """YELLOW 10: the npz lane wrote them on save and DROPPED them on load.

    ``_load_restart`` rebuilds from a fresh template whose optional slots are
    ``None`` and skips any such slot, so a written ``mass_flux_*`` was silently
    ignored.  Classified DIAGNOSTIC (matching the ``_SLOT_POLICY`` the
    run_omip_core2 restart uses), so it is not written either -- save and load
    now agree by construction.
    """
    from scripts.run.run_omip import _RESTART_DIAGNOSTIC_SLOTS

    assert set(_RESTART_DIAGNOSTIC_SLOTS) == set(_NEW_SLOTS)


def test_restart_round_trip_drops_nothing_it_wrote(tmp_path):
    """Behavioural half of YELLOW 10: save -> load must not lose a slot.

    Everything ``_save_restart`` writes must come back out of
    ``_load_restart``; the ONLY slots allowed to differ are the declared
    diagnostics.  This fails if a future change starts persisting them again
    without teaching the loader to restore them.
    """
    from scripts.run.run_omip import _load_restart, _save_restart
    from scripts.run.run_omip import _join_restart_writer

    _g, _z, model, state = _setup(store=True)
    out = model.step(_perturbed(state), _DT)
    assert out.mass_flux_u is not None

    _save_restart(out, 1.0, 1, tmp_path, grid_type="latlon")
    _join_restart_writer()
    path = tmp_path / "restart_day000001.npz"
    assert path.exists()
    with np.load(path) as npz:
        written = set(npz.files)
    assert not (written & set(_NEW_SLOTS)), (
        f"the restart persists {sorted(written & set(_NEW_SLOTS))}, which "
        "_load_restart silently drops (fresh template slot is None)")

    template = out._replace(mass_flux_u=None, mass_flux_v=None)
    loaded, day, step = _load_restart(path, template, grid_type="latlon")
    assert (day, step) == (1.0, 1)
    for name in out._fields:
        src, dst = getattr(out, name), getattr(loaded, name)
        if src is None or not hasattr(src, "data"):
            continue
        if name in _NEW_SLOTS:
            assert dst is None, f"{name} is a diagnostic; it must not restore"
            continue
        np.testing.assert_allclose(np.asarray(dst.data), np.asarray(src.data),
                                   rtol=0, atol=0,
                                   err_msg=f"{name} did not survive the "
                                           "restart round trip")
