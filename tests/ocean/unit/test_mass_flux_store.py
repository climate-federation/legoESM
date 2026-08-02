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

    Every perturbed leaf is written back at the SOURCE FIELD'S dtype, i.e. the
    run's storage precision.  Without that, ``jnp.asarray`` of a numpy float64
    array silently promotes the state to f64 under ``JAX_ENABLE_X64=1`` while
    ``_step_impl``'s closing ``cast_pytree(..., "storage")`` still returns f32
    -- carry-in dtype != carry-out dtype, which no production driver ever sees
    (their states come from ``_init_rest_state`` at storage precision) but
    which aborts any ``lax.scan`` driven straight from this helper.  The
    float64 arm of the precision-scaling test gets f64 from its POLICY, so it
    is unaffected: the source fields are f64 there too.
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
        eta=state.eta.replace(
            data=jnp.asarray(eta_p, dtype=state.eta.data.dtype)),
        u=state.u.replace(data=jnp.asarray(u_p, dtype=state.u.data.dtype)),
        T=state.T.replace(data=jnp.asarray(T_p, dtype=state.T.data.dtype)))


_NEW_SLOTS = ("mass_flux_u", "mass_flux_v", "mass_flux_w")
_HORIZ_SLOTS = ("mass_flux_u", "mass_flux_v")


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
    for nm in _NEW_SLOTS:
        assert getattr(out, nm) is None, f"{nm} populated with the flag off"


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
    # The vertical partner is at layer INTERFACES: one MORE level than
    # state.w, which is the interface pair averaged to cell centres.
    w_shape = np.asarray(out.w.data).shape
    assert np.asarray(out.mass_flux_w.data).shape == \
        w_shape[:-1] + (w_shape[-1] + 1,)


def test_stored_w_is_the_base_w_baro_when_gm_is_off():
    """With GM off, ``mass_flux_w`` IS the ``w_baro`` ``state.w`` averages.

    ``state.w = 0.5*(w_baro[..., :-1] + w_baro[..., 1:])`` and, with no bolus,
    the stored interface field is that same ``w_baro`` -- so averaging the
    stored field must reproduce ``state.w`` to ROUND-OFF.

    Not bit-equality, and the reason is measured rather than assumed: the step
    forms ``w_full`` at working precision and its single closing
    ``cast_pytree(..., "storage")`` then rounds ``w`` and ``mass_flux_w``
    INDEPENDENTLY, so averaging the rounded interfaces is not the rounded
    average.  Observed max relative difference 8.3e-8 < 1 eps(float32).  The
    bar is therefore in ULP of the stored dtype -- tight enough that a
    genuinely different array (the h*u-scale differences elsewhere in this
    file are 1e-5 and up) cannot slip under it.
    """
    _g, _z, model, state = _setup(store=True)
    assert model.config.gm_redi is None
    out = model.step(_perturbed(state), _DT)
    mfw = np.asarray(out.mass_flux_w.data, dtype=np.float64)
    eps = float(np.finfo(np.asarray(out.mass_flux_w.data).dtype).eps)
    np.testing.assert_allclose(
        0.5 * (mfw[..., :-1] + mfw[..., 1:]),
        np.asarray(out.w.data, dtype=np.float64), rtol=4.0 * eps, atol=0,
        err_msg="mass_flux_w is not the w_baro that state.w is built from")


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


def test_stored_triple_is_matched_and_state_w_is_not_part_of_it():
    """Closes YELLOW 9: the stored VERTICAL partner tracks the stored pair.

    Under ``gm_bolus_advection="through_fct"`` the step advects tracers with
    the bolus-inclusive ``_tr`` pair AND its own re-diagnosed ``w_baro_tr``,
    while ``state.w`` stays the BASE ``w_baro`` (the momentum/continuity/eta
    vertical velocity, which must not change).  Round 5 stored only the
    horizontal pair and DOCUMENTED the mismatch; codex round 6 rejected that
    ("documentation plus a mismatch test do not close the 3-D-budget hole"),
    so the vertical partner is stored too.

    Two things are asserted against the ``"centred"`` control, and they are
    what make the triple usable:
      * ``mass_flux_w`` RESPONDS to the bolus arm (it is the ``_tr`` vertical
        velocity, not a copy of the base one) -- if it did not, storing it
        would be pointless;
      * ``state.w`` does NOT respond (it is bit-identical across arms), which
        is what makes it the WRONG partner and the stored triple the right
        one.
    """
    _gB, _zB, mB, sB = _setup(store=True, gm_bolus="centred")
    _gC, _zC, mC, sC = _setup(store=True, gm_bolus="through_fct")
    outB = mB.step(_perturbed(sB), _DT)
    outC = mC.step(_perturbed(sC), _DT)

    wB = np.asarray(outB.w.data, dtype=np.float64)
    wC = np.asarray(outC.w.data, dtype=np.float64)
    np.testing.assert_allclose(wC, wB, rtol=0.0, atol=0.0, err_msg=(
        "state.w now responds to the GM bolus arm.  It is supposed to stay "
        "the BASE w_baro (momentum/continuity/eta); if it is intentionally "
        "built from w_baro_tr, that is a physics change, not a diagnostic "
        "one -- update the field comment on "
        "LatLonCGridOceanState.mass_flux_u and this test together"))

    mfwB = np.asarray(outB.mass_flux_w.data, dtype=np.float64)
    mfwC = np.asarray(outC.mass_flux_w.data, dtype=np.float64)
    scale = float(np.max(np.abs(mfwB)))
    assert scale > 0.0, "the base w_baro is zero -- the test cannot discriminate"
    assert float(np.max(np.abs(mfwC - mfwB))) > 1e-10 * scale, (
        "mass_flux_w is IDENTICAL across the centred/through-FCT arms, so it "
        "is the BASE w_baro and not the _tr vertical partner -- the stored "
        "triple is then no more consistent than pairing with state.w was")


def test_the_bolus_increment_the_triple_carries_is_divergence_free():
    """The invariant the stored triple ACTUALLY satisfies.

    RETRACTED CLAIMS (both refuted by review, both kept here as warnings):
      * ``div_h(mass_flux_u, mass_flux_v) + dz(mass_flux_w) == 0`` -- FALSE on
        the default moving z* column, where ``w_baro`` carries the
        layer-thickness (sigma) tendency.  The tracer update is a MOVING-CELL
        budget (``h_new*T_new = h_old*T_mid - dt*[...]``), not a
        divergence-free one.
      * the replacement ``... == -dh/dt`` -- ALSO FALSE, and do not reinstate
        it either: thickness additionally moves through the freshwater eta
        forcing, the eta floor and the volume-drift projection, none of which
        are advective and none of which appear in these arrays.
    NO closed-budget identity is implied by this test.  It asserts one thing:
    the bolus INCREMENT cancels.

    What IS exactly true, and what ``add_bolus_to_advecting_flux`` guarantees:
    the BOLUS INCREMENT is discretely non-divergent, because the bolus is
    column-non-divergent (psi = 0 at surface and floor) and its vertical
    partner is re-diagnosed through the SAME continuity operator.  So the
    through-FCT and centred arms -- which differ ONLY by that increment --
    must have IDENTICAL divergence.

    TWO NORMALIZERS, because they answer different questions and using only
    one is wrong in a different way each time:

    * against the INCREMENT (codex round-8 YELLOW 4): is the cancellation
      PHYSICALLY real, or is the "non-divergent bolus" claim hiding under a
      big base signal?  Measured 4.8e-5 of ``|div_h(bolus)|``.
    * against the BASE: what round-off floor is even ACHIEVABLE?  The
      increment is a DIFFERENCE OF TWO INDEPENDENTLY-ROUNDED f32 fields whose
      own divergence is ~1e3 larger, so catastrophic cancellation puts the
      floor at ``eps * |div_h(base)|`` -- NOT at ``eps * |div_h(bolus)|``.
      An increment-only bar of 200 eps demands 4e-16 absolute, which f32
      arithmetic cannot deliver: the first version of this test asked for
      exactly that and measured 405 eps, i.e. it was failing on precision, not
      on physics.  The diagnostic's printed precision bounds the claim it can
      support.

    So: the residual must be at the base's round-off floor AND negligible
    against the increment it is cancelling.  Both are asserted, with their
    scales printed, and both scales are asserted non-zero.
    """
    from legoesm.grids.operators_latlon_cgrid import divergence_cgrid

    _gB, _zB, mB, sB = _setup(store=True, gm_bolus="centred")
    _gC, _zC, mC, sC = _setup(store=True, gm_bolus="through_fct")
    outB = mB.step(_perturbed(sB), _DT)
    outC = mC.step(_perturbed(sC), _DT)

    def _arr(out, nm):
        return np.asarray(getattr(out, nm).data, dtype=np.float64)

    # The BOLUS INCREMENT itself: the two arms differ by nothing else.
    d_mfu = _arr(outC, "mass_flux_u") - _arr(outB, "mass_flux_u")
    d_mfv = _arr(outC, "mass_flux_v") - _arr(outB, "mass_flux_v")
    d_mfw = _arr(outC, "mass_flux_w") - _arr(outB, "mass_flux_w")

    div_h_bolus = np.asarray(
        divergence_cgrid(jnp.asarray(d_mfu), jnp.asarray(d_mfv), mB.grid),
        dtype=np.float64)
    # dz(w) with the SAME interface convention the tracer update uses:
    # w[k] is the TOP of layer k, w[k+1] the bottom.
    dz_w_bolus = d_mfw[..., :-1] - d_mfw[..., 1:]

    # The BASE arm's own horizontal divergence: the magnitude the increment
    # was differenced out of, hence the round-off floor.
    div_h_base = np.asarray(
        divergence_cgrid(jnp.asarray(_arr(outB, "mass_flux_u")),
                         jnp.asarray(_arr(outB, "mass_flux_v")), mB.grid),
        dtype=np.float64)

    scale_inc = float(np.max(np.abs(div_h_bolus)))
    scale_base = float(np.max(np.abs(div_h_base)))
    assert scale_inc > 0.0, (
        "the bolus increment's HORIZONTAL divergence is identically zero, so "
        "there is nothing for the vertical term to cancel and this test "
        "cannot discriminate -- the GM bolus is not active in this setup")
    assert scale_base > 0.0
    resid = float(np.max(np.abs(div_h_bolus + dz_w_bolus)))
    eps = float(np.finfo(np.asarray(outB.mass_flux_u.data).dtype).eps)

    assert resid < 200.0 * eps * scale_base, (
        f"|div_h(bolus) + dz(w_bolus)| = {resid:.3e} exceeds the f32 "
        f"round-off floor set by the base field it was differenced from "
        f"(|div_h(base)| = {scale_base:.3e}, {resid / scale_base / eps:.1f} "
        "eps).  That is beyond cancellation error: mass_flux_w is not the "
        "partner re-diagnosed from the bolus-augmented horizontal flux.")
    assert resid < 1.0e-3 * scale_inc, (
        f"the GM bolus increment carried by the stored triple is NOT "
        f"meaningfully divergence-free: the residual {resid:.3e} is "
        f"{resid / scale_inc:.2e} of |div_h(bolus)| = {scale_inc:.3e}, i.e. "
        "the vertical term does not actually cancel the horizontal one -- "
        "which breaks FCT constancy preservation on the augmented field.")


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
        MASS_FLUX_W_DIMS, MASS_FLUX_W_UNITS, MASS_FLUX_UNITS,
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
    # The vertical partner is a VELOCITY (the continuity operator already
    # divided by thickness), on INTERFACES -- so it must NOT inherit state.w's
    # cell-centred level dim, and it must not carry the m^2/s of the pair.
    mfw = out.mass_flux_w
    assert mfw.name == "mass_flux_w"
    assert mfw.units == MASS_FLUX_W_UNITS == "m/s"
    assert mfw.dims == MASS_FLUX_W_DIMS
    assert mfw.dims != out.w.dims, (
        "mass_flux_w must not claim state.w's dims -- it has one more level")
    assert mfw.staggering == out.w.staggering


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
    for nm in _NEW_SLOTS:
        f = getattr(seeded, nm)
        assert f is not None, f"{nm} was not seeded -- lax.scan would crash"
        assert not np.any(np.asarray(f.data)), "the seed must be zeros"
    for nm, src in (("mass_flux_u", "u"), ("mass_flux_v", "v")):
        assert np.asarray(getattr(seeded, nm).data).shape == \
            np.asarray(getattr(seeded, src).data).shape
    w_shape = np.asarray(seeded.w.data).shape
    assert np.asarray(seeded.mass_flux_w.data).shape == \
        w_shape[:-1] + (w_shape[-1] + 1,)
    # Idempotent: re-seeding an already-prepared carry is a no-op.
    again = model.seed_scan_carry(seeded, _DT)
    np.testing.assert_array_equal(np.asarray(again.mass_flux_u.data),
                                  np.asarray(seeded.mass_flux_u.data))


def test_seed_is_a_noop_when_the_flag_is_off():
    _g, _z, model, state = _setup(store=False)
    seeded = model.seed_scan_carry(_perturbed(state), _DT)
    for nm in _NEW_SLOTS:
        assert getattr(seeded, nm) is None


def test_seed_canonicalizes_legacy_metadata_and_partial_pairs():
    """Codex round-6 YELLOW 4: a populated slot must be REBUILT, not kept.

    A state produced by an earlier commit carries ``units="m/s"`` on the
    stored pair (the metadata bug YELLOW 7 fixed).  ``Field`` metadata is
    pytree AUX data, so preserving such a Field verbatim -- which the first
    version of ``seed_mass_flux_carry`` did for any non-``None`` slot -- is
    itself a treedef mismatch against what the step writes, i.e. exactly the
    scan crash the seeding exists to prevent.

    Also covers the PARTIAL carry (some slots Field, some ``None``), which was
    the other way to end up with a mixed-provenance pytree.

    BOTH cases are exercised, and the FULLY-POPULATED one is the load-bearing
    half: an implementation that early-returns when every slot is non-``None``
    (the natural "already seeded, nothing to do" shortcut) still canonicalizes
    the partial case, so a partial-only test passes while legacy metadata
    survives untouched.  That exact shortcut escaped as mutation N6.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        seed_mass_flux_carry,
    )

    _g, _z, model, state = _setup(store=True)
    stepped = model.step(_perturbed(state), _DT)

    legacy_cases = {
        # Every slot present, all with the pre-fix metadata: the case an
        # "if all(...) is not None: return state" shortcut would skip.
        "fully populated": stepped._replace(
            mass_flux_u=stepped.mass_flux_u.replace(units="m/s"),
            mass_flux_v=stepped.mass_flux_v.replace(units="m/s"),
            mass_flux_w=stepped.mass_flux_w.replace(
                units="m/s", dims=stepped.w.dims)),
        # Mixed: one legacy Field, one missing slot.
        "partial": stepped._replace(
            mass_flux_u=stepped.mass_flux_u.replace(units="m/s"),
            mass_flux_w=None),
    }
    for label, legacy in legacy_cases.items():
        assert legacy.mass_flux_u.units == "m/s", label
        fixed = seed_mass_flux_carry(legacy, True)
        for nm in _NEW_SLOTS:
            a, b = getattr(fixed, nm), getattr(stepped, nm)
            assert (a.name, a.dims, a.units, a.staggering) == \
                   (b.name, b.dims, b.units, b.staggering), (
                f"[{label}] {nm} kept its legacy metadata; the carry treedef "
                "still differs from what the step writes")
        assert (jax.tree_util.tree_structure(fixed)
                == jax.tree_util.tree_structure(stepped)), label
        # DATA is preserved wherever it existed -- canonicalizing metadata
        # must not silently discard values.
        np.testing.assert_array_equal(
            np.asarray(fixed.mass_flux_u.data),
            np.asarray(stepped.mass_flux_u.data), err_msg=label)
    assert not np.any(
        np.asarray(seed_mass_flux_carry(
            legacy_cases["partial"], True).mass_flux_w.data)), (
        "the missing slot must be zero-filled")


def test_seed_fast_path_is_keyed_on_metadata_not_on_presence():
    """Codex round-7 YELLOW 5: an already-canonical carry must be cheap...

    ...but the exit condition must be the METADATA, never "every slot is
    non-None".  The cheap presence test is exactly the shortcut that lets a
    legacy ``m/s`` pair through untouched (mutation N6).  Both halves are
    asserted: a canonical carry is returned IDENTICALLY (same object -- no
    rebuild, which is what makes the persistent-SPMD host loop cheap), and a
    non-canonical one is rebuilt even though every slot is present.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        seed_mass_flux_carry,
    )

    _g, _z, model, state = _setup(store=True)
    stepped = model.step(_perturbed(state), _DT)
    assert seed_mass_flux_carry(stepped, True) is stepped, (
        "an already-canonical carry was rebuilt; the fast path is not firing")

    legacy = stepped._replace(
        mass_flux_v=stepped.mass_flux_v.replace(units="m/s"))
    out = seed_mass_flux_carry(legacy, True)
    assert out is not legacy, "a legacy-metadata carry took the fast path"
    assert out.mass_flux_v.units == stepped.mass_flux_v.units


@pytest.mark.parametrize("slot", ["mass_flux_u", "mass_flux_v",
                                  "mass_flux_w"])
@pytest.mark.parametrize("attr,bad", [("name", "wrong"),
                                      ("dims", ("a", "b", "c")),
                                      ("units", "furlongs"),
                                      ("long_name", "not the donor's"),
                                      ("staggering", "vertex")])
def test_fast_path_rejects_a_difference_in_ANY_pytree_aux_member(slot, attr,
                                                                 bad):
    """Codex round-8 RED 2: the fast path must not ignore a metadata member.

    ``Field.tree_flatten`` puts name, dims, units, long_name AND staggering in
    the AUX tuple, so ANY of them differing is a different treedef and the
    next ``lax.scan`` rejects the carry.  The first fast path compared only
    ``name`` and ``units`` (plus ``dims`` for w), so a Field differing in
    ``dims`` (u/v), ``long_name`` or ``staggering`` sailed through -- and then
    ``_step_impl`` emitted the donor-derived metadata and broke the scan.

    Every member is mutated here, on every slot: 15 cases, each of which must
    force the rebuild.  A predicate that forgets one goes red on that case.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        seed_mass_flux_carry,
    )

    _g, _z, model, state = _setup(store=True)
    stepped = model.step(_perturbed(state), _DT)
    field = getattr(stepped, slot)
    assert getattr(field, attr) != bad, (
        f"the mutated {attr} equals the canonical one -- vacuous case")

    tweaked = stepped._replace(**{slot: field.replace(**{attr: bad})})
    fixed = seed_mass_flux_carry(tweaked, True)
    assert fixed is not tweaked, (
        f"a carry whose {slot}.{attr} differs from the canonical form took "
        "the fast path; the next lax.scan would reject it")
    assert getattr(getattr(fixed, slot), attr) == getattr(field, attr), (
        f"{slot}.{attr} was not restored to the canonical value")
    assert (jax.tree_util.tree_structure(fixed)
            == jax.tree_util.tree_structure(stepped))


def test_seed_normalizes_a_wrong_dtype_stored_slot():
    """Codex round-9 YELLOW 3: dtype is dynamic, so ``tree_flatten`` misses it.

    A same-shape, same-metadata slot at a DIFFERENT precision passes an
    aux-only canonical check, and then the step re-emits it at the storage
    precision -- a ``lax.scan`` carry mismatch on dtype rather than structure.
    The seeder must normalize it (and must not take the fast path on it).
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        seed_mass_flux_carry,
    )

    if not jax.config.jax_enable_x64:
        pytest.skip("needs JAX_ENABLE_X64=1 to build an f64 slot")
    _g, _z, model, state = _setup(store=True)
    stepped = model.step(_perturbed(state), _DT)
    want = np.asarray(stepped.mass_flux_u.data).dtype
    assert want == np.float32, "the preset no longer stores at f32"

    promoted = stepped._replace(
        mass_flux_u=stepped.mass_flux_u.replace(
            data=stepped.mass_flux_u.data.astype(jnp.float64)))
    fixed = seed_mass_flux_carry(promoted, True)
    assert fixed is not promoted, "an f64 slot took the canonical fast path"
    assert np.asarray(fixed.mass_flux_u.data).dtype == want, (
        "the f64 slot was not normalized to the donor's storage precision")
    # And the normalized carry really does survive a scan.
    seeded = model.seed_scan_carry(fixed, _DT)

    def _body(carry, _x):
        return model.step(carry, _DT), None

    final, _ = jax.lax.scan(_body, seeded, xs=None, length=2)
    assert final.mass_flux_u is not None


def test_seed_rejects_a_wrong_shaped_stored_slot():
    """Canonicalizing must not paper over a genuinely wrong layout."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        seed_mass_flux_carry,
    )

    _g, _z, model, state = _setup(store=True)
    stepped = model.step(_perturbed(state), _DT)
    bad = stepped._replace(
        mass_flux_v=stepped.mass_flux_v.replace(
            data=stepped.mass_flux_v.data[:-1]))     # v_lower, not v
    with pytest.raises(ValueError, match="mass_flux_v has shape"):
        seed_mass_flux_carry(bad, True)


def test_integrate_seeds_before_the_first_step(monkeypatch):
    """Codex round-6 YELLOW 5: ``integrate()`` did not seed its own state.

    Unseeded, ``trajectory[0]`` has a different pytree structure from every
    later entry (its slots are ``None``, theirs are ``Field``s), so a caller
    that stacks the trajectory gets a structure error; and the jitted step
    retraces between iteration 1 and 2.
    """
    _g, _z, model, state = _setup(store=True)
    final, traj = model.integrate(_perturbed(state), duration=2 * _DT, dt=_DT)
    assert len(traj) >= 2
    structs = {jax.tree_util.tree_structure(s) for s in traj}
    structs.add(jax.tree_util.tree_structure(final))
    assert len(structs) == 1, (
        "integrate() returned a trajectory whose entries have DIFFERENT "
        "pytree structures -- trajectory[0] was not seeded")
    for nm in _NEW_SLOTS:
        assert getattr(traj[0], nm) is not None, f"trajectory[0].{nm} is None"
        assert getattr(final, nm) is not None


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


def test_core2_scan_block_seeds_its_own_carry(monkeypatch):
    """Codex round-7 YELLOW 4, BEHAVIOURAL half of R6-2.

    The source-order test below pins WHERE the seed call sits; this runs the
    real ``build_omip2_scan_block_fn`` block with ``store_mass_flux=True`` and
    a RAW (unseeded) state, which is what the ``--scan-block`` driver does.
    Without the seed inside ``block_fn`` this raises a ``lax.scan`` carry
    structure mismatch.

    The CORE-II forcing sampler is stubbed to a zero ``OceanSurfaceForcing``:
    the forcing is not what is under test, and building a real device stack
    needs the multi-GB CORE-II files.  The scan, the step, and the carry are
    all real.
    """
    from legoesm.ocean.coupler import omip2_applicator as oa

    _g, _z, model, state = _setup(store=True)
    p = _perturbed(state)
    assert p.mass_flux_u is None, "the block must seed an unseeded state"

    zeros2d = jnp.zeros_like(state.eta.data)

    def _fake_forcing(st, **_kw):
        from legoesm.ocean.state import OceanSurfaceForcing
        return OceanSurfaceForcing(
            tau_x=zeros2d, tau_y=zeros2d, q_net=zeros2d, sw_down=zeros2d)

    monkeypatch.setattr(oa, "compute_omip2_surface_forcing_jax", _fake_forcing)

    block = oa.build_omip2_scan_block_fn(
        model, _DT, np.asarray(state.eta.data).shape)
    dummy = jnp.zeros((1,))
    out = block(p, dummy, dummy, dummy,
                jnp.arange(2, dtype=jnp.int32), jnp.int32(0))
    for nm in _NEW_SLOTS:
        f = getattr(out, nm)
        assert f is not None, (
            f"{nm} is None after the scan block -- it did not seed its carry")
        assert np.all(np.isfinite(np.asarray(f.data)))


def test_seeded_carry_survives_a_scan_with_a_tuple_carry():
    """Codex round-7 YELLOW 4: the JRA55 scans carry ``(state, ice_state)``.

    Seeding ``state`` BEFORE ``init = (state, ice_state)`` has to be enough --
    ``lax.scan`` matches the WHOLE carry structure, so a seed applied to the
    wrong member, or after the tuple is built, would not help.  Driving the
    real JRA55 block needs the multi-GB forcing cache, so this reproduces its
    carry SHAPE (a tuple whose first member is the ocean state) around the
    real model step.
    """
    _g, _z, model, state = _setup(store=True)
    seeded = model.seed_scan_carry(_perturbed(state), _DT)
    ice = jnp.zeros_like(state.eta.data)          # stand-in ice carry

    def _body(carry, _x):
        st, ic = carry
        return (model.step(st, _DT), ic), None

    (final, _ic), _ = jax.lax.scan(_body, (seeded, ice), xs=None, length=2)
    for nm in _NEW_SLOTS:
        assert getattr(final, nm) is not None
    assert (jax.tree_util.tree_structure((final, ice))
            == jax.tree_util.tree_structure((seeded, ice)))


def test_run_omip_scan_seeder_is_grid_agnostic_and_inert_when_off():
    """``_seed_mass_flux_for_scan`` must not touch a non-latlon model.

    It runs unconditionally at the JRA55 scan boundaries, which the MPAS and
    cubed-sphere lanes also reach.  Those configs have no ``store_mass_flux``
    field and those states have no such slots, so the helper must return the
    state UNCHANGED rather than raise.
    """
    from scripts.run.run_omip import _seed_mass_flux_for_scan

    class _NoFieldConfig:
        pass

    class _Model:
        config = _NoFieldConfig()

    sentinel = object()
    assert _seed_mass_flux_for_scan(_Model(), sentinel) is sentinel

    _g, _z, model_off, state = _setup(store=False)
    p = _perturbed(state)
    assert _seed_mass_flux_for_scan(model_off, p) is p

    _g2, _z2, model_on, state2 = _setup(store=True)
    p2 = _perturbed(state2)
    seeded = _seed_mass_flux_for_scan(model_on, p2)
    for nm in _NEW_SLOTS:
        assert getattr(seeded, nm) is not None


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
    assert np.asarray(sharded.mass_flux_w.data).shape == \
        np.asarray(out.mass_flux_w.data).shape, (
        "mass_flux_w is cell-centred horizontally and must keep its shape")
    back = gather_state_latlon(sharded, mesh)
    np.testing.assert_allclose(np.asarray(back.mass_flux_w.data),
                               np.asarray(out.mass_flux_w.data),
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(back.mass_flux_v.data), mfv,
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(back.mass_flux_u.data),
                               np.asarray(out.mass_flux_u.data),
                               rtol=0, atol=0)


@pytest.mark.parametrize("fused_halo", ["0", "1"])
def test_sharded_ocean_step_runs_with_the_flag_on(monkeypatch, fused_halo):
    """Codex round-6 YELLOW 7: exercise ``make_sharded_ocean_step`` ITSELF.

    The shard/gather round-trip above never calls the sharded STEP, so it
    would stay green with the ``sharded_step`` pre-seed removed while a
    production run died on ``out_specs`` (which is derived from the INPUT
    state, so a step that ADDS leaves has no spec for them).  This runs the
    real 2-device step from a RAW (unseeded) sharded state -- the exact
    production entry -- and compares it against the serial answer.

    Parametrized over ``LEGOESM_LATLON_SPMD_FUSED_HALO`` because the fused
    v-carrier reconstruction now packs THREE staggered carriers (v, v_mask,
    mass_flux_v) instead of two, and that path is selected at trace time.
    """
    from jax.sharding import Mesh

    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_state_latlon, make_sharded_ocean_step, shard_state_latlon,
    )

    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("needs >= 2 devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    monkeypatch.setenv("LEGOESM_LATLON_SPMD_FUSED_HALO", fused_halo)

    _g, _z, model, state = _setup(store=True)
    p = _perturbed(state)
    serial = model.step(p, _DT)

    mesh = Mesh(np.asarray(devices[:2]).reshape(2), ("lat",))
    # RAW state: slots are None.  The sharded step must seed them itself.
    assert p.mass_flux_u is None
    sharded_in = shard_state_latlon(p, mesh)
    assert sharded_in.mass_flux_u is None, (
        "the input to the sharded step must be UNSEEDED for this test to "
        "exercise the pre-seed inside sharded_step")

    step = make_sharded_ocean_step(model, mesh)
    out = step(sharded_in, _DT)
    got = gather_state_latlon(out, mesh)

    for nm in _NEW_SLOTS:
        f = getattr(got, nm)
        assert f is not None, (
            f"{nm} is None after the sharded step -- the pre-seed did not run")
        a = np.asarray(f.data)
        assert np.all(np.isfinite(a)), f"{nm} has non-finite entries"
        assert np.any(a != 0.0), f"{nm} is all zeros -- still the seed"
        np.testing.assert_allclose(
            a, np.asarray(getattr(serial, nm).data),
            rtol=2e-5, atol=1e-9,
            err_msg=(f"{nm} from the 2-device sharded step disagrees with the "
                     f"serial step (fused_halo={fused_halo})"))


def test_mpi_band_scatter_slices_every_mass_flux_slot():
    """RED 4 (scatter half): a populated global state MUST be sliced.

    Omitted, every rank kept the FULL-domain flux while every other field was
    band-local.  Uses two real band layouts; pure slicing, no MPI runtime.

    EVERY slot, not just the pair: an earlier version checked only ``u`` and
    ``v``, so dropping ``mass_flux_w`` from the scatter went uncaught
    (mutation N11).
    """
    from legoesm.parallel.latlon_mpi import (
        make_latlon_band_layout, scatter_state_latlon_cgrid_ocean,
    )

    _g, _z, model, state = _setup(store=True)
    out = model.step(_perturbed(state), _DT)
    mfu = np.asarray(out.mass_flux_u.data)
    mfv = np.asarray(out.mass_flux_v.data)
    mfw = np.asarray(out.mass_flux_w.data)
    n_lat, n_lon = np.asarray(out.T.data).shape[:2]

    for rank in (0, 1):
        layout = make_latlon_band_layout(rank, 2, n_lat, n_lon)
        band = scatter_state_latlon_cgrid_ocean(out, layout)
        s, e = layout.lat_start, layout.lat_end
        for nm in _NEW_SLOTS:
            assert getattr(band, nm) is not None, (
                f"{nm} vanished from the scattered band")
            assert np.asarray(getattr(band, nm).data).shape[0] < n_lat + 1, (
                f"{nm} kept its FULL-domain leading dim -- it was not sliced")
        # cell-centred (vertical-only stagger): rows [s, e), like ``w``
        assert np.asarray(band.mass_flux_w.data).shape[0] == \
            np.asarray(band.w.data).shape[0]
        np.testing.assert_allclose(np.asarray(band.mass_flux_w.data),
                                   mfw[s:e], rtol=0, atol=0)
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
    for nm in _NEW_SLOTS:
        assert getattr(band, nm) is None, f"{nm} appeared with the flag off"


def test_mpi_band_gather_collects_every_mass_flux_slot(monkeypatch):
    """RED 4 (gather half): rank 0 must not keep its BAND-LOCAL flux.

    ``gather_state_latlon_cgrid_ocean``'s per-field gather needs a live MPI
    communicator, so the collective itself is stubbed: ``gather_field_latlon``
    is replaced by a recorder.  What is under test is the WIRING -- that every
    slot goes through the gather and that ``mass_flux_v`` (and ONLY it) is
    declared a v-face -- which is exactly what was missing.

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
    SENTINEL = {"mass_flux_u": -11.5, "mass_flux_v": -13.25,
                "mass_flux_w": -17.75}
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
    assert got.get("mass_flux_w") == [False], (
        f"mass_flux_w gather calls: {got.get('mass_flux_w')!r}; expected "
        "exactly one with is_v_face=False (it is cell-centred horizontally; "
        "its extra dimension is VERTICAL, which the band never splits)")
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


def test_each_gateway_builder_body_actually_enables_the_flag():
    """Codex round-6 RED 3 (test vacuity): the parameter must be USED.

    The two tests above check the SIGNATURE and the CALL SITE.  Both would
    stay green if a builder accepted ``store_mass_flux`` and then ignored it,
    which is the same silent-reconstruction failure RED 6 was about, one level
    down.  This asserts each builder body puts the flag into the ``_ovr``
    override dict that gets applied to the config.
    """
    import inspect

    from scripts.run import run_omip_core2

    for fn_name in ("build_tripole", "build_latlon_bathy"):
        src = inspect.getsource(getattr(run_omip_core2, fn_name))
        assert "if store_mass_flux:" in src, (
            f"{fn_name} never branches on its store_mass_flux parameter")
        assert '_ovr["store_mass_flux"] = True' in src, (
            f"{fn_name} accepts store_mass_flux but never puts it into the "
            "config override dict -- the flag is dead on that grid")


def test_gateway_driver_demands_the_stored_flux_not_auto():
    """Codex round-6 RED 3: the per-step call must pass ``source="stored"``.

    Both supported grid branches turn the capture ON for
    ``--gateway-transports``, so a state WITHOUT it means some config path
    (a YAML override landing after the builder, a new unwired grid) silently
    disabled it.  With ``source="auto"`` the accumulator would quietly
    integrate the ``h*u`` reconstruction instead -- reporting a wrong number
    under a flag that promises the exact flux.  ``"stored"`` makes it raise.

    COMMENTS ARE STRIPPED FIRST, and that is not fussiness: the first version
    of this test searched the raw call block, which contains a COMMENT
    explaining why ``source="stored"`` is used -- so it passed with the kwarg
    mutated to ``"auto"``.  A source test satisfied by its own explanatory
    prose proves nothing (mutation N3).
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    src = (root / "scripts/run/run_omip_core2.py").read_text()
    m = re.search(r"_gw_acc = gateway_step\((.*?)\)\n", src, re.S)
    assert m, "the driver's gateway_step call site moved -- update this test"
    code = "\n".join(ln.split("#", 1)[0] for ln in m.group(1).splitlines())
    assert "source=" in code, (
        "the driver calls gateway_step with no explicit source=; provenance "
        "must be a choice at the call site, not a default")
    assert 'source="stored"' in code, (
        f"the driver calls gateway_step with source != 'stored' (effective "
        f"arguments: {code.strip()!r}); a silently disabled store_mass_flux "
        "would then be masked by the h*u fallback")


def test_gateway_transports_rejects_a_yaml_that_disables_the_capture():
    """Codex round-6 RED 3: YAML is applied AFTER the gateway builders.

    ``--config ocean.store_mass_flux=false`` would therefore switch the
    capture back off behind ``--gateway-transports``' back.  The driver must
    refuse the combination, exactly as it refuses ``--kpp-ri-crit`` against a
    YAML ``ocean.physics`` block for the same reason.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    lines = (root / "scripts/run/run_omip_core2.py").read_text().splitlines()
    guard = [i for i, ln in enumerate(lines)
             if '"store_mass_flux" in _ovr' in ln]
    assert guard, (
        "no guard rejecting a YAML store_mass_flux override against "
        "--gateway-transports")
    apply_at = next(i for i, ln in enumerate(lines)
                    if "model.config.replace_flat(**_ovr)" in ln)
    assert guard[0] < apply_at, (
        f"the guard (line {guard[0] + 1}) runs AFTER the YAML override is "
        f"applied (line {apply_at + 1}) -- it would never prevent anything")
    block = "\n".join(lines[guard[0]:apply_at])
    assert re.search(r"raise ValueError", block), (
        "the YAML/CLI store_mass_flux conflict is detected but not raised")


def test_scan_drivers_seed_the_carry_before_their_lax_scan():
    """Codex round-6 RED 1 / RED 2: every scan boundary must pre-seed.

    ``_step_impl`` turns the mass_flux_* slots from ``None`` into ``Field``s,
    which is a ``lax.scan`` carry-structure mismatch.  The SPMD pre-seed is
    INSIDE the sharded step, i.e. inside these scans' bodies -- too late.
    Each driver must seed before its own scan.  Asserting the ORDER of the two
    named symbols in the source is the only cheap way to pin that inside these
    long closures; it goes red if a seed call is dropped or moved after.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    checks = [
        ("scripts/run/run_omip.py", "_seed_mass_flux_for_scan(model, state)",
         "init = (state, ice_state) if enable_sea_ice else state", 2),
        ("packages/ocean/legoesm/ocean/coupler/omip2_applicator.py",
         "state = seed_mass_flux_carry(state, True)",
         "lax.scan(_body, (state, step0), idx_t_block)", 1),
    ]
    for rel, seed_tok, scan_tok, n_expected in checks:
        lines = (root / rel).read_text().splitlines()
        # CALL sites only -- the `def` line contains the same token.
        seeds = [i for i, ln in enumerate(lines)
                 if seed_tok in ln and not ln.lstrip().startswith("def ")]
        scans = [i for i, ln in enumerate(lines) if scan_tok in ln]
        # The seed's RESULT must be bound to `state`, and `state` must not be
        # rebound between the seed and the scan -- otherwise a call whose
        # return value is discarded, or an intervening `state = ...`, passes
        # an order check while the real carry stays unseeded (codex round-8
        # YELLOW 5).
        for s, c in zip(seeds, scans):
            assert lines[s].strip().startswith("state = "), (
                f"{rel}:{s + 1}: the seed call's result is not bound to "
                f"`state`: {lines[s].strip()!r}")
            rebinds = [k for k in range(s + 1, c)
                       if lines[k].strip().startswith("state = ")
                       or lines[k].strip().startswith("state, ")]
            assert not rebinds, (
                f"{rel}: `state` is rebound at line(s) "
                f"{[k + 1 for k in rebinds]} between the seed ({s + 1}) and "
                f"the scan ({c + 1}), discarding the seeded carry")
        assert len(seeds) == n_expected, (
            f"{rel}: expected {n_expected} seed call(s) {seed_tok!r}, found "
            f"{len(seeds)} -- a scan boundary lost its pre-seed")
        assert len(scans) == n_expected
        for s, c in zip(seeds, scans):
            assert s < c, (
                f"{rel}: the seed at line {s + 1} comes AFTER the scan at "
                f"line {c + 1}")


def test_generic_ocean_archive_neither_writes_nor_restores_the_diagnostics(
        tmp_path):
    """Codex round-7 YELLOW 2: the OTHER restart lane had the same hole.

    ``legoesm.ocean.restart.load_restart`` rebuilds a slot the template left
    ``None`` as a bare ``Field(data, name)`` -- WITHOUT dims or units.  For
    ``mass_flux_w`` (interface dims, m/s) that produces a Field whose pytree
    AUX DATA differs from what the step writes, so a run resumed from such an
    archive would abort its next ``lax.scan``.  Excluded from BOTH sides.

    The reject-an-old-archive half matters too: an npz written before the
    exclusion still carries the arrays, and the loader must skip them rather
    than reconstruct them with generic metadata.
    """
    from legoesm.ocean.restart import (
        DIAGNOSTIC_SLOTS, load_restart, save_restart,
    )

    assert set(DIAGNOSTIC_SLOTS) == set(_NEW_SLOTS)

    _g, _z, model, state = _setup(store=True)
    out = model.step(_perturbed(state), _DT)
    path = save_restart(out, tmp_path / "r.npz", time_s=0.0, step=1)
    with np.load(path) as npz:
        written = set(npz.files)
    assert not (written & set(_NEW_SLOTS)), (
        f"save_restart persisted {sorted(written & set(_NEW_SLOTS))}")
    assert "T" in written and "eta" in written, "nothing was written at all"

    template = out._replace(**{nm: None for nm in _NEW_SLOTS})
    # load_restart returns the STATE itself (a NamedTuple -- so an
    # `isinstance(x, tuple)` unwrap silently hands back its first FIELD).
    loaded = load_restart(path, template)
    for nm in _NEW_SLOTS:
        assert getattr(loaded, nm) is None, (
            f"{nm} came back from the archive; it is a diagnostic")
    np.testing.assert_allclose(np.asarray(loaded.T.data),
                               np.asarray(out.T.data), rtol=0, atol=0)

    # An OLD archive (written before the exclusion) must also be skipped, not
    # reconstructed with generic metadata.
    legacy_path = tmp_path / "legacy.npz"
    with np.load(path) as npz:
        payload = {k: npz[k] for k in npz.files}
    for nm in _NEW_SLOTS:
        payload[nm] = np.asarray(getattr(out, nm).data)
    np.savez(legacy_path, **payload)
    legacy_loaded = load_restart(legacy_path, template)
    for nm in _NEW_SLOTS:
        assert getattr(legacy_loaded, nm) is None, (
            f"{nm} was reconstructed from a legacy archive with generic "
            "metadata -- that state's treedef differs from what the step "
            "writes and the next lax.scan would abort")

    # A POPULATED template must have its diagnostic slots CLEARED, not left
    # alone (codex round-8 RED 3).  Skipping them leaves a STALE diagnostic
    # beside freshly loaded prognostics -- and this loader rebuilds the loaded
    # Fields WITHOUT their staggering, so a retained diagnostic (which kept
    # its donor's) and a reloaded u/v no longer agree, giving the state a
    # treedef the next step's output does not match.  Verified by driving a
    # real scan from the restored state.
    populated = load_restart(path, out)
    for nm in _NEW_SLOTS:
        assert getattr(populated, nm) is None, (
            f"{nm} survived a restart from a POPULATED template; it is a "
            "stale diagnostic, and its metadata no longer matches the "
            "reloaded prognostics")
    seeded = model.seed_scan_carry(populated, _DT)

    def _body(carry, _x):
        return model.step(carry, _DT), None

    final, _ = jax.lax.scan(_body, seeded, xs=None, length=2)
    assert final.mass_flux_w is not None


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

    # A FRESH template has every optional diagnostic slot at None, which is
    # what makes the load side drop anything the writer put there.
    template = out._replace(**{nm: None for nm in _NEW_SLOTS})
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
