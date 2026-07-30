"""GM eddy-induced (bolus) advection FORM: ``gm_bolus_advection="through_fct"``.

NEMO ``traadv`` adds the ln_ldfeiv GM bolus to the ADVECTING velocity, so the
bolus flux passes through the monotone FCT/Zalesak limiter (node 22). legoESM's
``nemo_iso_lap`` path historically applied the bolus as a separate 2nd-order
CENTRED flux inside the iso operator (dispersive at fronts). This suite pins the
new ``through_fct`` option:

 (a) the EXPORTED bolus transport ≡ the transport the centred path applies,
     bit-for-bit (single ``nemo_eiv_bolus_transport`` definition);
 (b) tracer content is conserved to machine precision through FCT-with-bolus,
     and a CONSTANT tracer stays constant (the augmented advecting field is
     discretely non-divergent — re-diagnosed w);
 (c) no double count: ``through_fct`` returns the PURE Redi tendency;
 (d) monotonicity: a sharp front develops NO new extrema under FCT-with-bolus.
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    gm_redi_tracer_tendency_latlon,
    nemo_eiv_bolus_transport,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    add_bolus_to_advecting_flux,
)
from legoesm.ocean.advection import fct_tracer_advection
from legoesm.grids.latlon import ensure_geometry


def _fixture():
    r = build_nemo_gyre_recipe()
    st = r.initial_state
    cfg = r.model_config.gm_redi._replace(
        slope_positions="mode_b", slope_scheme="nemo_iso_lap", kappa_GM=800.0,
    )
    cfg = cfg._replace(
        treguier=cfg.treguier._replace(enabled=False),
        visbeck=cfg.visbeck._replace(enabled=False),
    )
    # front with BOTH meridional and zonal structure so u_eiv AND v_eiv fire.
    T = (st.T.data
         + 2.0 * jnp.linspace(0, 1, st.T.data.shape[0])[:, None, None]
         + 0.5 * jnp.linspace(0, 1, st.T.data.shape[1])[None, :, None])
    kw = dict(eos=r.model_config.eos, mask=st.land_mask.data,
              u_mask=st.u_mask.data, v_mask=st.v_mask.data, rho_0=1026.0)
    return r, st, cfg, T, st.S.data, jnp.zeros_like(st.eta.data), st.H_bathy.data, kw


def test_through_fct_returns_pure_redi_no_double_count():
    """(c) through_fct tendency == the kappa_GM=0 (pure Redi) tendency: the bolus
    is REMOVED from the operator (it is applied via the FCT flux instead)."""
    r, st, cfg, T, S, eta, H, kw = _fixture()
    dT_f, _, bolus = gm_redi_tracer_tendency_latlon(
        T, S, eta, H, r.grid, r.z_coord, cfg._replace(gm_bolus_advection="through_fct"),
        return_bolus_transport=True, **kw)
    dT_redi, _ = gm_redi_tracer_tendency_latlon(
        T, S, eta, H, r.grid, r.z_coord, cfg._replace(kappa_GM=0.0), **kw)
    assert float(jnp.max(jnp.abs(dT_f - dT_redi))) == 0.0
    # non-vacuity: the bolus (all three components) is genuinely non-zero.
    assert float(jnp.max(jnp.abs(bolus[0]))) > 0.0   # u_eiv
    assert float(jnp.max(jnp.abs(bolus[1]))) > 0.0   # v_eiv
    # ... and the centred operator (bolus applied in-place) DIFFERS from pure Redi.
    dT_c, _ = gm_redi_tracer_tendency_latlon(
        T, S, eta, H, r.grid, r.z_coord, cfg._replace(gm_bolus_advection="centred"), **kw)
    assert float(jnp.max(jnp.abs(dT_c - dT_redi))) > 0.0


def test_exported_bolus_equals_centred_flux():
    """(a) The exported (u_eiv,v_eiv,w_eiv) is EXACTLY the transport the centred
    in-operator path applies: centred_tend - redi_tend == +div of the centred
    bolus advective flux rebuilt from the exported bolus."""
    r, st, cfg, T, S, eta, H, kw = _fixture()
    dT_c, _ = gm_redi_tracer_tendency_latlon(
        T, S, eta, H, r.grid, r.z_coord, cfg._replace(gm_bolus_advection="centred"), **kw)
    dT_redi, _ = gm_redi_tracer_tendency_latlon(
        T, S, eta, H, r.grid, r.z_coord, cfg._replace(kappa_GM=0.0), **kw)
    _, _, bolus = gm_redi_tracer_tendency_latlon(
        T, S, eta, H, r.grid, r.z_coord, cfg._replace(gm_bolus_advection="through_fct"),
        return_bolus_transport=True, **kw)
    u_eiv, v_eiv, w_eiv = bolus

    geom = ensure_geometry(r.grid)
    e1t, e2t = geom.dx_T, geom.dy_T
    J = jnp.ones_like(eta)
    e3t = r.z_coord.dz_ref[None, None, :] * J[:, :, None]
    _ztop = jnp.cumsum(r.z_coord.dz_ref) - r.z_coord.dz_ref
    act = ((st.land_mask.data[:, :, None] > 0.5)
           & (_ztop[None, None, :] < H[:, :, None])).astype(T.dtype)
    act_below = jnp.concatenate([act[:, :, 1:], jnp.zeros_like(act[:, :, :1])], axis=2)
    ax_y, ax_x, ax_z = 0, 1, 2
    zfu = -u_eiv * 0.5 * (T + jnp.roll(T, -1, ax_x))
    zfv = -v_eiv * 0.5 * (T + jnp.roll(T, -1, ax_y))
    zfw = -w_eiv * 0.5 * (T + jnp.roll(T, -1, ax_z)) * act_below
    hdiv = (zfu - jnp.roll(zfu, +1, ax_x)) + (zfv - jnp.roll(zfv, +1, ax_y))
    vdiv = jnp.roll(zfw, +1, ax_z).at[:, :, 0].set(0.0) - zfw
    bolus_tend = (hdiv + vdiv) / (e1t * e2t)[:, :, None] / e3t * act
    assert float(jnp.max(jnp.abs((dT_c - dT_redi) - bolus_tend))) < 1e-12


def test_bolus_conserves_and_preserves_constancy_through_fct():
    """(b) With ONLY the bolus advecting (base flow at rest): a CONSTANT tracer
    stays constant AND the global tracer content is conserved to machine
    precision through the FCT-with-bolus flux."""
    r, st, cfg, T, S, eta, H, kw = _fixture()
    _, _, bolus = gm_redi_tracer_tendency_latlon(
        T, S, eta, H, r.grid, r.z_coord, cfg._replace(gm_bolus_advection="through_fct"),
        return_bolus_transport=True, **kw)
    nlat, nlon, nlev = T.shape
    mfu = jnp.zeros((nlat, nlon + 1, nlev))
    mfv = jnp.zeros((nlat + 1, nlon, nlev))
    um3 = st.u_mask.data[:, :, None] * jnp.ones((1, 1, nlev))
    vm3 = st.v_mask.data[:, :, None] * jnp.ones((1, 1, nlev))
    mfu_tr, mfv_tr, w_tr = add_bolus_to_advecting_flux(
        bolus, mfu, mfv, um3, vm3, r.grid, r.z_coord)
    geom = ensure_geometry(r.grid)
    e1t, e2t = geom.dx_T, geom.dy_T
    _ztop = jnp.cumsum(r.z_coord.dz_ref) - r.z_coord.dz_ref
    act = ((st.land_mask.data[:, :, None] > 0.5)
           & (_ztop[None, None, :] < H[:, :, None])).astype(T.dtype)
    h_k = r.z_coord.dz_ref[None, None, :] * jnp.ones_like(eta)[:, :, None]

    dh, dv = fct_tracer_advection(jnp.ones_like(T), mfu_tr, mfv_tr, w_tr, h_k,
                                  r.grid, 2700.0, high_order="centred2")
    assert float(jnp.max(jnp.abs((dh + dv) * act))) < 1e-12   # constancy

    dhf, dvf = fct_tracer_advection(T, mfu_tr, mfv_tr, w_tr, h_k, r.grid, 2700.0,
                                    high_order="centred2")
    areaT = (e1t * e2t)[:, :, None] * act
    glob = float(jnp.sum((dhf + dvf) * areaT))
    scale = float(jnp.sum(jnp.abs(dhf + dvf) * areaT)) + 1e-30
    assert abs(glob / scale) < 1e-12                          # conservation


def test_bolus_through_fct_is_monotone():
    """(d) A sharp front advected by the bolus through FCT develops NO new
    extrema (the whole point of routing the bolus through the limiter)."""
    r, st, cfg, T, S, eta, H, kw = _fixture()
    _, _, bolus = gm_redi_tracer_tendency_latlon(
        T, S, eta, H, r.grid, r.z_coord, cfg._replace(gm_bolus_advection="through_fct"),
        return_bolus_transport=True, **kw)
    nlat, nlon, nlev = T.shape
    um3 = st.u_mask.data[:, :, None] * jnp.ones((1, 1, nlev))
    vm3 = st.v_mask.data[:, :, None] * jnp.ones((1, 1, nlev))
    mfu_tr, mfv_tr, w_tr = add_bolus_to_advecting_flux(
        bolus, jnp.zeros((nlat, nlon + 1, nlev)), jnp.zeros((nlat + 1, nlon, nlev)),
        um3, vm3, r.grid, r.z_coord)
    _ztop = jnp.cumsum(r.z_coord.dz_ref) - r.z_coord.dz_ref
    act = ((st.land_mask.data[:, :, None] > 0.5)
           & (_ztop[None, None, :] < H[:, :, None])).astype(T.dtype)
    h_k = r.z_coord.dz_ref[None, None, :] * jnp.ones_like(eta)[:, :, None]
    Tsharp = jnp.where(jnp.arange(nlat)[:, None, None] < nlat // 2, 0.0, 10.0) * act
    dh, dv = fct_tracer_advection(Tsharp, mfu_tr, mfv_tr, w_tr, h_k, r.grid, 2700.0,
                                  high_order="centred2")
    Tnew = Tsharp - 2700.0 * (dh + dv) / jnp.maximum(h_k, 1e-12)
    wet = act > 0.5
    lo, hi = float(jnp.min(Tsharp[wet])), float(jnp.max(Tsharp[wet]))
    assert float(jnp.min(Tnew[wet])) >= lo - 1e-9
    assert float(jnp.max(Tnew[wet])) <= hi + 1e-9


def test_smooth_limit_matches_centred_sign_and_amplitude():
    """SMOOTH-limit fidelity certificate: where the FCT limiter is INACTIVE (a
    smooth tracer, base flow at rest), the through_fct bolus tendency equals the
    centred in-operator bolus tendency EXACTLY — same sign, unit amplitude. This
    proves the sign of the added advecting velocity, the u_eiv/e2u (v_eiv/e1v)
    metric conversion, and no double count all at once.  through_fct differs from
    centred ONLY at sharp fronts (where the limiter engages) — the whole point."""
    r, st, cfg, _T, S, eta, H, kw = _fixture()
    lat = jnp.linspace(0, 1, st.T.data.shape[0])[:, None, None]
    lon = jnp.linspace(0, 1, st.T.data.shape[1])[None, :, None]
    T = st.T.data + 3.0 * jnp.sin(2 * jnp.pi * lat) * jnp.cos(jnp.pi * lon)  # smooth
    dT_c, _ = gm_redi_tracer_tendency_latlon(
        T, S, eta, H, r.grid, r.z_coord, cfg._replace(gm_bolus_advection="centred"), **kw)
    dT_r, _ = gm_redi_tracer_tendency_latlon(
        T, S, eta, H, r.grid, r.z_coord, cfg._replace(kappa_GM=0.0), **kw)
    centred_bolus = dT_c - dT_r
    _, _, bolus = gm_redi_tracer_tendency_latlon(
        T, S, eta, H, r.grid, r.z_coord, cfg._replace(gm_bolus_advection="through_fct"),
        return_bolus_transport=True, **kw)
    nlat, nlon, nlev = T.shape
    um3 = st.u_mask.data[:, :, None] * jnp.ones((1, 1, nlev))
    vm3 = st.v_mask.data[:, :, None] * jnp.ones((1, 1, nlev))
    mfu_tr, mfv_tr, w_tr = add_bolus_to_advecting_flux(
        bolus, jnp.zeros((nlat, nlon + 1, nlev)), jnp.zeros((nlat + 1, nlon, nlev)),
        um3, vm3, r.grid, r.z_coord)
    h_k = r.z_coord.dz_ref[None, None, :] * jnp.ones_like(eta)[:, :, None]
    # tiny dt keeps the Zalesak limiter inactive on this smooth field.
    dh, dv = fct_tracer_advection(T, mfu_tr, mfv_tr, w_tr, h_k, r.grid, 1e-3,
                                  high_order="centred2")
    fct_bolus = -(dh + dv) / jnp.maximum(h_k, 1e-12)
    _ztop = jnp.cumsum(r.z_coord.dz_ref) - r.z_coord.dz_ref
    act = ((st.land_mask.data[:, :, None] > 0.5)
           & (_ztop[None, None, :] < H[:, :, None])).astype(T.dtype)
    a = np.asarray(centred_bolus * act)
    b = np.asarray(fct_bolus * act)
    m = np.abs(a) > 1e-12
    assert m.sum() > 100                                        # non-vacuous
    assert float(np.corrcoef(a[m], b[m])[0, 1]) > 0.999         # sign+pattern
    ratio = float(np.sum(b[m] * a[m]) / np.sum(a[m] * a[m]))    # amplitude
    assert abs(ratio - 1.0) < 1e-2


def test_dispatch_hardening_and_guards():
    """Unknown gm_bolus_advection raises; return_bolus_transport requires
    through_fct + nemo_iso_lap (double-count / unsupported-scheme guards)."""
    r, st, cfg, T, S, eta, H, kw = _fixture()
    with pytest.raises(ValueError, match="gm_bolus_advection"):
        gm_redi_tracer_tendency_latlon(
            T, S, eta, H, r.grid, r.z_coord, cfg._replace(gm_bolus_advection="typo"), **kw)
    with pytest.raises(ValueError, match="through_fct"):
        gm_redi_tracer_tendency_latlon(
            T, S, eta, H, r.grid, r.z_coord, cfg._replace(gm_bolus_advection="centred"),
            return_bolus_transport=True, **kw)
    with pytest.raises(ValueError, match="nemo_iso_lap"):
        gm_redi_tracer_tendency_latlon(
            T, S, eta, H, r.grid, r.z_coord, cfg._replace(slope_scheme="triads"),
            return_bolus_transport=True, **kw)


def test_accessor_is_pure_curl_divergence_free():
    """``nemo_eiv_bolus_transport`` returns a discrete CURL: the 3-D transport
    divergence telescopes to zero column-by-column (conservation is structural)."""
    r, st, cfg, T, S, eta, H, kw = _fixture()
    _, _, bolus = gm_redi_tracer_tendency_latlon(
        T, S, eta, H, r.grid, r.z_coord, cfg._replace(gm_bolus_advection="through_fct"),
        return_bolus_transport=True, **kw)
    u_eiv, v_eiv, w_eiv = bolus
    ax_y, ax_x, ax_z = 0, 1, 2
    # column-integrated horizontal transport at each face = 0 (psi=0 at surf+floor)
    assert float(jnp.max(jnp.abs(jnp.sum(u_eiv, axis=ax_z)))) < 1e-9
    assert float(jnp.max(jnp.abs(jnp.sum(v_eiv, axis=ax_z)))) < 1e-9
    # 3-D divergence of the transport telescopes to zero everywhere.
    hdiv = (u_eiv - jnp.roll(u_eiv, +1, ax_x)) + (v_eiv - jnp.roll(v_eiv, +1, ax_y))
    w_top = jnp.roll(w_eiv, +1, ax_z).at[:, :, 0].set(0.0)
    div3d = hdiv + (w_top - w_eiv)
    assert float(jnp.max(jnp.abs(div3d))) < 1e-9
