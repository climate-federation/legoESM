"""FV3 physics coupling on the duo grid: A-grid tendency -> D-grid winds.

The first consumer of the port's physics-coupling wind vectors
(``compute_fv3_native_wind_vectors``, certified against the pinned Fortran
gridstruct to the float64 floor).  ``update_dwinds_phys_duo`` is the
``grid_type=0`` bounded-domain (duo) path of FV3's ``update_dwinds_phys``
(``model/fv_grid_utils.F90:3363-3547``): it rotates the A-grid physics wind
tendency into an Earth-frame 3-vector with ``vlon``/``vlat``, averages it onto
the cell edges, and projects onto the D-grid edge tangents ``es(:,:,:,1)`` /
``ew(:,:,:,2)`` to form the D-grid ``u``/``v`` increment.

DUO SCOPE.  ``bounded_domain`` is forced true on a duo grid
(``fv_arrays.F90:1512``), so the four ``.not. bounded_domain`` panel-edge blocks
in the reference (the ``edge_vect_w/e/s/n`` interpolations) are dead here and are
not ported; on an UNBOUNDED cubed sphere they are live and this routine would be
incomplete.  The routine is authored per level (2-D fields); the caller selects
the level, so it is level-agnostic and composes under ``vmap``/``scan``.

The NumPy function is the authority; ``update_dwinds_phys_duo_jax`` is its JAX
twin (same slicing, ``.at[].add`` for the scatter), gated against it in
``tests/grids/test_fv3_physics_coupling.py``.
"""
from __future__ import annotations

import numpy as np


def update_dwinds_phys_duo(u, v, u_dt, v_dt, dt, vlon, vlat, es1, ew2, n, ng):
    """Port of FV3 ``update_dwinds_phys``, duo (grid_type=0, bounded) path, one
    face, one level.  Dead ``edge_vect`` blocks omitted.

    Shapes (one face, full data domain, ``m = n + 2*ng``, ``ng >= 1``):

    =========  ===========  ====================================================
    ``u``      ``(m, m+1)``  D-grid west/east wind (updated in the returned copy)
    ``v``      ``(m+1, m)``  D-grid south/north wind
    ``u_dt``   ``(m, m)``    A-grid zonal wind tendency
    ``v_dt``   ``(m, m)``    A-grid meridional wind tendency
    ``vlon``   ``(m, m, 3)`` Earth-frame east unit vector, cell centred
    ``vlat``   ``(m, m, 3)`` Earth-frame north unit vector, cell centred
    ``es1``    ``(m, m+1,3)``N/S edge tangent ``es(:, i, j, 1)`` (component last)
    ``ew2``    ``(m+1, m,3)``E/W edge tangent ``ew(:, i, j, 2)``
    =========  ===========  ====================================================

    Returns ``(u_new, v_new)``; inputs are not mutated.
    """
    if ng < 1:
        raise ValueError(f"ng must be >= 1 (got {ng}); the edge projection would "
                         f"otherwise read the builder's NaN es/ew margins")
    dt5 = 0.5 * dt

    # v3 on (is-1:ie+1, js-1:je+1): A-grid tendency -> Earth-frame 3-vector.
    # local index 0 <-> global ng-1.
    w = slice(ng - 1, ng + n + 1)
    v3 = u_dt[w, w, None] * vlon[w, w, :] + v_dt[w, w, None] * vlat[w, w, :]

    # sum (NOT mean) the two straddling cell vectors onto each edge.
    # ue(i,j) = v3(i, j-1) + v3(i, j) for i in is:ie, j edges js:je+1
    ue = v3[1:n + 1, 0:n + 1, :] + v3[1:n + 1, 1:n + 2, :]      # (n, n+1, 3)
    # ve(i,j) = v3(i-1, j) + v3(i, j) for i edges is:ie+1, j in js:je
    ve = v3[0:n + 1, 1:n + 1, :] + v3[1:n + 2, 1:n + 1, :]      # (n+1, n, 3)

    # project onto the edge tangents -> D-grid increments.
    u_new = u.copy()
    u_new[ng:ng + n, ng:ng + n + 1] += dt5 * (
        ue * es1[ng:ng + n, ng:ng + n + 1, :]).sum(axis=-1)
    v_new = v.copy()
    v_new[ng:ng + n + 1, ng:ng + n] += dt5 * (
        ve * ew2[ng:ng + n + 1, ng:ng + n, :]).sum(axis=-1)
    return u_new, v_new


def update_dwinds_phys_duo_jax(u, v, u_dt, v_dt, dt, vlon, vlat, es1, ew2,
                               n, ng):
    """JAX twin of :func:`update_dwinds_phys_duo` -- identical slicing, ``.at``
    scatter for the non-mutating update.  Differentiable and jittable.

    ``n`` and ``ng`` set slice bounds, so they are static under ``jax.jit``
    (``static_argnums``).  Numerical identity with the NumPy authority holds at
    matched precision -- run with ``jax_enable_x64`` for the duo lane's fp64.
    """
    import jax.numpy as jnp

    if ng < 1:
        raise ValueError(f"ng must be >= 1 (got {ng})")
    dt5 = 0.5 * dt
    w = slice(ng - 1, ng + n + 1)
    v3 = u_dt[w, w, None] * vlon[w, w, :] + v_dt[w, w, None] * vlat[w, w, :]
    ue = v3[1:n + 1, 0:n + 1, :] + v3[1:n + 1, 1:n + 2, :]
    ve = v3[0:n + 1, 1:n + 1, :] + v3[1:n + 2, 1:n + 1, :]
    du = dt5 * (ue * es1[ng:ng + n, ng:ng + n + 1, :]).sum(axis=-1)
    dv = dt5 * (ve * ew2[ng:ng + n + 1, ng:ng + n, :]).sum(axis=-1)
    u_new = jnp.asarray(u).at[ng:ng + n, ng:ng + n + 1].add(du)
    v_new = jnp.asarray(v).at[ng:ng + n + 1, ng:ng + n].add(dv)
    return u_new, v_new


# --- Held-Suarez forcing (Held & Suarez 1994; driver/solo/hswf.F90) ---
_HS_DTY = 60.0            # equator-pole equilibrium dT [K]
_HS_DTZ = 10.0           # static-stability dtheta [K]
_HS_KAPPA = 2.0 / 7.0
_HS_P0 = 1.0e5           # reference pressure [Pa]
_HS_T0 = 200.0           # equilibrium-T floor [K]
_HS_T_EQ0 = 315.0        # equatorial surface equilibrium theta [K]
_HS_H0 = 7.0             # scale height [km], strat/meso lapse thickness
_HS_SDAY = 86400.0       # seconds per day
_HS_KA_DAYS = 40.0       # tropospheric k_a = 1/40 day
_HS_KS_DAYS = 4.0        # surface k_s = 1/4 day
_HS_KF_DAYS = 1.0        # Rayleigh k_f = 1/1 day
_HS_SIGB = 0.7           # boundary-layer sigma
_HS_MS_DAYS = 10.0       # mesosphere relaxation [day]
_HS_ST_DAYS = 40.0       # stratosphere relaxation at 100 mb [day]
_HS_STRAT_LAPSE = 2.25   # strat/meso lapse [K/km]
_HS_P_MESO = 1.0e2       # mesosphere top boundary, 1 mb [Pa]
_HS_P_STRAT = 100.0e2    # stratosphere boundary, 100 mb [Pa]
_HS_TAU_PREF = 100.0     # relaxation-time log reference (pl in mb)


def held_suarez_tend(pt, ua, va, delp, peln, pkz, pe, lat, pdt,
                     strat=True, radius=None):
    """Held-Suarez forcing, one cube face, all levels: faithful port of
    ``Held_Suarez_Tend`` (``driver/solo/hswf.F90``).  Tendencies are accumulated
    from zero; the k-loop runs bottom-up so the equilibrium-temperature column
    reproduces the Fortran ``teq(k) = teq(k+1) + dt_tropic`` recursion.

    Shapes: pt, ua, va, delp, pkz ``(n, n, npz)``; peln, pe ``(n, n, npz+1)``;
    lat ``(n, n)`` radians; pdt [s]; radius [m] (default ``constants.R_earth``).
    Returns ``(t_dt, u_dt, v_dt)`` each ``(n, n, npz)``.  NumPy authority;
    :func:`held_suarez_tend_jax` is its ``lax.scan`` JAX twin.
    """
    from legoesm import constants
    if radius is None:
        radius = constants.R_earth
    pt = np.asarray(pt, dtype=np.float64)
    ua = np.asarray(ua, dtype=np.float64)
    va = np.asarray(va, dtype=np.float64)
    delp = np.asarray(delp, dtype=np.float64)
    peln = np.asarray(peln, dtype=np.float64)
    pkz = np.asarray(pkz, dtype=np.float64)
    pe = np.asarray(pe, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    ny, nx, npz = pt.shape

    rdt = 1.0 / pdt
    rad_ratio = radius / constants.R_earth
    kf_day = _HS_SDAY * rad_ratio
    rkv = pdt / (_HS_KF_DAYS * kf_day)
    rka = pdt / (_HS_KA_DAYS * kf_day)
    rks = pdt / (_HS_KS_DAYS * kf_day)
    t_ms = _HS_MS_DAYS * rad_ratio
    t_st = _HS_ST_DAYS * rad_ratio
    tau = (t_st - t_ms) / np.log(_HS_TAU_PREF)
    rms = pdt / (t_ms * _HS_SDAY)
    rmr = 1.0 / (1.0 + rms)
    rsgb = 1.0 / (1.0 - _HS_SIGB)
    ap0k = 1.0 / _HS_P0 ** _HS_KAPPA
    algpk = np.log(ap0k)

    clat = np.cos(lat)
    c2 = clat ** 2
    tey = ap0k * (_HS_T_EQ0 - _HS_DTY * np.sin(lat) ** 2)
    tez = _HS_DTZ * (ap0k / _HS_KAPPA) * c2
    ps = pe[:, :, npz]                                  # surface pressure
    pl = delp / (peln[:, :, 1:] - peln[:, :, :-1])      # layer-mean pressure

    t_dt = np.zeros((ny, nx, npz))
    u_dt = np.zeros((ny, nx, npz))
    v_dt = np.zeros((ny, nx, npz))
    teq = np.zeros((ny, nx, npz + 1))                   # slot npz = teq(npz+1), unread

    for k in range(npz - 1, -1, -1):                    # bottom-up (Fortran k=npz..1)
        plk = pl[:, :, k]
        ptk = pt[:, :, k]
        pkzk = pkz[:, :, k]

        # troposphere: standard Held & Suarez
        sigl = plk / ps
        f1 = np.maximum(0.0, (sigl - _HS_SIGB) * rsgb)
        teq_t = tey - tez * (np.log(pkzk) + algpk)
        teq_t = np.maximum(_HS_T0, teq_t * pkzk)
        rkt = rka + (rks - rka) * f1 * c2 * c2          # cos^4 lat
        t_trop = rkt * (teq_t - ptk) / (1.0 + rkt) * rdt

        sigf = (sigl - _HS_SIGB) * rsgb * rkv
        fric = sigf > 0.0

        if strat:
            # dz = h0 * log(pl_{k+1}/pl_k) (LAYER-mean, not interface); the
            # bottom (k=npz-1) is always tropospheric so its clamped pl_{k+1}
            # is discarded by the meso/strat masks.
            plk1 = pl[:, :, k + 1] if k + 1 < npz else plk
            dz = _HS_H0 * np.log(plk1 / plk)
            meso = plk <= _HS_P_MESO
            stratm = (~meso) & (plk <= _HS_P_STRAT)
            teq_m = teq[:, :, k + 1] - _HS_STRAT_LAPSE * clat * dz
            t_meso = ((ptk + rms * teq_m) * rmr - ptk) * rdt
            with np.errstate(divide="ignore", invalid="ignore"):
                relx = pdt / ((t_ms + tau * np.log(0.01 * plk)) * _HS_SDAY)
                teq_s = teq[:, :, k + 1] + _HS_STRAT_LAPSE * clat * dz
                t_strat = relx * (teq_s - ptk) / (1.0 + relx) * rdt
            t_dt[:, :, k] += np.where(meso, t_meso,
                                      np.where(stratm, t_strat, t_trop))
            teq[:, :, k] = np.where(meso, teq_m,
                                    np.where(stratm, teq_s, teq_t))
            fric = fric & ~(meso | stratm)
        else:
            t_dt[:, :, k] += t_trop
            teq[:, :, k] = teq_t

        sigf = np.where(fric, sigf, 0.0)
        tmp = sigf / (1.0 + sigf) * rdt
        u_dt[:, :, k] -= (ua[:, :, k] + u_dt[:, :, k]) * tmp
        v_dt[:, :, k] -= (va[:, :, k] + v_dt[:, :, k]) * tmp
    return t_dt, u_dt, v_dt


def fv_update_phys_dry_duo(u, v, pt, ua, va, u_dt, v_dt, t_dt, dt,
                           vlon, vlat, es1, ew2, ng):
    """Port of FV3 ``fv_update_phys`` restricted to the DRY, HYDROSTATIC,
    ``moist_phys=.false.`` case on the duo grid (grid_type=0, bounded).

    Fortran (``model/fv_update_phys.F90``), hydrostatic branch, per level k:
        call moist_cp(...) -> cvm      ! nwat=0 (dry): case default -> cpm=cp_air
        pt = pt + t_dt*dt*con_cp/cvm   ! con_cp=cp_air, so the dry factor is 1.0
        ua = ua + dt*u_dt              ! GFS_PHYS undefined -> this block is live
        va = va + dt*v_dt
    then, once after the k-loop:
        call update_dwinds_phys(dt, u_dt, v_dt, u, v, ...)  ! D-grid increment
    There is NO p_var call: delp is unchanged and there are no physics tracers,
    so pe/peln/pk/pkz/ps are not rebuilt.  (The oracle DOES rebuild pe/peln/pk/ps
    unconditionally at fv_update_phys.F90:662-686, but with unchanged, internally
    consistent delp that rebuild is idempotent -- it returns the same arrays, so
    this kernel omitting it is numerically exact for this scope.)  SCOPE: the ``con_cp/cvm`` ratio is
    1.0 only because the air is dry; a moist coupling must restore it.

    Shapes (one cube face, full data domain ``m = n + 2*ng``, float64): pt, ua,
    va, u_dt, v_dt, t_dt ``(m,m,npz)``; u ``(m,m+1,npz)`` D-grid west/east;
    v ``(m+1,m,npz)``; vlon, vlat ``(m,m,3)``, es1 ``(m,m+1,3)``, ew2
    ``(m+1,m,3)`` the level-invariant wind-vector metrics consumed by
    ``update_dwinds_phys_duo``; ng the halo width (>=1).  The cell count n is
    recovered as ``pt.shape[0] - 2*ng``.

    PRECONDITION -- exchanged tendency halos.  The D-grid projection reads the
    ONE-CELL halo of ``u_dt``/``v_dt``, so the caller MUST pass them with their
    one-cell cross-face halos ALREADY EXCHANGED.  With ``dwind_2d=.false.`` (the
    HS deck) the Fortran completes a ``whalo=ehalo=shalo=nhalo=1`` group update
    on ``u_dt``/``v_dt`` before ``update_dwinds_phys`` (fv_update_phys.F90:645,
    698, 761).  This per-face kernel does NOT exchange them -- that is a six-face
    operation for the end-to-end assembly, exactly as ``update_dwinds_phys_duo``
    requires already-exchanged ``es``/``ew``.

    OMITS the surface-wind diagnostics ``u_srf = ua[...,npz-1]`` /
    ``v_srf = va[...,npz-1]`` (fv_update_phys.F90:675-676) and the ``phys_diag``
    tendency archive: both are diagnostics, not core prognostic state, and are
    not needed for the fv_core one-step parity.  Returns fresh
    ``(u_new, v_new, pt_new, ua_new, va_new)``; no input is mutated.

    The Fortran's single ``update_dwinds_phys`` (k-loop internal) is factored to
    one ``update_dwinds_phys_duo`` per level: levels are independent, the metrics
    do not vary with k, and the D-grid update reads the RAW A-grid tendencies
    (never the updated ua/va), so the ordering is equivalent.
    """
    dt = float(dt)
    ng = int(ng)
    n = pt.shape[0] - 2 * ng   # cell count; the helper derives m = n + 2*ng
    npz = pt.shape[2]

    # Scalar/A-grid update on the COMPUTE DOMAIN only (Fortran loops i=is:ie,
    # j=js:je; fv_update_phys.F90:365-373,420-427).  Halos are left as passed --
    # the oracle does NOT touch them here, and the D-grid projection below reads
    # the u_dt/v_dt halos, not ua/va.
    ci = slice(ng, ng + n)
    pt_new = pt.copy()
    ua_new = ua.copy()
    va_new = va.copy()
    pt_new[ci, ci] = pt[ci, ci] + t_dt[ci, ci] * dt
    ua_new[ci, ci] = ua[ci, ci] + u_dt[ci, ci] * dt
    va_new[ci, ci] = va[ci, ci] + v_dt[ci, ci] * dt

    u_new = np.empty_like(u, dtype=np.float64)
    v_new = np.empty_like(v, dtype=np.float64)
    for k in range(npz):
        uk, vk = update_dwinds_phys_duo(
            u[:, :, k], v[:, :, k],
            u_dt[:, :, k], v_dt[:, :, k],
            dt, vlon, vlat, es1, ew2, n, ng)
        u_new[:, :, k] = uk
        v_new[:, :, k] = vk

    return u_new, v_new, pt_new, ua_new, va_new


def fv_update_phys_dry_duo_jax(u, v, pt, ua, va, u_dt, v_dt, t_dt, dt,
                               vlon, vlat, es1, ew2, ng):
    """JAX twin of :func:`fv_update_phys_dry_duo`: identical math, jit-safe and
    differentiable.  npz is a static shape, so the level loop unrolls at trace
    time (no lax.scan carry, no branch on traced values); the D-grid buffers are
    built with functional ``.at[].set`` scatters, so nothing is mutated and
    autodiff is exact.  Assumes ``jax_enable_x64`` as elsewhere in this port.
    """
    import jax.numpy as jnp

    (u, v, pt, ua, va, u_dt, v_dt, t_dt,
     vlon, vlat, es1, ew2) = map(jnp.asarray, (u, v, pt, ua, va,
                                               u_dt, v_dt, t_dt,
                                               vlon, vlat, es1, ew2))
    ng = int(ng)
    n = int(pt.shape[0]) - 2 * ng   # cell count; helper derives m = n + 2*ng
    npz = int(pt.shape[2])

    # compute-domain-only scalar/A-grid update (see the NumPy authority)
    ci = slice(ng, ng + n)
    pt_new = pt.at[ci, ci].set(pt[ci, ci] + t_dt[ci, ci] * dt)
    ua_new = ua.at[ci, ci].set(ua[ci, ci] + u_dt[ci, ci] * dt)
    va_new = va.at[ci, ci].set(va[ci, ci] + v_dt[ci, ci] * dt)

    u_new = jnp.zeros_like(u)
    v_new = jnp.zeros_like(v)
    for k in range(npz):
        uk, vk = update_dwinds_phys_duo_jax(
            u[:, :, k], v[:, :, k],
            u_dt[:, :, k], v_dt[:, :, k],
            dt, vlon, vlat, es1, ew2, n, ng)
        u_new = u_new.at[:, :, k].set(uk)
        v_new = v_new.at[:, :, k].set(vk)

    return u_new, v_new, pt_new, ua_new, va_new


def moist_cp_warm_rain(q_v, q_c, q_r, *, cp_air, cp_vapor, c_liq):
    """``moist_cp`` for ``nwat = 4`` (fv_mapz.F90:3704-3708, the
    "K_warm_rain scheme with fake ice" case): the moist heat capacity
    ``(1 - qv - qd)*cp_air + qv*cp_vapor + qd*c_liq`` with
    ``qd = liq_wat + rainwat``.  Kessler carries no ice, so the fake ice
    slot is zero and drops out."""
    qd = q_c + q_r
    return (1.0 - (q_v + qd)) * cp_air + q_v * cp_vapor + qd * c_liq


def heating_to_fv3_cp_air(t_dt):
    """legoESM physics heating (``Q/(m c_pd)``, ``constants.c_pd``) in
    FV3's ``cp_air`` convention: ``fv_update_phys`` rescales ``t_dt`` by
    ``cp_air/cvm``, so handing it ``t_dt*c_pd/cp_air`` lands the energy
    ``Q`` on ``cvm`` exactly (codex 2026-09-24; the two ``c_p`` differ by
    5e-5)."""
    from legoesm import constants
    from legoesm.grids.fv3_native_gridstruct import FV3_CP_AIR
    return t_dt * (constants.c_pd / FV3_CP_AIR)


def fv_update_phys_moist_duo_jax(pt, delp, q, t_dt, q_dt, dt, *, n, ng,
                                 cp_air, cp_vapor, c_liq):
    """Port of FV3 ``fv_update_phys``'s ``nwat > 0`` scalar block on the
    compute window, hydrostatic (fv_update_phys.F90:318-372), for the
    Kessler tracer set ``q = [sphum, liq_wat, rainwat]`` (``nwat = 4``
    with the fake-ice slot absent).  Winds are not touched here (the
    D-grid increment is the dry twin's job and Kessler has none).

    Per level k, compute window only:
        q(m)   = q(m) + dt*q_dt(m)                          (:324)
        ps_dt  = 1 + dt*sum(q_dt(1:nwat))                   (:335)
        delp   = delp * ps_dt                               (:336)
        q(m)   = q(m) / ps_dt        (every mass-adjusted tracer, :352)
        pt     = pt + t_dt*dt*cp_air/cvm, cvm = moist_cp(q AFTER the
                 update, :367-371 -- moist_cp reads ``q``, which :324
                 has already advanced)

    The pressures (pe/peln/pk/pkz/ps, :662-686) are NOT rebuilt here:
    the caller rebuilds them from the returned delp with the lane's own
    ``p_var_hydrostatic`` -- the same producer that built them at the IC.

    WHY IT MATTERS (GLM 2026-09-24): condensation is layer-neutral
    (``sum(q_dt) = 0``, ``ps_dt = 1``); rain SEDIMENTATION moves water
    between layers and out of the column bottom, and only this block lets
    the layer mass follow it.  Without it precipitated water becomes dry
    air (~1 hPa of surface pressure per 100 mm accumulated rain).

    NO ORACLE RECEIPT: every Fortran deck in the wdump runs ``nwat = 0``,
    so this block is gated analytically (tests): identity at
    ``q_dt = 0``; column mass change == -(bottom rain flux)*dt; the
    dry-mass mixing ratio ``q/(1 - sum q)`` invariant under the
    renormalisation; the cp/cvm factor reproduced from ``moist_cp``.

    Shapes: pt, delp ``(m, m, npz)``; q, q_dt lists of three
    ``(m, m, npz)``; t_dt ``(m, m, npz)``.  Halos untouched.  Returns
    ``(pt_new, delp_new, [q_new x3], ps_dt)`` -- ``ps_dt`` on the compute
    block ``(n, n, npz)`` so the caller can renormalise every OTHER
    mass tracer it carries (:349-357 adjusts all of them); no input is
    mutated.
    """
    import jax.numpy as jnp
    pt, delp, t_dt = map(jnp.asarray, (pt, delp, t_dt))
    q = [jnp.asarray(a) for a in q]
    q_dt = [jnp.asarray(a) for a in q_dt]
    if len(q) != 3 or len(q_dt) != 3:
        raise ValueError(
            f"fv_update_phys_moist_duo_jax: nwat = 3 warm-rain tracers "
            f"[sphum, liq_wat, rainwat] expected, got {len(q)}/{len(q_dt)}")
    ng = int(ng)
    ci = slice(ng, ng + int(n))
    qc = [a[ci, ci] for a in q]
    dqc = [a[ci, ci] for a in q_dt]
    q_upd = [a + dt * da for a, da in zip(qc, dqc)]          # :324
    ps_dt = 1.0 + dt * (dqc[0] + dqc[1] + dqc[2])             # :335
    delp_c = delp[ci, ci] * ps_dt                             # :336
    q_adj = [a / ps_dt for a in q_upd]                        # :352
    cvm = moist_cp_warm_rain(q_adj[0], q_adj[1], q_adj[2], cp_air=cp_air,
                             cp_vapor=cp_vapor, c_liq=c_liq)  # :367
    pt_c = pt[ci, ci] + t_dt[ci, ci] * dt * cp_air / cvm      # :371
    pt_new = pt.at[ci, ci].set(pt_c)
    delp_new = delp.at[ci, ci].set(delp_c)
    q_new = [a.at[ci, ci].set(b) for a, b in zip(q, q_adj)]
    return pt_new, delp_new, q_new, ps_dt


def held_suarez_tend_jax(pt, ua, va, delp, peln, pkz, pe, lat, pdt,
                         strat=True, radius=None):
    """JAX twin of :func:`held_suarez_tend` (~1e-12 vs the NumPy authority at
    fp64; bit-exactness not required for this lane).  The Fortran
    ``teq(k) = teq(k+1) + dt_tropic`` recursion -- the ONLY cross-level
    dependence -- becomes a ``lax.scan`` over k REVERSED (bottom-up, k axis
    moved to the front and flipped), carrying ``teq_{k+1}`` of shape
    ``(ny, nx)`` with a zeros init (the authority's never-written slot npz).
    ``strat`` is a static Python bool (plain ``if``, one branch traced;
    ``static_argnums`` under ``jax.jit``); the meso/strat/fric DATA masks stay
    ``jnp.where``/``jnp.maximum``.  The authority's ``u_dt -= (ua + u_dt)*tmp``
    onto zeros, each level written once, reduces exactly to
    ``u_dt = -ua*tmp`` (same for v), and its ``t_dt += ...`` onto zeros to a
    plain per-level value -- so all three outputs are per-level maps and only
    ``teq`` rides the carry.  Assumes ``jax_enable_x64`` as elsewhere here.
    """
    import jax
    import jax.numpy as jnp

    from legoesm import constants
    if radius is None:
        radius = constants.R_earth
    pt = jnp.asarray(pt, dtype=jnp.float64)
    ua = jnp.asarray(ua, dtype=jnp.float64)
    va = jnp.asarray(va, dtype=jnp.float64)
    delp = jnp.asarray(delp, dtype=jnp.float64)
    peln = jnp.asarray(peln, dtype=jnp.float64)
    pkz = jnp.asarray(pkz, dtype=jnp.float64)
    pe = jnp.asarray(pe, dtype=jnp.float64)
    lat = jnp.asarray(lat, dtype=jnp.float64)
    ny, nx, npz = pt.shape

    rdt = 1.0 / pdt
    rad_ratio = radius / constants.R_earth
    kf_day = _HS_SDAY * rad_ratio
    rkv = pdt / (_HS_KF_DAYS * kf_day)
    rka = pdt / (_HS_KA_DAYS * kf_day)
    rks = pdt / (_HS_KS_DAYS * kf_day)
    t_ms = _HS_MS_DAYS * rad_ratio
    t_st = _HS_ST_DAYS * rad_ratio
    tau = (t_st - t_ms) / np.log(_HS_TAU_PREF)       # np on constants only
    rms = pdt / (t_ms * _HS_SDAY)
    rmr = 1.0 / (1.0 + rms)
    rsgb = 1.0 / (1.0 - _HS_SIGB)
    ap0k = 1.0 / _HS_P0 ** _HS_KAPPA
    algpk = np.log(ap0k)

    clat = jnp.cos(lat)
    c2 = clat ** 2
    tey = ap0k * (_HS_T_EQ0 - _HS_DTY * jnp.sin(lat) ** 2)
    tez = _HS_DTZ * (ap0k / _HS_KAPPA) * c2
    ps = pe[:, :, npz]                                  # surface pressure
    pl = delp / (peln[:, :, 1:] - peln[:, :, :-1])      # layer-mean pressure
    # plk1 per level: pl[:, :, k+1], clamped to plk at the bottom (k=npz-1),
    # exactly the authority's ``pl[:, :, k+1] if k+1 < npz else plk``.
    pl1 = jnp.concatenate([pl[:, :, 1:], pl[:, :, npz - 1:npz]], axis=2)

    def _rev_k(a):          # (ny, nx, npz) -> (npz, ny, nx), bottom level first
        return jnp.moveaxis(a, 2, 0)[::-1]

    def step(teq_kp1, x):
        ptk, uak, vak, plk, plk1, pkzk = x

        # troposphere: standard Held & Suarez
        sigl = plk / ps
        f1 = jnp.maximum(0.0, (sigl - _HS_SIGB) * rsgb)
        teq_t = tey - tez * (jnp.log(pkzk) + algpk)
        teq_t = jnp.maximum(_HS_T0, teq_t * pkzk)
        rkt = rka + (rks - rka) * f1 * c2 * c2          # cos^4 lat
        t_trop = rkt * (teq_t - ptk) / (1.0 + rkt) * rdt

        sigf = (sigl - _HS_SIGB) * rsgb * rkv
        fric = sigf > 0.0

        if strat:
            dz = _HS_H0 * jnp.log(plk1 / plk)
            meso = plk <= _HS_P_MESO
            stratm = (~meso) & (plk <= _HS_P_STRAT)
            teq_m = teq_kp1 - _HS_STRAT_LAPSE * clat * dz
            t_meso = ((ptk + rms * teq_m) * rmr - ptk) * rdt
            relx = pdt / ((t_ms + tau * jnp.log(0.01 * plk)) * _HS_SDAY)
            teq_s = teq_kp1 + _HS_STRAT_LAPSE * clat * dz
            t_strat = relx * (teq_s - ptk) / (1.0 + relx) * rdt
            t_dt_k = jnp.where(meso, t_meso,
                               jnp.where(stratm, t_strat, t_trop))
            teq_k = jnp.where(meso, teq_m,
                              jnp.where(stratm, teq_s, teq_t))
            fric = fric & ~(meso | stratm)
        else:
            t_dt_k = t_trop
            teq_k = teq_t

        sigf = jnp.where(fric, sigf, 0.0)
        tmp = sigf / (1.0 + sigf) * rdt
        u_dt_k = -uak * tmp        # authority: u_dt -= (ua + u_dt)*tmp, u_dt=0
        v_dt_k = -vak * tmp
        return teq_k, (t_dt_k, u_dt_k, v_dt_k)

    teq0 = jnp.zeros((ny, nx), dtype=jnp.float64)       # teq slot npz, unread
    xs = (_rev_k(pt), _rev_k(ua), _rev_k(va),
          _rev_k(pl), _rev_k(pl1), _rev_k(pkz))
    _, (t_dt_r, u_dt_r, v_dt_r) = jax.lax.scan(step, teq0, xs)

    def _unrev_k(a):        # (npz, ny, nx) bottom-up -> (ny, nx, npz)
        return jnp.moveaxis(a[::-1], 0, 2)

    return _unrev_k(t_dt_r), _unrev_k(u_dt_r), _unrev_k(v_dt_r)


# ---------------------------------------------------------------------------
# apply_held_suarez_step: the certified 3-pass six-face HS orchestration.
# PROMOTED VERBATIM from scripts/validate/fv3_native/full_step_oracle_parity.py
# (gated there at 1.7645e-8 vs the Fortran oracle, jobs 9473297/9473298) so
# the ModelDriver duo lane can import it; the script re-imports it from here
# -- ONE implementation total.  Byte-identical body; do not "clean up".
# ---------------------------------------------------------------------------
def apply_held_suarez_step(ctx, state, press, *, dt, n, ng, km, strat=True,
                           backend="numpy"):
    """In place: advance the six-face POST-DYNAMICS duo state by one
    Held-Suarez physics step -- the port of fv_phys + fv_update_phys for
    do_Held_Suarez=.true., dry, hydrostatic, nwat=0 (driver/solo/fv_phys.F90:
    533-591).  fv_dynamics has already run; this mutates state[t]["u"/"v"/"pt"].

    Three passes because the u_dt/v_dt one-cell halo exchange is a cross-face
    barrier (fv_update_phys.F90:645/698, dwind_2d=.false.): (1) per face, D->A
    Earth-frame winds (c2l_ord4_face, matching the deck's c2l_ord=4; d2a2c
    would be the wrong, local frame),
    the Held-Suarez tendencies on the compute domain,
    scattered back into full-domain arrays with zero halos; (2) exchange the
    u_dt/v_dt halos; (3) per face, apply fv_update_phys_dry_duo.  The pe/peln
    axis fix (i,k,j)->(i,j,k) and the pe 1-ring window match p_var_hydrostatic
    (verified against fv3_native_dynamics.py:198-234); pkz is already cell-domain
    k-last.  GLM-authored; codex + Claude reviewed.

    ``backend`` ("numpy" | "jax") selects the implementation of the THREE
    per-face kernels only: "numpy" is the authorities and the established
    score; "jax" routes them through their twins (c2l_ord4_face_jax,
    held_suarez_tend_jax, fv_update_phys_dry_duo_jax), marshalling np->jnp
    on the way in and back to np on the way out at each call.  The two halo
    exchanges and the state/press dicts stay NumPy in BOTH backends -- they
    are the exact mpp_update_domains strip-copy assembly glue (the ord4
    stencil needs the strip copy, NOT ext_vector's wedge re-extrapolation).
    """
    from legoesm.grids.fv3_native_ext_vector import c2l_ord4_face
    from legoesm.grids.fv3_native_gridstruct import (
        exchange_dgrid_vector_halos)
    from legoesm.grids.fv3_native_metrics import compute_fv3_native_wind_vectors
    from legoesm.core.fv3_native_physics_coupling import (
        held_suarez_tend, fv_update_phys_dry_duo)
    from legoesm.grids.fv3_native_gridstruct import exchange_agrid_scalar_halos

    # The backend switches ONLY the three per-face kernels: the jax backend
    # exercises the per-face physics twins; the strip-copy halo exchange is
    # numpy assembly glue in both backends, matching the dyn lane where the
    # exchange is not the per-face kernel.
    if backend == "jax":
        import jax.numpy as jnp

        from legoesm.grids.fv3_native_ext_vector import c2l_ord4_face_jax
        from legoesm.core.fv3_native_physics_coupling import (
            held_suarez_tend_jax, fv_update_phys_dry_duo_jax)

        # np -> jnp on the way in, np.asarray on the way out, at each
        # per-face call (the _make_jax_step marshalling style), so the
        # NumPy scatter/assembly around the kernels is identical in both
        # backends.
        def _c2l(u, v, amat, n, ng):
            ua, va = c2l_ord4_face_jax(
                jnp.asarray(u), jnp.asarray(v),
                tuple(jnp.asarray(a) for a in amat), n, ng)
            return np.asarray(ua), np.asarray(va)

        def _hs_tend(pt, ua, va, delp, peln, pkz, pe, lat, pdt,
                     strat=True):
            outs = held_suarez_tend_jax(
                jnp.asarray(pt), jnp.asarray(ua), jnp.asarray(va),
                jnp.asarray(delp), jnp.asarray(peln), jnp.asarray(pkz),
                jnp.asarray(pe), jnp.asarray(lat), pdt, strat=strat)
            return tuple(np.asarray(o) for o in outs)

        def _upd(u, v, pt, ua, va, u_dt, v_dt, t_dt, dt,
                 vlon, vlat, es1, ew2, ng):
            outs = fv_update_phys_dry_duo_jax(
                jnp.asarray(u), jnp.asarray(v), jnp.asarray(pt),
                jnp.asarray(ua), jnp.asarray(va), jnp.asarray(u_dt),
                jnp.asarray(v_dt), jnp.asarray(t_dt), dt,
                jnp.asarray(vlon), jnp.asarray(vlat),
                jnp.asarray(es1), jnp.asarray(ew2), ng)
            return tuple(np.asarray(o) for o in outs)
    elif backend == "numpy":
        _c2l = c2l_ord4_face
        _hs_tend = held_suarez_tend
        _upd = fv_update_phys_dry_duo
    else:
        raise ValueError(
            f"apply_held_suarez_step: unknown backend {backend!r} "
            "(choices: 'numpy', 'jax')")

    m = n + 2 * ng
    ci = slice(ng, ng + n)
    assert state[0]["pt"].shape == (m, m, km)
    assert press[0]["pkz"].shape == (n, n, km)
    ectx = ctx.get("ectx")
    if ectx is None:
        raise ValueError(
            "apply_held_suarez_step needs ctx['ectx'] (build the duo context "
            "with use_ext_bundle=True) for the c2l Earth-frame winds")

    ua6 = [None] * 6
    va6 = [None] * 6
    t_dt6 = [None] * 6
    u_dt6 = [None] * 6
    v_dt6 = [None] * 6

    # The oracle's c2l_ord4 exchanges the D-grid u/v halos before interpolating
    # (mpp_update_domains gridtype=DGRID_NE, fv_grid_utils.F90:2441); the ord4
    # edge stencil (compute cells i or j = 1 or n) reads those halos, so exchange
    # them first (exchange_dgrid_vector_halos = the exact mpp_update_domains
    # DGRID_NE strip copy, NOT ext_vector's wedge re-extrapolation -- codex).
    # Halo-only
    # (the compute interior is untouched), so PASS 3's D-grid update is unaffected.
    u6 = [state[t]["u"] for t in range(6)]
    v6 = [state[t]["v"] for t in range(6)]
    for tile in range(1, 7):
        exchange_dgrid_vector_halos(u6, v6, tile, n, ng)

    # PASS 1: per-face tendencies on the compute domain
    for t in range(6):
        gs = ctx["gs6"][t]
        # A-grid winds in the EARTH (lat-lon) frame -- the frame Held-Suarez
        # friction and update_dwinds (v3 = u_dt*vlon + v_dt*vlat) require.
        # c2l_ord4_face is the geographic 4th-order c2l (a-matrix rotation),
        # matching the deck's c2l_ord=4 (fv_arrays.F90:573); d2a2c_vect_duo
        # would give LOCAL-grid winds (wrong frame) and c2l_ord2 the wrong order.
        amat = ectx["amat6"][t]
        ua_f = np.empty((m, m, km), dtype=np.float64)
        va_f = np.empty((m, m, km), dtype=np.float64)
        for k in range(km):
            uak, vak = _c2l(state[t]["u"][:, :, k],
                            state[t]["v"][:, :, k],
                            amat, n, ng)
            ua_f[:, :, k] = uak
            va_f[:, :, k] = vak
        # c2l is valid is-1..ie+1; halos are NaN and never read on the compute
        # domain, but zero them so a stray downstream read cannot propagate NaN.
        ua_f = np.nan_to_num(ua_f, nan=0.0)
        va_f = np.nan_to_num(va_f, nan=0.0)
        ua6[t], va6[t] = ua_f, va_f

        pt_c = state[t]["pt"][ci, ci]                        # (n, n, km)
        delp_c = state[t]["delp"][ci, ci]
        ua_c, va_c = ua_f[ci, ci], va_f[ci, ci]
        pkz_c = press[t]["pkz"]                              # already (n, n, km)
        peln_c = np.transpose(press[t]["peln"], (0, 2, 1))   # (i,k,j)->(i,j,k)
        pe_c = np.transpose(press[t]["pe"][1:n + 1, :, 1:n + 1], (0, 2, 1))
        lat_c = gs["agrid_lat"][ci, ci]                      # (n, n) radians

        t_dt_c, u_dt_c, v_dt_c = _hs_tend(
            pt_c, ua_c, va_c, delp_c, peln_c, pkz_c, pe_c, lat_c, dt,
            strat=strat)

        t_dt_f = np.zeros((m, m, km), dtype=np.float64)
        u_dt_f = np.zeros((m, m, km), dtype=np.float64)
        v_dt_f = np.zeros((m, m, km), dtype=np.float64)
        t_dt_f[ci, ci] = t_dt_c
        u_dt_f[ci, ci] = u_dt_c
        v_dt_f[ci, ci] = v_dt_c
        t_dt6[t], u_dt6[t], v_dt6[t] = t_dt_f, u_dt_f, v_dt_f

    # PASS 2: cross-face one-cell halo exchange of the vector tendencies
    for tile in range(1, 7):
        exchange_agrid_scalar_halos(u_dt6, tile, n, ng)
    for tile in range(1, 7):
        exchange_agrid_scalar_halos(v_dt6, tile, n, ng)

    # PASS 3: apply
    for t in range(6):
        gs = ctx["gs6"][t]
        wv = compute_fv3_native_wind_vectors(
            gs["grid_lon"], gs["grid_lat"], gs["agrid_lon"], gs["agrid_lat"])
        u2, v2, pt2, _, _ = _upd(
            state[t]["u"], state[t]["v"], state[t]["pt"], ua6[t], va6[t],
            u_dt6[t], v_dt6[t], t_dt6[t], dt,
            wv["vlon"], wv["vlat"], wv["es1"], wv["ew2"], ng)
        state[t]["u"], state[t]["v"], state[t]["pt"] = u2, v2, pt2
    return None


def stack_held_suarez_metrics(ctx):
    """Face-stack the per-face metric inputs of the Held-Suarez step once.

    Returns ``(amat6, agrid_lat6, wind_vectors6)`` for
    :func:`apply_held_suarez_step_sixface_jax`: ``amat6`` a 4-tuple of
    ``(6, m, m)`` arrays, ``agrid_lat6`` ``(6, m, m)``, ``wind_vectors6``
    a dict of ``(6, ...)`` stacks -- exactly the arrays the NumPy
    authority reads per face from ``ctx["ectx"]["amat6"]``,
    ``ctx["gs6"]`` and ``compute_fv3_native_wind_vectors``.
    """
    from legoesm.grids.fv3_native_metrics import compute_fv3_native_wind_vectors
    ectx = ctx.get("ectx")
    if ectx is None:
        raise ValueError(
            "stack_held_suarez_metrics needs ctx['ectx'] (build the duo "
            "context with use_ext_bundle=True) for the c2l Earth-frame winds")
    amat6 = tuple(np.stack([np.asarray(ectx["amat6"][t][c]) for t in range(6)])
                  for c in range(4))
    agrid_lat6 = np.stack([np.asarray(ctx["gs6"][t]["agrid_lat"])
                           for t in range(6)])
    wvs = []
    for t in range(6):
        gs = ctx["gs6"][t]
        wvs.append(compute_fv3_native_wind_vectors(
            gs["grid_lon"], gs["grid_lat"], gs["agrid_lon"], gs["agrid_lat"]))
    wind_vectors6 = {k: np.stack([np.asarray(wv[k]) for wv in wvs])
                     for k in ("vlon", "vlat", "es1", "ew2")}
    return amat6, agrid_lat6, wind_vectors6


def column_view_sixface_jax(state, tab, amat6, *, n, ng, km):
    """The duo's physics-facing COLUMN view (passes 0-1 of the six-face
    Held-Suarez twin, factored out so every column physics shares them):
    D-grid halo strips (DGRID_NE) refreshed, then Earth-frame A-grid
    winds by the order-4 c2l per face and level, halo NaNs zeroed as the
    NumPy authority does.  Returns ``(u6, v6, ua6, va6)`` on ``(6, m, m,
    km)``; ``u6``/``v6`` are the exchanged D winds the dry twin must be
    handed (its contract, fv3_native_physics_coupling:750-799)."""
    import jax
    import jax.numpy as jnp
    from legoesm.grids.fv3_duo_halos import exchange_dgrid_vector_halos
    from legoesm.grids.fv3_native_ext_vector import c2l_ord4_face_jax

    n, ng, km = int(n), int(ng), int(km)
    m = n + 2 * ng
    u6 = jnp.asarray(state["u"])
    v6 = jnp.asarray(state["v"])
    if jnp.asarray(state["pt"]).shape != (6, m, m, km):
        raise ValueError(
            f"column_view_sixface_jax: pt {jnp.asarray(state['pt']).shape} "
            f"!= {(6, m, m, km)}")

    # PASS 0: D-grid u/v halo strips (DGRID_NE), one level at a time
    ex_d = jax.vmap(lambda u, v: exchange_dgrid_vector_halos(u, v, tab),
                    in_axes=(-1, -1), out_axes=(-1, -1))
    u6, v6 = ex_d(u6, v6)

    # PASS 1: Earth-frame A-grid winds (per face, per level)
    c2l_k = jax.vmap(lambda u, v, a: c2l_ord4_face_jax(u, v, a, n, ng),
                     in_axes=(-1, -1, None), out_axes=(-1, -1))
    c2l_6 = jax.vmap(c2l_k, in_axes=(0, 0, 0), out_axes=(0, 0))
    ua6, va6 = c2l_6(u6, v6, tuple(jnp.asarray(a) for a in amat6))
    # c2l leaves the halo NaN by contract; the authority's bare
    # np.nan_to_num(nan=0.0) also clamps +-inf to the float64 extrema, and
    # jnp.nan_to_num's defaults do the same -- port its exclusions verbatim.
    ua6 = jnp.nan_to_num(ua6, nan=0.0)
    va6 = jnp.nan_to_num(va6, nan=0.0)
    return u6, v6, ua6, va6


def apply_column_increments_sixface_jax(state, press, q, view, tab,
                                        wind_vectors6, u_dt_c, v_dt_c,
                                        t_dt_c, q_dt_c, *, dt, n, ng, km,
                                        ptop, akap, moist_cp=False):
    """fv_update_phys on six faces for COLUMN tendencies given on the
    compute window ``(6, n, n, km)`` (passes 2-3 of the Held-Suarez twin
    plus the moist scalar block of the Kessler bridge, so both and the
    column lane apply increments through ONE function):

    * ``q_dt_c`` non-empty (``{slot: (6, n, n, km)}``, slots 0..2 =
      sphum/liq_wat/rainwat only): tracers, layer mass and ``pt`` (on
      ``cvm``) through :func:`fv_update_phys_moist_duo_jax`, every other
      tracer renormalised to the new layer mass, pressures rebuilt with
      ``p_var_hydrostatic`` (``ptop``/``akap``) -- the Kessler bridge's
      block.  A tendency on a slot beyond the three warm-rain tracers is
      REFUSED: the ``nwat`` sum would miss its mass (an ice/snow deck
      needs the nwat=6 block).  Empty ``{}``: pressures and tracers
      untouched; ``pt += t_dt*dt`` on the compute window, or -- with
      ``moist_cp=True`` (a MOIST deck, ``FV3DuoConfig.moist``) -- the
      same block with zero water tendencies, i.e. ``pt += t_dt*dt*
      cp_air/cvm`` (fv_update_phys.F90:367-371 rescales EVERY physics
      heating by the moist heat capacity, water tendency or not; the
      layer mass is exactly unchanged since ``ps_dt == 1``).  So one deck
      applies heating one way, whichever tendencies physics returns.
      Heating convention: the moist block takes legoESM ``c_pd`` heating
      (what every legoESM physics returns) and rescales it to FV3's
      ``cp_air`` first (:func:`heating_to_fv3_cp_air`); the dry path
      applies ``t_dt_c`` as given (the closed lane's Held-Suarez is
      FV3's own hswf, already in its convention).
    * ``u_dt_c`` given: pad, one-ring A-grid exchange of ``u_dt``/``v_dt``
      (``tab``), then the dry twin (geographic A-grid increments -> D
      winds with ``wind_vectors6``) on ``view`` =
      :func:`column_view_sixface_jax`'s ``(u6, v6, ua6, va6)`` for the
      SAME ``state``.  ``None``: winds untouched (Kessler), and
      ``view``/``tab``/``wind_vectors6`` are not read.

    Returns ``(state_new, press_new, q_new)``; no input is mutated.
    """
    import jax
    import jax.numpy as jnp
    from legoesm.core.fv3_dynamics import p_var_hydrostatic
    from legoesm.grids.fv3_duo_halos import exchange_agrid_scalar_halos
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_C_LIQ, FV3_CP_AIR, FV3_CP_VAPOR)

    n, ng, km = int(n), int(ng), int(km)
    m = n + 2 * ng
    ci = slice(ng, ng + n)
    pt6 = jnp.asarray(state["pt"])
    delp6 = jnp.asarray(state["delp"])
    zeros = jnp.zeros((6, m, m, km), dtype=pt6.dtype)
    t_dt6 = zeros.at[:, ci, ci].set(t_dt_c)

    q_dt_c = dict(q_dt_c)
    water = bool(q_dt_c)
    moist = water or bool(moist_cp)
    if moist:
        bad = sorted(k for k in q_dt_c if k not in (0, 1, 2))
        if bad or len(q) < 3:
            raise ValueError(
                f"apply_column_increments_sixface_jax: water tendencies on "
                f"tracer slots {bad} (bundle carries {len(q)}); only the "
                f"warm-rain slots 0..2 [sphum, liq_wat, rainwat] enter the "
                f"nwat mass block -- an ice/snow deck needs the nwat=6 port")
        q3 = [jnp.asarray(q[i]) for i in range(3)]
        q_dt6 = [zeros.at[:, ci, ci].set(q_dt_c[i]) if i in q_dt_c else zeros
                 for i in range(3)]

        def _moist(pt, delp, qv, qc, qr, tdt, dqv, dqc, dqr):
            pt_n, delp_n, q_n, ps_dt = fv_update_phys_moist_duo_jax(
                pt, delp, [qv, qc, qr], tdt, [dqv, dqc, dqr], dt, n=n, ng=ng,
                cp_air=FV3_CP_AIR, cp_vapor=FV3_CP_VAPOR, c_liq=FV3_C_LIQ)
            return pt_n, delp_n, q_n[0], q_n[1], q_n[2], ps_dt
        # rescale on the compute block BEFORE padding (the bridge's op
        # order; rescaling the padded array lets XLA fold the factor into
        # the block's dt*cp_air chain and moves pt by ~1e-12 under jit)
        t_dt6_m = zeros.at[:, ci, ci].set(heating_to_fv3_cp_air(t_dt_c))
        pt6, delp_m, qv2, qc2, qr2, ps_dt6 = jax.vmap(_moist)(
            pt6, delp6, q3[0], q3[1], q3[2], t_dt6_m, *q_dt6)
        t_dt6 = zeros            # pt already advanced on cvm
    if water:
        delp6 = delp_m
        press_new = p_var_hydrostatic(delp6, ptop=ptop, akap=akap, n=n,
                                      ng=ng, km=km)
        q_new = list(q)
        q_new[0], q_new[1], q_new[2] = qv2, qc2, qr2
        # every OTHER mass tracer rides the same layer mass and is
        # renormalised with it (fv_update_phys.F90:349-357 adjusts all of
        # them; codex 2026-09-24: leaving a passenger's mixing ratio while
        # delp moves changes its mass without a source)
        for i in range(3, len(q_new)):
            qi = jnp.asarray(q_new[i])
            q_new[i] = qi.at[:, ci, ci].set(qi[:, ci, ci] / ps_dt6)
    else:
        # no water tendency: ps_dt == 1 exactly, so the layer mass, the
        # tracers and the step's own pressures pass through untouched
        press_new, q_new = press, q

    state_new = {**state, "pt": pt6}
    if water:
        state_new["delp"] = delp6
    if u_dt_c is None:
        if not moist:
            state_new["pt"] = pt6.at[:, ci, ci].set(
                pt6[:, ci, ci] + t_dt6[:, ci, ci] * dt)
        return state_new, press_new, q_new

    u6, v6, ua6, va6 = view
    u_dt6 = zeros.at[:, ci, ci].set(u_dt_c)
    v_dt6 = zeros.at[:, ci, ci].set(v_dt_c)

    # PASS 2: A-grid halo strips of the vector tendencies, per level
    ex_a = jax.vmap(lambda f: exchange_agrid_scalar_halos(f, tab),
                    in_axes=-1, out_axes=-1)
    u_dt6 = ex_a(u_dt6)
    v_dt6 = ex_a(v_dt6)

    # PASS 3: apply (per face)
    wv = {k: jnp.asarray(wind_vectors6[k]) for k in ("vlon", "vlat", "es1",
                                                     "ew2")}

    def _upd(u, v, pt, ua, va, u_dt, v_dt, t_dt, vlon, vlat, es1, ew2):
        u2, v2, pt2, _, _ = fv_update_phys_dry_duo_jax(
            u, v, pt, ua, va, u_dt, v_dt, t_dt, dt, vlon, vlat, es1, ew2, ng)
        return u2, v2, pt2
    u2, v2, pt2 = jax.vmap(_upd)(
        u6, v6, pt6, ua6, va6, u_dt6, v_dt6, t_dt6,
        wv["vlon"], wv["vlat"], wv["es1"], wv["ew2"])
    state_new.update(u=u2, v=v2, pt=pt2)
    return state_new, press_new, q_new


def apply_held_suarez_step_sixface_jax(state, press, tab, amat6, agrid_lat6,
                                       wind_vectors6, *, dt, n, ng, km,
                                       strat=True):
    """Face-stacked, pure-JAX twin of :func:`apply_held_suarez_step`.

    Same three passes on ``(6, ...)`` stacks (now the shared
    :func:`column_view_sixface_jax` / :func:`apply_column_increments_sixface_jax`
    pair around the Held-Suarez tendency); the NumPy path stays the
    authority.  Returns a new ``state`` dict with ``u``/``v``/``pt``
    replaced; inputs are never mutated.
    """
    import jax
    import jax.numpy as jnp

    n, ng, km = int(n), int(ng), int(km)
    ci = slice(ng, ng + n)
    view = column_view_sixface_jax(state, tab, amat6, n=n, ng=ng, km=km)
    _, _, ua6, va6 = view
    pt6 = jnp.asarray(state["pt"])
    delp6 = jnp.asarray(state["delp"])
    peln6 = jnp.transpose(jnp.asarray(press["peln"]), (0, 1, 3, 2))
    pe6 = jnp.transpose(jnp.asarray(press["pe"])[:, 1:n + 1, :, 1:n + 1],
                        (0, 1, 3, 2))
    pkz6 = jnp.asarray(press["pkz"])
    lat6 = jnp.asarray(agrid_lat6)[:, ci, ci]

    def _tend(pt, ua, va, delp, peln, pkz, pe, lat):
        return held_suarez_tend_jax(pt, ua, va, delp, peln, pkz, pe, lat, dt,
                                    strat=strat)
    t_dt_c, u_dt_c, v_dt_c = jax.vmap(_tend)(
        pt6[:, ci, ci], ua6[:, ci, ci], va6[:, ci, ci], delp6[:, ci, ci],
        peln6, pkz6, pe6, lat6)
    state_new, _, _ = apply_column_increments_sixface_jax(
        state, press, [], view, tab, wind_vectors6, u_dt_c, v_dt_c, t_dt_c,
        {}, dt=dt, n=n, ng=ng, km=km, ptop=None, akap=None)
    return state_new
