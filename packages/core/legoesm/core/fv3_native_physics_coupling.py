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
_HS_REF_RADIUS = 6.371e6  # H&S reference radius for the time-scale ratio [m]
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
    Returns ``(t_dt, u_dt, v_dt)`` each ``(n, n, npz)``.  NumPy authority; the
    strat teq recursion needs ``lax.scan`` for a JAX twin (deferred).
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
    rad_ratio = radius / _HS_REF_RADIUS
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
