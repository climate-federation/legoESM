"""Python transcription of CAM6 ``zm_conv.F90`` (CESM2.1) for oracle pins.

Loop-for-loop, 1-based-index transcription of ``zm_convr`` (with
``buoyan_dilute``/``parcel_dilute``, ``cldprp``, ``closure``, ``q1q2_pjr``),
``zm_conv_evap`` (+ ``cldfrc_fice``), ``momtran`` and ``convtran`` for ONE
column, with ``zmconv_microp = .false.`` and ``zm_org = .false.``.  Arrays are
allocated with a dummy index 0 so the Fortran indices can be used verbatim.

Independent of the JAX port except for the two shared physical inputs it
must agree on to be comparable at 1e-10: the saturation curve
(``legoesm.thermo.saturation_specific_humidity``, CLAUDE.md) and the constants
(``legoesm.constants``).  The entropy inversion here is scipy ``brentq``
(the oracle's Brent, run to 1e-13) so it is an independent root finder from
the port's Newton iteration.
"""

from __future__ import annotations

import math

import jax
import numpy as np
from scipy.optimize import brentq

from legoesm import constants
from legoesm.thermo import saturation_specific_humidity, saturation_vapor_pressure

_qsat_jit = jax.jit(saturation_specific_humidity)   # CAM qsat is SPECIFIC humidity: eps*es/(p-(1-eps)es)


def qsat_hpa(t, p_hpa):
    """``qsat_hPa`` -> qm (kg/kg); es is not needed by the callers here."""
    return float(_qsat_jit(np.float64(t), np.float64(p_hpa) * 100.0))


def qsat_pa(t, p_pa):
    return float(_qsat_jit(np.float64(t), np.float64(p_pa)))


# constants named as in the Fortran (values from legoesm.constants)
cpres = constants.c_pd
rgas = constants.R_d
grav = constants.g
rl = constants.L_v
latice = constants.L_f
cpliq = constants.c_pw
cpwv = constants.c_pv
tfreez = constants.T_freeze
rh2o = constants.R_v
eps1 = constants.epsilon
tiedke_add = 0.5
capelmt_default = 70.0
tau_default = 3600.0
lwmax = 1.0e-3


def entropy(TK, p, qtot):
    """zm_conv.F90:4566 ``entropy`` (p in hPa)."""
    L = rl - (cpliq - cpwv) * (TK - tfreez)
    qst = qsat_hpa(TK, p)
    qv = min(qtot, qst)
    e = qv * p / (eps1 + qv)
    # the port floors qv and (p-e) (1e-12, 1 Pa = 0.01 hPa) for AD safety; the
    # floors never bind on the physical columns used by the pins
    qv_safe = max(qv, 1.0e-12)
    p_dry = max(p - e, 0.01)
    return ((cpres + qtot * cpliq) * math.log(TK / tfreez)
            - rgas * math.log(p_dry / 1000.0) + L * qv / TK
            - qv * rh2o * math.log(qv_safe / max(qst, 1.0e-12)))


def ientropy(s, p, qt, Tfg):
    """zm_conv.F90:4590 ``ientropy`` (Brent), converged to 1e-13 K."""
    a, b = Tfg - 10.0, Tfg + 10.0
    fa, fb = entropy(a, p, qt) - s, entropy(b, p, qt) - s
    while fa * fb > 0.0 and b - a < 200.0:
        a -= 10.0
        b += 10.0
        fa, fb = entropy(a, p, qt) - s, entropy(b, p, qt) - s
    T = brentq(lambda x: entropy(x, p, qt) - s, a, b, xtol=1e-13, rtol=1e-15, maxiter=500)
    return T, qsat_hpa(T, p)


def _arr(n):
    return np.zeros(n + 2)


def buoyan_dilute(q, t, p, z, pf, pblt, tpert, pver, msg, num_cin, dmpdz):
    """zm_conv.F90:3970 ``buoyan_dilute`` + ``parcel_dilute`` for one column.

    Inputs are 1-based arrays (index 0 unused); ``pf`` has ``pver+1`` levels.
    Returns tp, qstp, tl, cape, lcl, lel, lon, mx.
    """
    lelten = [pver] * (num_cin + 1)
    capeten = [0.0] * (num_cin + 1)
    lon = pver
    knt = 0
    lel = pver
    mx = lon
    cape = 0.0
    hmax = 0.0
    tp = t.copy()
    qstp = q.copy()
    tv = _arr(pver)
    tpv = _arr(pver)
    buoy = _arr(pver)
    for k in range(1, pver + 1):
        tv[k] = t[k] * (1.0 + 1.608 * q[k]) / (1.0 + q[k])
        tpv[k] = tv[k]
    for k in range(pver, msg, -1):
        hmn = cpres * t[k] + grav * z[k] + rl * q[k]
        if k >= int(round(pblt)) and k <= lon and hmn > hmax:
            hmax = hmn
            mx = k
    lcl = mx
    tl = t[mx]
    pl = p[mx]

    # ---- parcel_dilute ----------------------------------------------------
    tmix, qtmix, qsmix, smix = _arr(pver), _arr(pver), _arr(pver), _arr(pver)
    xsh2o, ds_xsh2o, ds_freeze = _arr(pver), _arr(pver), _arr(pver)
    qtp0 = sp0 = mp0 = 0.0
    qtp = sp = mp = 0.0
    for k in range(pver, msg, -1):
        if k == mx:
            qtp0 = q[k]
            sp0 = entropy(t[k], p[k], qtp0)
            mp0 = 1.0
            smix[k] = sp0
            qtmix[k] = qtp0
            tmix[k], qsmix[k] = ientropy(smix[k], p[k], qtmix[k], t[k])
        if k < mx:
            dp = p[k] - p[k + 1]
            qtenv = 0.5 * (q[k] + q[k + 1])
            tenv = 0.5 * (t[k] + t[k + 1])
            penv = 0.5 * (p[k] + p[k + 1])
            senv = entropy(tenv, penv, qtenv)
            dpdz = -(penv * grav) / (rgas * tenv)
            dzdp = 1.0 / dpdz
            dmpdp = dmpdz * dzdp
            sp = sp - dmpdp * dp * senv
            qtp = qtp - dmpdp * dp * qtenv
            mp = mp - dmpdp * dp
            smix[k] = (sp0 + sp) / (mp0 + mp)
            qtmix[k] = (qtp0 + qtp) / (mp0 + mp)
            tmix[k], qsmix[k] = ientropy(smix[k], p[k], qtmix[k], tmix[k + 1])
            if qsmix[k] <= qtmix[k] and qsmix[k + 1] > qtmix[k + 1]:
                lcl = k
                qxsk = qtmix[k] - qsmix[k]
                qxskp1 = qtmix[k + 1] - qsmix[k + 1]
                dqxsdp = (qxsk - qxskp1) / dp
                pl = p[k + 1] - qxskp1 / dqxsdp
                dsdp = (smix[k] - smix[k + 1]) / dp
                dqtdp = (qtmix[k] - qtmix[k + 1]) / dp
                slcl = smix[k + 1] + dsdp * (pl - p[k + 1])
                qtlcl = qtmix[k + 1] + dqtdp * (pl - p[k + 1])
                tl, _ = ientropy(slcl, pl, qtlcl, tmix[k])
    for k in range(pver, msg, -1):
        if k == mx:
            tp[k] = tmix[k]
            qstp[k] = q[k]
            tpv[k] = (tp[k] + tpert) * (1.0 + 1.608 * qstp[k]) / (1.0 + qstp[k])
        if k < mx:
            new_q = qtmix[k]
            for _ in range(2):
                xsh2o[k] = max(0.0, qtmix[k] - qsmix[k] - lwmax)
                ds_xsh2o[k] = ds_xsh2o[k + 1] - cpliq * math.log(tmix[k] / tfreez) * max(
                    0.0, xsh2o[k] - xsh2o[k + 1])
                if tmix[k] <= tfreez and ds_freeze[k + 1] == 0.0:
                    ds_freeze[k] = (latice / tmix[k]) * max(0.0, qtmix[k] - qsmix[k] - xsh2o[k])
                if tmix[k] <= tfreez and ds_freeze[k + 1] != 0.0:
                    ds_freeze[k] = ds_freeze[k + 1] + (latice / tmix[k]) * max(
                        0.0, qsmix[k + 1] - qsmix[k])
                new_s = smix[k] + ds_xsh2o[k] + ds_freeze[k]
                new_q = qtmix[k] - xsh2o[k]
                tmix[k], qsmix[k] = ientropy(new_s, p[k], new_q, tmix[k])
            tp[k] = tmix[k]
            qstp[k] = qsmix[k] if new_q > qsmix[k] else new_q
            tpv[k] = (tp[k] + tpert) * (1.0 + 1.608 * qstp[k]) / (1.0 + new_q)

    # ---- buoyancy ---------------------------------------------------------
    plge600 = pl >= 600.0
    for k in range(pver, msg, -1):
        if k <= mx and plge600:
            tv[k] = t[k] * (1.0 + 1.608 * q[k]) / (1.0 + q[k])
            buoy[k] = tpv[k] - tv[k] + tiedke_add
        else:
            qstp[k] = q[k]
            tp[k] = t[k]
            tpv[k] = tv[k]
    for k in range(msg + 2, pver + 1):
        if k < lcl and plge600:
            if buoy[k + 1] > 0.0 and buoy[k] <= 0.0:
                knt = min(num_cin, knt + 1)
                lelten[knt] = k
    for n in range(1, num_cin + 1):
        for k in range(msg + 1, pver + 1):
            if plge600 and k <= mx and k > lelten[n]:
                capeten[n] += rgas * buoy[k] * math.log(pf[k + 1] / pf[k])
    for n in range(1, num_cin + 1):
        if capeten[n] > cape:
            cape = capeten[n]
            lel = lelten[n]
    cape = max(cape, 0.0)
    return tp, qstp, tl, cape, lcl, lel, lon, mx


def cldprp(q, t, p, z, s, zf, shat, qhat, jb, lel, jt_in, mx, pver, msg, limcnv,
           landfrac, c0_lnd, c0_ocn, alfa):
    """zm_conv.F90:2689 ``cldprp`` for one column (microp off)."""
    A = lambda: _arr(pver)
    k1, i2, i3, i4 = A(), A(), A(), A()
    ihat, idag, iprm = A(), A(), A()
    mu, f, eps, eu, du, ql, cu, evp, cmeg = A(), A(), A(), A(), A(), A(), A(), A(), A()
    qds, md, ed, sd, qd, mc, qu, su = A(), A(), A(), A(), A(), A(), A(), A()
    qst, gamma, hmn, hsat, hu, hd, rprd, qcde = A(), A(), A(), A(), A(), A(), A(), A()
    qsthat, hsthat, gamhat, dz = A(), A(), A(), A()
    pflx = np.zeros(pver + 3)
    c0mask = c0_ocn * (1.0 - landfrac) + c0_lnd * landfrac
    for k in range(1, pver + 1):
        dz[k] = zf[k] - zf[k + 1]
    for k in range(1, pver + 1):
        qds[k] = q[k]
        sd[k] = s[k]
        qd[k] = q[k]
        qu[k] = q[k]
        su[k] = s[k]
        qst[k] = qsat_hpa(t[k], p[k])
        if p[k] * 100.0 - float(saturation_vapor_pressure(t[k])) <= 0.0:
            qst[k] = 1.0
        gamma[k] = qst[k] * (1.0 + qst[k] / eps1) * eps1 * rl / (rgas * t[k] ** 2) * rl / cpres
        hmn[k] = cpres * t[k] + grav * z[k] + rl * q[k]
        hsat[k] = cpres * t[k] + grav * z[k] + rl * qst[k]
        hu[k] = hmn[k]
        hd[k] = hmn[k]
    for k in range(1, msg + 2):
        hsthat[k] = hsat[k]
        qsthat[k] = qst[k]
        gamhat[k] = gamma[k]
    totpcp = 0.0
    totevp = 0.0
    for k in range(msg + 2, pver + 1):
        if abs(qst[k - 1] - qst[k]) > 1.0e-6:
            qsthat[k] = math.log(qst[k - 1] / qst[k]) * qst[k - 1] * qst[k] / (qst[k - 1] - qst[k])
        else:
            qsthat[k] = qst[k]
        hsthat[k] = cpres * shat[k] + rl * qsthat[k]
        if abs(gamma[k - 1] - gamma[k]) > 1.0e-6:
            gamhat[k] = (math.log(gamma[k - 1] / gamma[k]) * gamma[k - 1] * gamma[k]
                         / (gamma[k - 1] - gamma[k]))
        else:
            gamhat[k] = gamma[k]
    jt = max(lel, limcnv + 1)
    jt = min(jt, pver)
    jd = pver
    jlcl = lel
    hmin = 1.0e6
    j0 = None
    for k in range(msg + 1, pver + 1):
        if hsat[k] <= hmin and k >= jt and k <= jb:
            hmin = hsat[k]
            j0 = k
    j0 = min(j0, jb - 2)
    j0 = max(j0, jt + 2)
    j0 = min(j0, pver)
    for k in range(msg + 1, pver + 1):
        if k >= jt and k <= jb:
            hu[k] = hmn[mx] + cpres * tiedke_add
            su[k] = s[mx] + tiedke_add
    for k in range(pver - 1, msg, -1):
        if k < jb and k >= jt:
            k1[k] = k1[k + 1] + (hmn[mx] - hmn[k]) * dz[k]
            ihat[k] = 0.5 * (k1[k + 1] + k1[k])
            i2[k] = i2[k + 1] + ihat[k] * dz[k]
            idag[k] = 0.5 * (i2[k + 1] + i2[k])
            i3[k] = i3[k + 1] + idag[k] * dz[k]
            iprm[k] = 0.5 * (i3[k + 1] + i3[k])
            i4[k] = i4[k + 1] + iprm[k] * dz[k]
    hmin = 1.0e6
    expdif = 0.0
    for k in range(msg + 1, pver + 1):
        if k >= j0 and k <= jb and hmn[k] <= hmin:
            hmin = hmn[k]
            expdif = hmn[mx] - hmin
    for k in range(msg + 2, pver + 1):
        expnum = 0.0
        ftemp = 0.0
        if k < jt or k >= jb:
            k1[k] = 0.0
            expnum = 0.0
        else:
            expnum = hmn[mx] - (hsat[k - 1] * (zf[k] - z[k]) + hsat[k] * (z[k - 1] - zf[k])) / (
                z[k - 1] - z[k])
        if (expdif > 100.0 and expnum > 0.0) and k1[k] > expnum * dz[k]:
            ftemp = expnum / k1[k]
            f[k] = (ftemp + i2[k] / k1[k] * ftemp ** 2
                    + (2.0 * i2[k] ** 2 - k1[k] * i3[k]) / k1[k] ** 2 * ftemp ** 3
                    + (-5.0 * k1[k] * i2[k] * i3[k] + 5.0 * i2[k] ** 3 + k1[k] ** 2 * i4[k])
                    / k1[k] ** 3 * ftemp ** 4)
            f[k] = max(f[k], 0.0)
            f[k] = min(f[k], 0.0002)
    if j0 < jb:
        if f[j0] < 1.0e-6 and f[j0 + 1] > f[j0]:
            j0 = j0 + 1
    for k in range(msg + 2, pver + 1):
        if k >= jt and k <= j0:
            f[k] = max(f[k], f[k - 1])
    eps0 = f[j0]
    eps[jb] = eps0
    for k in range(pver, msg, -1):
        if k >= j0 and k <= jb:
            eps[k] = f[j0]
    for k in range(pver, msg, -1):
        if k < j0 and k >= jt:
            eps[k] = f[k]
    # ---- updraft ----------------------------------------------------------
    if eps0 > 0.0:
        mu[jb] = 1.0
        eu[jb] = mu[jb] / dz[jb]
    tmplel = jt
    for k in range(pver, msg, -1):
        if eps0 > 0.0 and (k >= tmplel and k < jb):
            zuef = zf[k] - zf[jb]
            rmue = (1.0 / eps0) * (math.exp(eps[k + 1] * zuef) - 1.0) / zuef
            mu[k] = (1.0 / eps0) * (math.exp(eps[k] * zuef) - 1.0) / zuef
            eu[k] = (rmue - mu[k + 1]) / dz[k]
            du[k] = (rmue - mu[k]) / dz[k]
    khighest = lel
    klowest = jb
    for k in range(klowest - 1, khighest - 1, -1):
        if k <= jb - 1 and k >= lel and eps0 > 0.0:
            if mu[k] < 0.02:
                hu[k] = hmn[k]
                mu[k] = 0.0
                eu[k] = 0.0
                du[k] = mu[k + 1] / dz[k]
            else:
                hu[k] = mu[k + 1] / mu[k] * hu[k + 1] + dz[k] / mu[k] * (
                    eu[k] * hmn[k] - du[k] * hsat[k])
    doit = True
    for k in range(klowest - 2, khighest - 2, -1):
        if doit and k <= jb - 2 and k >= lel - 1:
            if hu[k] <= hsthat[k] and hu[k + 1] > hsthat[k + 1] and mu[k] >= 0.02:
                if hu[k] - hsthat[k] < -2000.0:
                    jt = k + 1
                    doit = False
                else:
                    jt = k
                    doit = False
            elif hu[k] > hu[jb] or mu[k] < 0.02:
                jt = k + 1
                doit = False
    for k in range(pver, msg, -1):
        if k >= lel and k <= jt and eps0 > 0.0:
            mu[k] = 0.0
            eu[k] = 0.0
            du[k] = 0.0
            hu[k] = hmn[k]
        if k == jt and eps0 > 0.0:
            du[k] = mu[k + 1] / dz[k]
            eu[k] = 0.0
            mu[k] = 0.0
    done = False
    for k in range(pver, msg + 1, -1):
        if k == jb and eps0 > 0.0:
            qu[k] = q[mx]
            su[k] = (hu[k] - rl * qu[k]) / cpres
        if (not done and k > jt and k < jb) and eps0 > 0.0:
            su[k] = mu[k + 1] / mu[k] * su[k + 1] + dz[k] / mu[k] * (eu[k] - du[k]) * s[k]
            qu[k] = mu[k + 1] / mu[k] * qu[k + 1] + dz[k] / mu[k] * (
                eu[k] * q[k] - du[k] * qst[k])
            tu = su[k] - grav / cpres * zf[k]
            qstu = qsat_hpa(tu, (p[k] + p[k - 1]) / 2.0)
            if qu[k] >= qstu:
                jlcl = k
                done = True
    for k in range(msg + 2, pver + 1):
        if (k > jt and k <= jlcl) and eps0 > 0.0:
            su[k] = shat[k] + (hu[k] - hsthat[k]) / (cpres * (1.0 + gamhat[k]))
            qu[k] = qsthat[k] + gamhat[k] * (hu[k] - hsthat[k]) / (rl * (1.0 + gamhat[k]))
    tmplel = jb
    for k in range(pver, msg + 1, -1):
        if k >= jt and k < tmplel and eps0 > 0.0:
            cu[k] = ((mu[k] * su[k] - mu[k + 1] * su[k + 1]) / dz[k]
                     - (eu[k] - du[k]) * s[k]) / (rl / cpres)
            if k == jt:
                cu[k] = 0.0
            cu[k] = max(0.0, cu[k])
    for k in range(pver, msg + 1, -1):
        rprd[k] = 0.0
        if k >= jt and k < jb and eps0 > 0.0 and mu[k] >= 0.0:
            if mu[k] > 0.0:
                ql1 = 1.0 / mu[k] * (mu[k + 1] * ql[k + 1] - dz[k] * du[k] * ql[k + 1]
                                     + dz[k] * cu[k])
                ql[k] = ql1 / (1.0 + dz[k] * c0mask)
            else:
                ql[k] = 0.0
            totpcp = totpcp + dz[k] * (cu[k] - du[k] * ql[k + 1])
            rprd[k] = c0mask * mu[k] * ql[k]
            qcde[k] = ql[k]
    # ---- downdraft --------------------------------------------------------
    jt = min(jt, jb - 1)
    jd = max(j0, jt + 1)
    jd = min(jd, jb)
    hd[jd] = hmn[jd - 1]
    epsm = eps0
    if jd < jb and eps0 > 0.0:
        md[jd] = -alfa * epsm / eps0
    for k in range(msg + 1, pver + 1):
        if (k > jd and k <= jb) and eps0 > 0.0:
            zdef = zf[jd] - zf[k]
            md[k] = -alfa / (2.0 * eps0) * (math.exp(2.0 * epsm * zdef) - 1.0) / zdef
    for k in range(msg + 1, pver + 1):
        if (k >= jt and k <= jb) and eps0 > 0.0 and jd < jb:
            ratmjb = min(abs(mu[jb] / md[jb]), 1.0)
            md[k] = md[k] * ratmjb
    small = 1.0e-20
    for k in range(msg + 1, pver + 1):
        if (k >= jt and k <= pver) and eps0 > 0.0:
            ed[k - 1] = (md[k - 1] - md[k]) / dz[k - 1]
            mdt = min(md[k], -small)
            hd[k] = (md[k - 1] * hd[k - 1] - dz[k - 1] * ed[k - 1] * hmn[k - 1]) / mdt
    for k in range(msg + 2, pver + 1):
        if (k >= jd and k <= jb) and eps0 > 0.0 and jd < jb:
            qds[k] = qsthat[k] + gamhat[k] * (hd[k] - hsthat[k]) / (rl * (1.0 + gamhat[k]))
    qd[jd] = qds[jd]
    sd[jd] = (hd[jd] - rl * qd[jd]) / cpres
    for k in range(msg + 2, pver + 1):
        if k >= jd and k < jb and eps0 > 0.0:
            qd[k + 1] = qds[k + 1]
            evp[k] = -ed[k] * q[k] + (md[k] * qd[k] - md[k + 1] * qd[k + 1]) / dz[k]
            evp[k] = max(evp[k], 0.0)
            mdt = min(md[k + 1], -small)
            sd[k + 1] = ((rl / cpres * evp[k] - ed[k] * s[k]) * dz[k] + md[k] * sd[k]) / mdt
            totevp = totevp - dz[k] * ed[k] * q[k]
    totevp = totevp + md[jd] * qd[jd] - md[jb] * qd[jb]
    totpcp = max(totpcp, 0.0)
    totevp = max(totevp, 0.0)
    for k in range(msg + 2, pver + 1):
        if totevp > 0.0 and totpcp > 0.0:
            fac = min(1.0, totpcp / (totevp + totpcp))
            md[k] = md[k] * fac
            ed[k] = ed[k] * fac
            evp[k] = evp[k] * fac
        else:
            md[k] = 0.0
            ed[k] = 0.0
            evp[k] = 0.0
        cmeg[k] = cu[k] - evp[k]
        rprd[k] = rprd[k] - evp[k]
    pflx[1] = 0.0
    for k in range(2, pver + 2):
        pflx[k] = pflx[k - 1] + rprd[k - 1] * dz[k - 1]
    for k in range(msg + 1, pver + 1):
        mc[k] = mu[k] + md[k]
    return dict(mu=mu, eu=eu, du=du, md=md, ed=ed, sd=sd, qd=qd, mc=mc, qu=qu, su=su,
                qst=qst, hmn=hmn, hsat=hsat, ql=ql, cmeg=cmeg, jt=jt, jlcl=jlcl, j0=j0,
                jd=jd, pflx=pflx, evp=evp, cu=cu, rprd=rprd, qcde=qcde, eps0=eps0)


def closure(q, t, p, z, s, tp, qs, qu, su, mc, du, mu, md, qd, sd, qhat, shat, dp,
            qstp, zf, ql, dsubcld, cape, tl, lcl, lel, jt, mx, pver, msg, capelmt, tau):
    """zm_conv.F90:3609 ``closure`` for one column -> mb."""
    A = lambda: _arr(pver)
    dtmdt, dqmdt, dboydt, thetavp, thetavm, dtpdt, dqsdtp = A(), A(), A(), A(), A(), A(), A()
    mb = 0.0
    eb = p[mx] * q[mx] / (eps1 + q[mx])
    dtbdt = (1.0 / dsubcld) * (mu[mx] * (shat[mx] - su[mx]) + md[mx] * (shat[mx] - sd[mx]))
    dqbdt = (1.0 / dsubcld) * (mu[mx] * (qhat[mx] - qu[mx]) + md[mx] * (qhat[mx] - qd[mx]))
    debdt = eps1 * p[mx] / (eps1 + q[mx]) ** 2 * dqbdt
    dtldt = -2840.0 * (3.5 / t[mx] * dtbdt - debdt / eb) / (
        3.5 * math.log(t[mx]) - math.log(eb) - 4.805) ** 2
    for k in range(msg + 1, pver):
        if k == jt:
            dtmdt[k] = (1.0 / dp[k]) * (mu[k + 1] * (su[k + 1] - shat[k + 1] - rl / cpres * ql[k + 1])
                                        + md[k + 1] * (sd[k + 1] - shat[k + 1]))
            dqmdt[k] = (1.0 / dp[k]) * (mu[k + 1] * (qu[k + 1] - qhat[k + 1] + ql[k + 1])
                                        + md[k + 1] * (qd[k + 1] - qhat[k + 1]))
    beta = 0.0
    for k in range(msg + 1, pver):
        if k > jt and k < mx:
            dtmdt[k] = ((mc[k] * (shat[k] - s[k]) + mc[k + 1] * (s[k] - shat[k + 1])) / dp[k]
                        - rl / cpres * du[k] * (beta * ql[k] + (1 - beta) * ql[k + 1]))
            dqmdt[k] = ((mu[k + 1] * (qu[k + 1] - qhat[k + 1] + cpres / rl * (su[k + 1] - s[k]))
                         - mu[k] * (qu[k] - qhat[k] + cpres / rl * (su[k] - s[k]))
                         + md[k + 1] * (qd[k + 1] - qhat[k + 1] + cpres / rl * (sd[k + 1] - s[k]))
                         - md[k] * (qd[k] - qhat[k] + cpres / rl * (sd[k] - s[k]))) / dp[k]
                        + du[k] * (beta * ql[k] + (1 - beta) * ql[k + 1]))
    for k in range(msg + 1, pver + 1):
        if k >= lel and k <= lcl:
            thetavp[k] = tp[k] * (1000.0 / p[k]) ** (rgas / cpres) * (1.0 + 1.608 * qstp[k] - q[mx])
            thetavm[k] = t[k] * (1000.0 / p[k]) ** (rgas / cpres) * (1.0 + 0.608 * q[k])
            dqsdtp[k] = qstp[k] * (1.0 + qstp[k] / eps1) * eps1 * rl / (rgas * tp[k] ** 2)
            dtpdt[k] = tp[k] / (1.0 + rl / cpres * (dqsdtp[k] - qstp[k] / tp[k])) * (
                dtbdt / t[mx] + rl / cpres * (dqbdt / tl - q[mx] / tl ** 2 * dtldt))
            dboydt[k] = ((dtpdt[k] / tp[k] + 1.0 / (1.0 + 1.608 * qstp[k] - q[mx])
                          * (1.608 * dqsdtp[k] * dtpdt[k] - dqbdt))
                         - (dtmdt[k] / t[k] + 0.608 / (1.0 + 0.608 * q[k]) * dqmdt[k])
                         ) * grav * thetavp[k] / thetavm[k]
    for k in range(msg + 1, pver + 1):
        if k > lcl and k < mx:
            thetavp[k] = tp[k] * (1000.0 / p[k]) ** (rgas / cpres) * (1.0 + 0.608 * q[mx])
            thetavm[k] = t[k] * (1000.0 / p[k]) ** (rgas / cpres) * (1.0 + 0.608 * q[k])
            dboydt[k] = (dtbdt / t[mx] + 0.608 / (1.0 + 0.608 * q[mx]) * dqbdt
                         - dtmdt[k] / t[k] - 0.608 / (1.0 + 0.608 * q[k]) * dqmdt[k]
                         ) * grav * thetavp[k] / thetavm[k]
    dadt = 0.0
    for k in range(lel, mx):
        dadt += dboydt[k] * (zf[k] - zf[k + 1])
    dltaa = -1.0 * (cape - capelmt)
    if dadt != 0.0:
        mb = max(dltaa / tau / dadt, 0.0)
    return mb


def q1q2_pjr(q, qs, qu, su, du, qhat, shat, dp, mu, md, sd, qd, ql, dsubcld, jt, mx,
             pver, msg, evp, cu):
    """zm_conv.F90:3821 ``q1q2_pjr`` for one column."""
    dsdt, dqdt, dl = _arr(pver), _arr(pver), _arr(pver)
    ktm = jt
    kbm = mx
    for k in range(ktm, pver):
        emc = -cu[k] + evp[k]
        dsdt[k] = -rl / cpres * emc + (mu[k + 1] * (su[k + 1] - shat[k + 1])
                                       - mu[k] * (su[k] - shat[k])
                                       + md[k + 1] * (sd[k + 1] - shat[k + 1])
                                       - md[k] * (sd[k] - shat[k])) / dp[k]
        dqdt[k] = emc + (mu[k + 1] * (qu[k + 1] - qhat[k + 1])
                         - mu[k] * (qu[k] - qhat[k])
                         + md[k + 1] * (qd[k + 1] - qhat[k + 1])
                         - md[k] * (qd[k] - qhat[k])) / dp[k]
        dl[k] = du[k] * ql[k + 1]
    for k in range(kbm, pver + 1):
        if k == mx:
            dsdt[k] = (1.0 / dsubcld) * (-mu[k] * (su[k] - shat[k]) - md[k] * (sd[k] - shat[k]))
            dqdt[k] = (1.0 / dsubcld) * (-mu[k] * (qu[k] - qhat[k]) - md[k] * (qd[k] - qhat[k]))
        elif k > mx:
            dsdt[k] = dsdt[k - 1]
            dqdt[k] = dqdt[k - 1]
    return dqdt, dsdt, dl


def limcnv_from_pref_edge(pref_edge, pver):
    """zm_conv_intr.F90:334-345 (pref_edge 1-based, Pa)."""
    if pref_edge[1] >= 4.0e3:
        return 1
    for k in range(1, pver + 1):
        if pref_edge[k] < 4.0e3 and pref_edge[k + 1] >= 4.0e3:
            return k
    return pver + 1


def zm_convr(t0, qh0, pap0, paph0, zm0, zi0, landfrac, delt, pblh, tpert, *,
             capelmt=capelmt_default, tau=tau_default, num_cin=1, dmpdz=-1.0e-3,
             c0_lnd=0.0075, c0_ocn=0.03, alfa=0.1, pbl_top_pa=None, pref_edge=None):
    """zm_conv.F90:142 ``zm_convr`` for ONE column (0-based surface-last inputs).

    ``delt`` is the oracle's half step (``0.5*ztodt``).  ``pblh`` None ->
    the port's departure (launch bounded by ``pbl_top_pa``).  Returns a dict
    of 0-based arrays.
    """
    pver = len(t0)
    to1 = lambda a: np.concatenate([[0.0], np.asarray(a, dtype=np.float64)])
    t, qh, pap, zm = to1(t0), to1(qh0), to1(pap0), to1(zm0)
    paph, zi = to1(paph0), to1(zi0)
    dpp = _arr(pver)
    for k in range(1, pver + 1):
        dpp[k] = paph[k + 1] - paph[k]
    # zm_conv_init: limcnv from the REFERENCE interfaces (pref_edge); the
    # column's own interfaces only when the host has no reference coordinate.
    limcnv = limcnv_from_pref_edge(paph if pref_edge is None else to1(pref_edge), pver)
    msg = limcnv - 1
    zs = 0.0
    p, z, s, pf, zf = _arr(pver), _arr(pver), _arr(pver), _arr(pver + 1), _arr(pver + 1)
    pf[pver + 1] = paph[pver + 1] * 0.01
    zf[pver + 1] = zi[pver + 1] + zs
    for k in range(1, pver + 1):
        p[k] = pap[k] * 0.01
        pf[k] = paph[k] * 0.01
        z[k] = zm[k] + zs
        zf[k] = zi[k] + zs
    if pblh is None:
        pblt = pver
        for k in range(pver, msg, -1):
            if pap[k] >= pbl_top_pa:
                pblt = k
    else:
        pblt = pver
        for k in range(pver - 1, msg, -1):
            if abs(z[k] - zs - pblh) < (zf[k] - zf[k + 1]) * 0.5:
                pblt = k
    q = qh.copy()
    shat, qhat = _arr(pver), _arr(pver)
    for k in range(1, pver + 1):
        s[k] = t[k] + (grav / cpres) * z[k]
        shat[k] = s[k]
        qhat[k] = q[k]
    tp, qstp, tl, cape, lcl, lel, lon, maxi = buoyan_dilute(
        q, t, p, z, pf, pblt, tpert, pver, msg, num_cin, dmpdz)
    out = dict(cape=cape, ideep=cape > capelmt, limcnv=limcnv, msg=msg, mx=maxi, lcl=lcl,
               lel=lel, tl=tl, tp=tp[1:], qstp=qstp[1:])
    zero = np.zeros(pver)
    if not out["ideep"]:
        out.update(dqdt=zero, heat=zero, dlf=zero, rprd=zero, prec=0.0, mb=0.0, mu=zero,
                   md=zero, du=zero, eu=zero, ed=zero, dp=0.01 * dpp[1:], dsubcld=0.0,
                   jt=None, pflx=np.zeros(pver + 1))
        return out
    dp = _arr(pver)
    for k in range(1, pver + 1):
        dp[k] = 0.01 * dpp[k]
    dsubcld = 0.0
    for k in range(msg + 1, pver + 1):
        if k >= maxi:
            dsubcld += dp[k]
    for k in range(msg + 2, pver + 1):
        sdifr = 0.0
        qdifr = 0.0
        if s[k] > 0.0 or s[k - 1] > 0.0:
            sdifr = abs((s[k] - s[k - 1]) / max(s[k - 1], s[k]))
        if q[k] > 0.0 or q[k - 1] > 0.0:
            qdifr = abs((q[k] - q[k - 1]) / max(q[k - 1], q[k]))
        if sdifr > 1.0e-6:
            shat[k] = math.log(s[k - 1] / s[k]) * s[k - 1] * s[k] / (s[k - 1] - s[k])
        else:
            shat[k] = 0.5 * (s[k] + s[k - 1])
        if qdifr > 1.0e-6:
            qhat[k] = math.log(q[k - 1] / q[k]) * q[k - 1] * q[k] / (q[k - 1] - q[k])
        else:
            qhat[k] = 0.5 * (q[k] + q[k - 1])
    c = cldprp(q, t, p, z, s, zf, shat, qhat, maxi, lel, None, maxi, pver, msg, limcnv,
               landfrac, c0_lnd, c0_ocn, alfa)
    jt = c["jt"]
    mu, eu, du, md, ed = c["mu"], c["eu"], c["du"], c["md"], c["ed"]
    cug, cmeg, rprdg, evpg = c["cu"], c["cmeg"], c["rprd"], c["evp"]
    for k in range(msg + 1, pver + 1):
        fac = (zf[k] - zf[k + 1]) / dp[k]
        du[k] *= fac
        eu[k] *= fac
        ed[k] *= fac
        cug[k] *= fac
        cmeg[k] *= fac
        rprdg[k] *= fac
        evpg[k] *= fac
    mb = closure(q, t, p, z, s, tp, c["qst"], c["qu"], c["su"], c["mc"], du, mu, md, c["qd"],
                 c["sd"], qhat, shat, dp, qstp, zf, c["ql"], dsubcld, cape, tl, lcl, lel, jt,
                 maxi, pver, msg, capelmt, tau)
    mumax = 0.0
    for k in range(msg + 2, pver + 1):
        mumax = max(mumax, mu[k] / dp[k])
    if mumax > 0.0:
        mb = min(mb, 0.5 / (delt * mumax))
    else:
        mb = 0.0
    mc = c["mc"]
    pflxg = c["pflx"]
    for k in range(msg + 1, pver + 1):
        mu[k] *= mb
        md[k] *= mb
        mc[k] *= mb
        du[k] *= mb
        eu[k] *= mb
        ed[k] *= mb
        cmeg[k] *= mb
        rprdg[k] *= mb
        cug[k] *= mb
        evpg[k] *= mb
        pflxg[k + 1] = pflxg[k + 1] * mb * 100.0 / grav
    dqdt, dsdt, dlg = q1q2_pjr(q, c["qst"], c["qu"], c["su"], du, qhat, shat, dp, mu, md,
                               c["sd"], c["qd"], c["qcde"], dsubcld, jt, maxi, pver, msg,
                               evpg, cug)
    heat = _arr(pver)
    qn = _arr(pver)
    for k in range(msg + 1, pver + 1):
        qn[k] = qh[k] + 2.0 * delt * dqdt[k]
        heat[k] = dsdt[k] * cpres
    prec = 0.0
    for k in range(pver, msg, -1):
        prec = prec - dpp[k] * (qn[k] - qh[k]) - dpp[k] * dlg[k] * 2.0 * delt
    prec = (1.0 / grav) * max(prec, 0.0) / (2.0 * delt) / 1000.0   # m/s
    out.update(dqdt=dqdt[1:pver + 1], heat=heat[1:pver + 1], dlf=dlg[1:pver + 1],
               rprd=rprdg[1:pver + 1], prec=prec * 1000.0, mb=mb, mu=mu[1:pver + 1],
               md=md[1:pver + 1], du=du[1:pver + 1], eu=eu[1:pver + 1], ed=ed[1:pver + 1],
               dp=dp[1:pver + 1], dsubcld=dsubcld, jt=jt, pflx=pflxg[1:pver + 2],
               mc=mc[1:pver + 1], cmeg=cmeg[1:pver + 1], ql=c["ql"][1:pver + 1],
               jlcl=c["jlcl"], j0=c["j0"], jd=c["jd"], eps0=c["eps0"])
    return out


def cldfrc_fice(t):
    tmax_fice = tfreez - 10.0
    tmin_fice = tmax_fice - 30.0
    tmax_fsnow = tfreez
    tmin_fsnow = tfreez - 5.0
    if t > tmax_fice:
        fice = 0.0
    elif t < tmin_fice:
        fice = 1.0
    else:
        fice = (tmax_fice - t) / (tmax_fice - tmin_fice)
    if t > tmax_fsnow:
        fsnow = 0.0
    elif t < tmin_fsnow:
        fsnow = 1.0
    else:
        fsnow = (tmax_fsnow - t) / (tmax_fsnow - tmin_fsnow)
    return fice, fsnow


def zm_conv_evap(t0, pmid0, pdel0, q0, prdprec0, cldfrc0, deltat, prec_in, ke):
    """zm_conv.F90:1396 ``zm_conv_evap`` (old_snow) for one column; prec in kg/m2/s."""
    pver = len(t0)
    to1 = lambda a: np.concatenate([[0.0], np.asarray(a, dtype=np.float64)])
    t, pmid, pdel, q, prdprec, cldfrc = (to1(t0), to1(pmid0), to1(pdel0), to1(q0),
                                         to1(prdprec0), to1(cldfrc0))
    prec = prec_in
    flxprec, flxsnow = _arr(pver + 1), _arr(pver + 1)
    ntprprd, ntsnprd, tend_s, tend_q = _arr(pver), _arr(pver), _arr(pver), _arr(pver)
    evpvint = 0.0
    for k in range(1, pver + 1):
        qs = qsat_pa(t[k], pmid[k])
        fice, fsnow_conv = cldfrc_fice(t[k])
        if t[k] > tfreez:
            flxsntm = 0.0
            snowmlt = flxsnow[k] * grav / pdel[k]
        else:
            flxsntm = flxsnow[k]
            snowmlt = 0.0
        evplimit = max(1.0 - q[k] / qs, 0.0)
        kemask = ke
        evpprec = kemask * (1.0 - cldfrc[k]) * evplimit * math.sqrt(flxprec[k])
        evplimit = max(0.0, (qs - q[k]) / deltat)
        evplimit = min(evplimit, flxprec[k] * grav / pdel[k])
        evplimit = min(evplimit, (prec - evpvint) * grav / pdel[k])
        evpprec = min(evplimit, evpprec)
        if flxprec[k] > 0.0:
            work1 = min(max(0.0, flxsntm / flxprec[k]), 1.0)
            evpsnow = evpprec * work1
        else:
            evpsnow = 0.0
        evpvint = evpvint + evpprec * pdel[k] / grav
        ntprprd[k] = prdprec[k] - evpprec
        if flxprec[k] > 0.0:
            work1 = min(max(0.0, flxsnow[k] / flxprec[k]), 1.0)
        else:
            work1 = 0.0
        work2 = max(fsnow_conv, work1)
        if snowmlt > 0.0:
            work2 = 0.0
        ntsnprd[k] = prdprec[k] * work2 - evpsnow - snowmlt
        flxprec[k + 1] = flxprec[k] + ntprprd[k] * pdel[k] / grav
        flxsnow[k + 1] = flxsnow[k] + ntsnprd[k] * pdel[k] / grav
        flxprec[k + 1] = max(flxprec[k + 1], 0.0)
        flxsnow[k + 1] = max(flxsnow[k + 1], 0.0)
        tend_s[k] = -evpprec * rl + ntsnprd[k] * latice
        tend_q[k] = evpprec
    return dict(tend_s=tend_s[1:pver + 1], tend_q=tend_q[1:pver + 1],
                prec=flxprec[pver + 1], snow=flxsnow[pver + 1],
                ntprprd=ntprprd[1:pver + 1], ntsnprd=ntsnprd[1:pver + 1],
                flxprec=flxprec[1:pver + 2], flxsnow=flxsnow[1:pver + 2])


def momtran(u0, v0, mu0, md0, du0, eu0, ed0, dp0, jt, mx, dt, momcu, momcd):
    """zm_conv.F90:1982 ``momtran`` for one column -> (dudt, dvdt, seten)."""
    pver = len(u0)
    to1 = lambda a: np.concatenate([[0.0], np.asarray(a, dtype=np.float64)])
    mu, md, du, eu, ed, dp = to1(mu0), to1(md0), to1(du0), to1(eu0), to1(ed0), to1(dp0)
    winds = [None, to1(u0), to1(v0)]
    mbsth = 1.0e-15
    ktm = jt
    dqdt = [None, _arr(pver), _arr(pver)]
    mflux = [None, np.zeros(pver + 3), np.zeros(pver + 3)]
    wind0 = [None, _arr(pver), _arr(pver)]
    windf = [None, _arr(pver), _arr(pver)]
    for m in (1, 2):
        const = winds[m].copy()
        for k in range(1, pver + 1):
            wind0[m][k] = const[k]
        chat, conu, cond, dcondt = _arr(pver), _arr(pver), _arr(pver), _arr(pver)
        pgu, pgd = _arr(pver), _arr(pver)
        for k in range(1, pver + 1):
            km1 = max(1, k - 1)
            chat[k] = 0.5 * (const[k] + const[km1])
            conu[k] = chat[k]
            cond[k] = chat[k]
        pgu[1] = 0.0
        pgd[1] = 0.0
        for k in range(2, pver):
            km1 = max(1, k - 1)
            kp1 = min(pver, k + 1)
            mududp = (mu[k] * (const[k] - const[km1]) / dp[km1]
                      + mu[kp1] * (const[kp1] - const[k]) / dp[k])
            pgu[k] = -momcu * 0.5 * mududp
            mddudp = (md[k] * (const[k] - const[km1]) / dp[km1]
                      + md[kp1] * (const[kp1] - const[k]) / dp[k])
            pgd[k] = -momcd * 0.5 * mddudp
        k = pver
        km1 = max(1, k - 1)
        mududp = mu[k] * (const[k] - const[km1]) / dp[km1]
        pgu[k] = -momcu * mududp
        mddudp = md[k] * (const[k] - const[km1]) / dp[km1]
        pgd[k] = -momcd * mddudp
        k = 2
        km1 = 1
        kk = pver
        mupdudp = mu[kk] + du[kk] * dp[kk]
        if mupdudp > mbsth:
            conu[kk] = (eu[kk] * const[kk] * dp[kk] + pgu[kk] * dp[kk]) / mupdudp
        if md[k] < -mbsth:
            cond[k] = (-ed[km1] * const[km1] * dp[km1]) - pgd[km1] * dp[km1] / md[k]
        for kk in range(pver - 1, 0, -1):
            kkp1 = min(pver, kk + 1)
            mupdudp = mu[kk] + du[kk] * dp[kk]
            if mupdudp > mbsth:
                conu[kk] = (mu[kkp1] * conu[kkp1] + eu[kk] * const[kk] * dp[kk]
                            + pgu[kk] * dp[kk]) / mupdudp
        for k in range(3, pver + 1):
            km1 = max(1, k - 1)
            if md[k] < -mbsth:
                cond[k] = (md[km1] * cond[km1] - ed[km1] * const[km1] * dp[km1]
                           - pgd[km1] * dp[km1]) / md[k]
        for k in range(ktm, pver + 1):
            kp1 = min(pver, k + 1)
            dcondt[k] = (mu[kp1] * (conu[kp1] - chat[kp1]) - mu[k] * (conu[k] - chat[k])
                         + md[kp1] * (cond[kp1] - chat[kp1]) - md[k] * (cond[k] - chat[k])
                         ) / dp[k]
        for k in range(mx, pver + 1):
            if k == mx:
                dcondt[k] = (1.0 / dp[k]) * (-mu[k] * (conu[k] - chat[k])
                                             - md[k] * (cond[k] - chat[k]))
        for k in range(1, pver + 1):
            dqdt[m][k] = dcondt[k]
        for k in range(ktm, pver + 1):
            mflux[m][k] = -mu[k] * (conu[k] - chat[k]) - md[k] * (cond[k] - chat[k])
        for k in range(ktm, pver + 1):
            windf[m][k] = const[k] - (mflux[m][k + 1] - mflux[m][k]) * dt / dp[k]
    seten = _arr(pver)
    for k in range(ktm, pver + 1):
        km1 = max(1, k - 1)
        kp1 = min(pver, k + 1)
        utop = (wind0[1][k] + wind0[1][km1]) / 2.0
        vtop = (wind0[2][k] + wind0[2][km1]) / 2.0
        ubot = (wind0[1][kp1] + wind0[1][k]) / 2.0
        vbot = (wind0[2][kp1] + wind0[2][k]) / 2.0
        fket = utop * mflux[1][k] + vtop * mflux[2][k]
        fkeb = ubot * mflux[1][k + 1] + vbot * mflux[2][k + 1]
        ketend_cons = (fket - fkeb) / dp[k]
        ketend = ((windf[1][k] ** 2 + windf[2][k] ** 2)
                  - (wind0[1][k] ** 2 + wind0[2][k] ** 2)) * 0.5 / dt
        seten[k] = ketend_cons - ketend
    return dqdt[1][1:pver + 1], dqdt[2][1:pver + 1], seten[1:pver + 1]


def convtran(c0, mu0, md0, du0, eu0, ed0, dp0, jt, mx):
    """zm_conv.F90:1643 ``convtran`` for one moist tracer, one column."""
    pver = len(c0)
    to1 = lambda a: np.concatenate([[0.0], np.asarray(a, dtype=np.float64)])
    const, mu, md, du, eu, ed, dp = (to1(c0), to1(mu0), to1(md0), to1(du0), to1(eu0),
                                     to1(ed0), to1(dp0))
    small = 1.0e-36
    mbsth = 1.0e-15
    ktm = jt
    kbm = mx
    chat, conu, cond, dcondt = _arr(pver), _arr(pver), _arr(pver), _arr(pver)
    for k in range(1, pver + 1):
        km1 = max(1, k - 1)
        minc = min(const[km1], const[k])
        maxc = max(const[km1], const[k])
        if minc < 0:
            cdifr = 0.0
        else:
            cdifr = abs(const[k] - const[km1]) / max(maxc, small)
        if cdifr > 1.0e-6:
            cabv = max(const[km1], maxc * 1.0e-12)
            cbel = max(const[k], maxc * 1.0e-12)
            chat[k] = math.log(cabv / cbel) / (cabv - cbel) * cabv * cbel
        else:
            chat[k] = 0.5 * (const[k] + const[km1])
        conu[k] = chat[k]
        cond[k] = chat[k]
    k = 2
    km1 = 1
    kk = pver
    mupdudp = mu[kk] + du[kk] * dp[kk]
    if mupdudp > mbsth:
        conu[kk] = (eu[kk] * const[kk] * dp[kk]) / mupdudp
    if md[k] < -mbsth:
        cond[k] = (-ed[km1] * const[km1] * dp[km1]) / md[k]
    for kk in range(pver - 1, 0, -1):
        kkp1 = min(pver, kk + 1)
        mupdudp = mu[kk] + du[kk] * dp[kk]
        if mupdudp > mbsth:
            conu[kk] = (mu[kkp1] * conu[kkp1] + eu[kk] * const[kk] * dp[kk]) / mupdudp
    for k in range(3, pver + 1):
        km1 = max(1, k - 1)
        if md[k] < -mbsth:
            cond[k] = (md[km1] * cond[km1] - ed[km1] * const[km1] * dp[km1]) / md[k]
    for k in range(ktm, pver + 1):
        km1 = max(1, k - 1)
        kp1 = min(pver, k + 1)
        fluxin = (mu[kp1] * conu[kp1] + mu[k] * min(chat[k], const[km1])
                  - (md[k] * cond[k] + md[kp1] * min(chat[kp1], const[kp1])))
        fluxout = (mu[k] * conu[k] + mu[kp1] * min(chat[kp1], const[k])
                   - (md[kp1] * cond[kp1] + md[k] * min(chat[k], const[k])))
        netflux = fluxin - fluxout
        if abs(netflux) < max(fluxin, fluxout) * 1.0e-12:
            netflux = 0.0
        dcondt[k] = netflux / dp[k]
    for k in range(kbm, pver + 1):
        km1 = max(1, k - 1)
        if k == mx:
            fluxin = mu[k] * min(chat[k], const[km1]) - md[k] * cond[k]
            fluxout = mu[k] * conu[k] - md[k] * min(chat[k], const[k])
            netflux = fluxin - fluxout
            if abs(netflux) < max(fluxin, fluxout) * 1.0e-12:
                netflux = 0.0
            dcondt[k] = netflux / dp[k]
        elif k > mx:
            dcondt[k] = 0.0
    return dcondt[1:pver + 1]
