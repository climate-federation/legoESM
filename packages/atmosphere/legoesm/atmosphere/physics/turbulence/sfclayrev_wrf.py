"""WRF revised MM5 surface layer (``sf_sfclay_physics = 1``), JAX port.

Faithful port of WRF v4.8.0 (06d4240ae989cc3e50af412bb472df3d9048783c)
``phys/module_sf_sfclayrev.F::sfclayrev`` and ``phys/physics_mmm/sf_sfclayrev.F90``
(physics_mmm 550b5b494758ad2756394e93cb08b5709b8feb83) ``sf_sfclayrev_run`` /
``sf_sfclayrev_init`` / ``zolri`` / ``zolri2`` / the integrated similarity
functions and their look-up tables, matched to the running compiled Fortran
(``tests/atmosphere/hydrostatic/unit/test_sfclayrev_oracle.py``, 1e-10 fp64;
harness ``scripts/validate/oracle/sfclayrev/``).  Line numbers below refer to
``sf_sfclayrev.F90``.

Ported path = WRF's namelist defaults (R4 vote X2): ``isfflx = 1``,
``isftcflx = 0``, ``iz0tlnd = 0``, ``shalwater_z0 = 0``, ``kim_tofd =
.false.``, ``lakemask = 0``; the alternative roughness / thermal-roughness /
form-drag branches are NOT ported, and the diagnostic outputs ``ustm``, ``ck``,
``cka``, ``cd``, ``cda`` (unused by YSU) are not produced.  ``scm_force_flux``
(static) keeps the caller's ``hfx``/``qfx`` (WRF's own forced-flux switch; its
``lh`` is then unassigned in the oracle and is returned as NaN here).

The scheme is stateful (WRF carries these between time steps): ``ust`` (the new
value is the mean of the old one and the new estimate, l.780), ``mol`` (a
previously unstable column may not become stable, l.363), ``znt`` over water
(Charnock update, l.818-822, read next step), land ``qsfc`` (l.297), and the
previous ``hfx``/``qfx``/``pblh`` (land convective velocity, l.343-345).

Hard switches are the oracle's, as ``jnp.where`` (user-approved exception for
the R1-R4 oracle routing, 2026-11-03); unselected branches are evaluated on
sanitised operands only so their values and gradients stay finite.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants

__physics_contract__ = {
    "summary": (
        "WRF v4.8.0 revised MM5 surface layer (sfclayrev, sf_sfclay_physics=1) "
        "on its namelist-default path: bulk Richardson number, Monin-Obukhov "
        "stability (Cheng-Brutsaert stable / Fairall-type unstable integrated "
        "functions via the oracle's look-up tables), friction velocity, "
        "Charnock roughness over water, Fairall thermal/moisture roughness over "
        "water, exchange coefficients, surface sensible/latent fluxes, 2/10 m "
        "diagnostics. Oracle-matched to the running Fortran."
    ),
    "inputs": {
        "u1": "m/s", "v1": "m/s", "t1": "K", "qv1": "kg/kg", "p1": "Pa",
        "dz1": "m", "psfc": "Pa", "tsk": "K", "xland": "1 land / 2 water",
        "mavail": "1", "pblh": "m", "dx": "m",
        "ust": "m/s", "mol": "K", "znt": "m", "hfx": "W/m^2",
        "qfx": "kg/m^2/s", "qsfc": "kg/kg",
    },
    "outputs": {
        "ust": "m/s", "mol": "K", "znt": "m", "hfx": "W/m^2",
        "qfx": "kg/m^2/s", "lh": "W/m^2", "br": "1", "fm": "1", "fh": "1",
        "wspd": "m/s", "u10": "m/s", "v10": "m/s", "t2": "K", "q2": "kg/kg",
    },
    "sign_convention": (
        "hfx, qfx, lh positive UPWARD (surface to atmosphere): hfx = "
        "flhc*(theta_g - theta_1), qfx = flqc*(qsfc - q_1). br > 0 stable "
        "(theta_v increases upward), br < 0 unstable. z up; one level (the "
        "lowest model layer), za = dz1/2 above the surface."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "WRF v4.8.0 phys/physics_mmm/sf_sfclayrev.F90; Jimenez et al. (2012), "
        "Mon. Wea. Rev. 140, 898-918; Cheng & Brutsaert (2005), BLM 114, 519-538"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_sfclayrev_oracle.py: running "
        "WRF oracle on stable/neutral/unstable, land/water, calm-wind and "
        "carried-state (3-call) columns, 1e-10; jit parity; finite gradients."
    ),
}

# --- module parameters (l.11-15, l.205-207) ---
_VCONVC = 1.0            # convective velocity coefficient vconvc
_CZO = 0.0185            # Charnock coefficient czo
_XKA = 2.4e-5            # molecular diffusivity of water vapour xka [m2/s]
_PRT = 1.0               # turbulent Prandtl number prt
_SALINITY_FACTOR = 0.98  # saturation vapour pressure over salty water
# --- WRF saturation vapour pressure (module_model_constants SVP1-3) ---
_SVP1 = 0.6112           # [kPa]
_SVP2 = 17.67            # [-]
_SVP3 = 29.65            # [K]
_PA_PER_KPA = 1000.0     # psfc/1000. and pl = p/1000. (cb, l.234, l.264)
_CPM_QV_COEFF = 0.8      # cpm = cp*(1 + 0.8*qv) (l.303)
# --- literals inside sf_sfclayrev_run ---
_XLAND_WATER = 1.5       # water when xland - 1.5 >= 0
_ZL_LAND = 0.01          # land "roughness" in psiq / zl (l.333, l.410)
_Z2 = 2.0                # 2 m and 10 m diagnostic heights
_Z10 = 10.0
_VCONV_EXP = 0.33        # **.33 (l.346, l.358)
_VSGD_COEFF = 0.32       # Mahrt-Sun subgrid wind (l.358)
_VSGD_DX = 5000.0
_WSPD_MIN = 0.1          # amax1(wspd, 0.1) (l.360)
_BR_MAX = 250.0          # zolri clip of |br| (l.389)
_UST_ZOLRI_MIN = 0.001   # ust < 0.001: zol = br*gz1oz0 (l.397)
_PSI_LIMIT = 0.9         # psi <= 0.9*gz?oz0 in thin layers (l.483)
_VISC_A = 1.32           # visc = (1.32 + 0.009*(T - T_freeze))*1e-5 (l.524)
_VISC_B = 0.009
_VISC_SCALE = 1.0e-5
_Z0T_A = 5.5e-5          # z0t = 5.5e-5*restar**(-0.60), clipped (l.526-528)
_Z0T_EXP = -0.60
_Z0T_MAX = 1.0e-4
_Z0T_MIN = 2.0e-9
_UST_LAND_MIN = 0.001    # amax1(ust, 0.001) over land (l.796)
_CHARNOCK_VISC = 0.11 * 1.5e-5   # 0.11*1.5e-5/ust (l.818)
_ZNT_WATER_MAX = 2.85e-3
_DTTHX_MIN = 1.0e-5      # flhc = 0 when |theta_1 - theta_g| <= 1e-5 (l.868)
_REGIME_STABLE, _REGIME_NEUTRAL, _REGIME_UNSTABLE = 1.0, 3.0, 4.0
# --- zolri (l.957-989) ---
_ZOLRI_BRACKET = 5.0
_ZOLRI_TOL = 0.01
_ZOLRI_MAXITER = 10
# --- integrated similarity functions (l.1013-1060) ---
_PSIM_STAB_A = 6.1
_PSIM_STAB_P = 2.5
_PSIH_STAB_A = 5.3
_PSIH_STAB_P = 1.1
_PSI_UNST_K = 16.0
_PSIM_UNST_C = 10.0
_PSIH_UNST_C = 34.0
_PSI_UNST_EXP = 0.33
_PSI_UNST_15 = 1.5
# --- look-up tables (sf_sfclayrev_init, l.40-50) ---
_TAB_N = 1000
_TAB_STEP = 0.01
_TAB_SCALE = 100.0


def _psim_stable_full(z, xp=jnp):
    return -_PSIM_STAB_A * xp.log(z + (1 + z ** _PSIM_STAB_P) ** (1.0 / _PSIM_STAB_P))


def _psih_stable_full(z, xp=jnp):
    return -_PSIH_STAB_A * xp.log(z + (1 + z ** _PSIH_STAB_P) ** (1.0 / _PSIH_STAB_P))


def _psi_c(y, xp):
    s3 = xp.sqrt(3.0)
    return (_PSI_UNST_15 * xp.log((y ** 2.0 + y + 1.0) / 3.0)
            - s3 * xp.arctan((2.0 * y + 1) / s3) + 4.0 * xp.arctan(1.0) / s3)


def _psim_unstable_full(z, xp=jnp):
    x = (1.0 - _PSI_UNST_K * z) ** 0.25
    psimk = (2 * xp.log(0.5 * (1 + x)) + xp.log(0.5 * (1 + x * x))
             - 2.0 * xp.arctan(x) + 2.0 * xp.arctan(1.0))
    ym = (1.0 - _PSIM_UNST_C * z) ** _PSI_UNST_EXP
    return (psimk + z ** 2 * _psi_c(ym, xp)) / (1 + z ** 2.0)


def _psih_unstable_full(z, xp=jnp):
    y = (1.0 - _PSI_UNST_K * z) ** 0.5
    psihk = 2.0 * xp.log((1 + y) / 2.0)
    yh = (1.0 - _PSIH_UNST_C * z) ** _PSI_UNST_EXP
    return (psihk + z ** 2 * _psi_c(yh, xp)) / (1 + z ** 2.0)


# sf_sfclayrev_init: zolf = float(n)*0.01 (stable), -float(n)*0.01 (unstable)
_ZTAB = np.arange(_TAB_N + 1, dtype=np.float64) * _TAB_STEP
_TABLES = {
    "psim_stab": _psim_stable_full(_ZTAB, np),
    "psih_stab": _psih_stable_full(_ZTAB, np),
    "psim_unstab": _psim_unstable_full(-_ZTAB, np),
    "psih_unstab": _psih_unstable_full(-_ZTAB, np),
}


def _lookup(table, full, zabs, zolf):
    """Oracle look-up (l.1066-1140): nzol = int(|zolf|*100), linear in-cell
    interpolation while nzol+1 < 1000, else the full function at zolf.
    ``zabs`` (>= 0) is zolf for the stable and -zolf for the unstable tables."""
    tab = jnp.asarray(_TABLES[table], dtype=zabs.dtype)
    x = zabs * _TAB_SCALE
    nzol = jnp.trunc(x)
    rzol = x - nzol
    in_tab = nzol + 1 < _TAB_N
    k = jnp.clip(nzol, 0, _TAB_N - 1).astype(jnp.int32)
    interp = tab[k] + rzol * (tab[k + 1] - tab[k])
    return jnp.where(in_tab, interp, full(jnp.where(in_tab, 0.0, zolf)))


def psim_stable(z):
    z = jnp.maximum(z, 0.0)   # sanitised: only selected on zolf >= 0
    return _lookup("psim_stab", _psim_stable_full, z, z)


def psih_stable(z):
    z = jnp.maximum(z, 0.0)
    return _lookup("psih_stab", _psih_stable_full, z, z)


def psim_unstable(z):
    z = jnp.minimum(z, 0.0)   # sanitised: only selected on zolf <= 0
    return _lookup("psim_unstab", _psim_unstable_full, -z, z)


def psih_unstable(z):
    z = jnp.minimum(z, 0.0)
    return _lookup("psih_unstab", _psih_unstable_full, -z, z)


def _zolri2(zol2, ri2, z, z0):
    """l.993-1010; returns (zolri2, zol2): zol2 is intent(inout) and is set
    to 0 when its sign differs from ri2's (the caller's x1/x2 with it)."""
    zol2 = jnp.where(zol2 * ri2 < 0.0, 0.0, zol2)
    zol20 = zol2 * z0 / z
    zol3 = zol2 + zol20
    unst = ri2 < 0
    lg = jnp.log((z + z0) / z0)
    psix2 = lg - jnp.where(unst, psim_unstable(zol3) - psim_unstable(zol20),
                           psim_stable(zol3) - psim_stable(zol20))
    psih2 = lg - jnp.where(unst, psih_unstable(zol3) - psih_unstable(zol20),
                           psih_stable(zol3) - psih_stable(zol20))
    return zol2 * psih2 / psix2 ** 2 - ri2, zol2


def zolri(ri, z, z0):
    """l.957-989: secant search for z/L given the bulk Ri (at most 10
    updates; returns when the bracket is within 0.01 or fx1 == fx2).  The
    oracle leaves the result unassigned when it returns before the first
    update; that case is NaN here (R4 vote X18)."""
    neg = ri < 0.0
    x1 = jnp.where(neg, -_ZOLRI_BRACKET, 0.0)
    x2 = jnp.where(neg, 0.0, _ZOLRI_BRACKET)
    fx1, x1 = _zolri2(x1, ri, z, z0)
    fx2, x2 = _zolri2(x2, ri, z, z0)
    res = jnp.full_like(ri, jnp.nan)
    done = jnp.zeros(ri.shape, bool)

    def body(_, c):
        x1, x2, fx1, fx2, res, done = c
        done = done | ~(jnp.abs(x1 - x2) > _ZOLRI_TOL) | (fx1 == fx2)
        den = jnp.where(done, 1.0, fx2 - fx1)
        left = jnp.abs(fx2) < jnp.abs(fx1)
        f1n, nx1 = _zolri2(x1 - fx1 / den * (x2 - x1), ri, z, z0)
        f2n, nx2 = _zolri2(x2 - fx2 / den * (x2 - x1), ri, z, z0)
        up1 = ~done & left
        up2 = ~done & ~left
        return (jnp.where(up1, nx1, x1), jnp.where(up2, nx2, x2),
                jnp.where(up1, f1n, fx1), jnp.where(up2, f2n, fx2),
                jnp.where(up1, nx1, jnp.where(up2, nx2, res)), done)

    out = jax.lax.fori_loop(0, _ZOLRI_MAXITER, body, (x1, x2, fx1, fx2, res, done))
    return out[4]


class SfclayState(NamedTuple):
    """sfclayrev inout state WRF carries between time steps, each (ncol,)."""
    ust: jax.Array
    mol: jax.Array
    znt: jax.Array
    hfx: jax.Array
    qfx: jax.Array
    qsfc: jax.Array


class SfclayOutput(NamedTuple):
    """sfclayrev outputs (oracle names), each (ncol,)."""
    ust: jax.Array
    mol: jax.Array
    znt: jax.Array
    hfx: jax.Array
    qfx: jax.Array
    lh: jax.Array
    qsfc: jax.Array
    br: jax.Array
    fm: jax.Array
    fh: jax.Array
    psim: jax.Array
    psih: jax.Array
    wspd: jax.Array
    u10: jax.Array
    v10: jax.Array
    th2: jax.Array
    t2: jax.Array
    q2: jax.Array
    zol: jax.Array
    rmol: jax.Array
    regime: jax.Array
    gz1oz0: jax.Array
    chs: jax.Array
    chs2: jax.Array
    cqs2: jax.Array
    cpm: jax.Array
    flhc: jax.Array
    flqc: jax.Array
    qgh: jax.Array


def _safe_sqrt(x):
    """sqrt(x) for x >= 0 with gradient 0 (not inf) at x == 0 (calm wind)."""
    ok = x > 0.0
    return jnp.where(ok, jnp.sqrt(jnp.where(ok, x, 1.0)), 0.0)


def _pos_pow(x, p):
    """x**p for x >= 0 with 0**p = 0 and a finite gradient at 0."""
    ok = x > 0.0
    return jnp.where(ok, jnp.where(ok, x, 1.0) ** p, 0.0)


def _psih_water(zol, za, z0x):
    """psih at (za+z0x)/L, (10+z0x)/L, (2+z0x)/L minus at z0x/L for the water
    thermal/moisture roughness (l.533-551): stable if zol > 0, 0 if zol == 0,
    else unstable (no 0.9*gz limits on this branch)."""
    zol0 = zol * z0x / za

    def f(zh):
        a = zol * (zh + z0x) / za
        return jnp.where(zol > 0.0, psih_stable(a) - psih_stable(zol0),
                         jnp.where(zol == 0.0, 0.0,
                                   psih_unstable(a) - psih_unstable(zol0)))
    return f(za), f(_Z10), f(_Z2)


def sfclayrev_run(u1, v1, t1, qv1, p1, dz1, psfc, tsk, xland, mavail, pblh, dx,
                  state: SfclayState, *, scm_force_flux: bool) -> SfclayOutput:
    """``sfclayrev`` -> ``sf_sfclayrev_run`` on one level, batched over columns.

    All arrays (ncol,): lowest-layer wind, temperature, water vapour (the
    oracle's qv3d), pressure [Pa] and layer thickness dz8w(kts) [m]; surface
    pressure [Pa], skin temperature, land/water flag, moisture availability,
    PBL height [m], grid spacing [m]; and the carried ``state``.  Constants are
    legoesm.constants (vote X15).
    """
    cp, g, r, xlv = constants.c_pd, constants.g, constants.R_d, constants.L_v  # latent-ok: WRF sfclayrev oracle constant
    rovcp = constants.R_d / constants.c_pd
    karman = constants.kappa_von_karman
    ep1 = constants.R_v / constants.R_d - 1.0
    ep2 = constants.R_d / constants.R_v
    p1000mb, svpt0 = constants.p_ref, constants.T_freeze
    ust, mol, znt = state.ust, state.mol, state.znt
    water = (xland - _XLAND_WATER) >= 0.0
    land = ~water

    psfc_cb = psfc / _PA_PER_KPA
    thgb = tsk * (p1000mb / psfc) ** rovcp
    pl = p1 / _PA_PER_KPA
    thx = t1 * (p1000mb / _PA_PER_KPA / pl) ** rovcp
    qx = qv1
    tvcon = 1.0 + ep1 * qx
    thvx = thx * tvcon
    scr4 = t1 * tvcon
    e1 = _SVP1 * jnp.exp(_SVP2 * (tsk - svpt0) / (tsk - _SVP3))  # satcurve-ok: WRF sfclayrev oracle (l.294)
    e1 = jnp.where(xland > _XLAND_WATER, e1 * _SALINITY_FACTOR, e1)   # lakemask = 0
    qsfc = jnp.where((xland > _XLAND_WATER) | (state.qsfc <= 0.0),
                     ep2 * e1 / (psfc_cb - e1), state.qsfc)
    e1a = _SVP1 * jnp.exp(_SVP2 * (t1 - svpt0) / (t1 - _SVP3))  # satcurve-ok: WRF sfclayrev oracle (l.300)
    qgh = ep2 * e1a / (pl - e1a)
    cpm = cp * (1.0 + _CPM_QV_COEFF * qx)

    rhox = psfc_cb * _PA_PER_KPA / (r * scr4)
    za = 0.5 * dz1
    govrth = g / thx

    # --- bulk Richardson number (l.326-366) ---
    gz1oz0 = jnp.log((za + znt) / znt)
    gz2oz0 = jnp.log((_Z2 + znt) / znt)
    gz10oz0 = jnp.log((_Z10 + znt) / znt)
    wspd = _safe_sqrt(u1 * u1 + v1 * v1)
    tskv = thgb * (1.0 + ep1 * qsfc)
    dthvdz = thvx - tskv
    fluxc = jnp.maximum(state.hfx / rhox / cp + ep1 * tskv * state.qfx / rhox, 0.0)
    vconv_land = _VCONVC * _pos_pow(g / tsk * pblh * fluxc, _VCONV_EXP)
    dthvm = jnp.where(-dthvdz >= 0, -dthvdz, 0.0)
    vconv = jnp.where(xland < _XLAND_WATER, vconv_land, _safe_sqrt(dthvm))
    vsgd = _VSGD_COEFF * _pos_pow(jnp.maximum(dx / _VSGD_DX - 1.0, 0.0), _VCONV_EXP)
    wspd = _safe_sqrt(wspd * wspd + vconv * vconv + vsgd * vsgd)
    wspd = jnp.maximum(wspd, _WSPD_MIN)
    br = govrth * za * dthvdz / (wspd * wspd)
    br = jnp.where(mol < 0.0, jnp.minimum(br, 0.0), br)

    # --- z/L and stability class (l.384-489) ---
    stab, unst = br > 0.0, br < 0.0
    zol_s = zolri(jnp.minimum(jnp.where(stab, br, 1.0), _BR_MAX), za, znt)
    zol_u = jnp.where(ust < _UST_ZOLRI_MIN, br * gz1oz0,
                      zolri(jnp.maximum(jnp.where(unst, br, -1.0), -_BR_MAX), za, znt))
    zol = jnp.where(stab, zol_s, jnp.where(unst, zol_u, 0.0))
    zol0 = zol * znt / za
    zl = jnp.where(land, _ZL_LAND / za * zol, zol0)
    S_m, S_h, U_m, U_h = psim_stable, psih_stable, psim_unstable, psih_unstable

    def pair(zh, lim):
        """(psim, psih) at (zh+z0)/L minus at z0/L; unstable limited to
        0.9*lim (lim=None: no limit, psim2 is never limited)."""
        a = zol * (zh + znt) / za
        um, uh = U_m(a) - U_m(zol0), U_h(a) - U_h(zol0)
        if lim is not None:
            um, uh = jnp.minimum(um, _PSI_LIMIT * lim), jnp.minimum(uh, _PSI_LIMIT * lim)
        return (jnp.where(stab, S_m(a) - S_m(zol0), jnp.where(unst, um, 0.0)),
                jnp.where(stab, S_h(a) - S_h(zol0), jnp.where(unst, uh, 0.0)))

    psim, psih = pair(za, gz1oz0)
    psim10, psih10 = pair(_Z10, gz10oz0)
    _, psih2 = pair(_Z2, gz2oz0)

    def pqf(a):
        return jnp.where(stab, S_h(a) - S_h(zl), jnp.where(unst, U_h(a) - U_h(zl), 0.0))
    pq, pq2, pq10 = pqf(zol), pqf(_Z2 / za * zol), pqf(_Z10 / za * zol)
    regime = jnp.where(stab, _REGIME_STABLE, jnp.where(unst, _REGIME_UNSTABLE, _REGIME_NEUTRAL))
    rmol = zol / za

    # --- friction velocity, water thermal roughness (l.494-796) ---
    dtg = thx - thgb
    psix = gz1oz0 - psim
    psix10 = gz10oz0 - psim10
    zlq = jnp.where(water, znt, _ZL_LAND)
    psiq = jnp.log(karman * ust * za / _XKA + za / zlq) - pq
    psiq2 = jnp.log(karman * ust * _Z2 / _XKA + _Z2 / zlq) - pq2
    psit = gz1oz0 - psih
    psit2 = gz2oz0 - psih2
    visc = (_VISC_A + _VISC_B * (t1 - constants.T_freeze)) * _VISC_SCALE
    restar = ust * znt / visc
    z0t = jnp.where(restar > 0.0, _Z0T_A * _pos_pow(restar, _Z0T_EXP), jnp.inf)
    z0t = jnp.maximum(jnp.minimum(z0t, _Z0T_MAX), _Z0T_MIN)
    wh, wh10, wh2 = _psih_water(zol, za, z0t)
    psih = jnp.where(water, wh, psih)
    psih10 = jnp.where(water, wh10, psih10)
    psih2 = jnp.where(water, wh2, psih2)
    lzt, lzt2 = jnp.log((za + z0t) / z0t), jnp.log((_Z2 + z0t) / z0t)
    psit = jnp.where(water, lzt - wh, psit)
    psit2 = jnp.where(water, lzt2 - wh2, psit2)
    psiq = jnp.where(water, lzt - wh, psiq)      # z0q = z0t
    psiq2 = jnp.where(water, lzt2 - wh2, psiq2)

    ust_new = 0.5 * ust + 0.5 * karman * wspd / psix
    u10 = u1 * psix10 / psix
    v10 = v1 * psix10 / psix
    th2 = thgb + dtg * psit2 / psit
    q2 = qsfc + (qx - qsfc) * psiq2 / psiq
    t2 = th2 * (psfc / p1000mb) ** rovcp
    ust_new = jnp.where(land, jnp.maximum(ust_new, _UST_LAND_MIN), ust_new)
    mol_new = karman * dtg / psit / _PRT

    # --- fluxes, isfflx = 1 (l.802-916) ---
    znt_w = jnp.minimum(_CZO * ust_new * ust_new / g + _CHARNOCK_VISC / ust_new,
                        _ZNT_WATER_MAX)
    znt_new = jnp.where(water, znt_w, znt)
    flqc = rhox * mavail * ust_new * karman / psiq
    big = jnp.abs(dtg) > _DTTHX_MIN
    flhc = jnp.where(big, cpm * rhox * ust_new * mol_new / jnp.where(big, dtg, 1.0), 0.0)
    if scm_force_flux:
        hfx, qfx = state.hfx, state.qfx
        lh = jnp.full_like(qfx, jnp.nan)   # unassigned in the oracle
    else:
        qfx = flqc * (qsfc - qx)
        lh = xlv * qfx
        hfx = jnp.where(xland != _XLAND_WATER, flhc * (thgb - thx), 0.0)

    return SfclayOutput(
        ust=ust_new, mol=mol_new, znt=znt_new, hfx=hfx, qfx=qfx, lh=lh, qsfc=qsfc,
        br=br, fm=psix, fh=psit, psim=psim, psih=psih, wspd=wspd, u10=u10, v10=v10,
        th2=th2, t2=t2, q2=q2, zol=zol, rmol=rmol, regime=regime, gz1oz0=gz1oz0,
        chs=ust_new * karman / psiq, chs2=ust_new * karman / psit2,
        cqs2=ust_new * karman / psiq2, cpm=cpm, flhc=flhc, flqc=flqc, qgh=qgh)
