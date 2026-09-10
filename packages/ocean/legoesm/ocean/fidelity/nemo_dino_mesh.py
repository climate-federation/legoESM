"""Analytic transcription of NEMO 5.0.2's DINO mesh — no NEMO file at run time.

NEMO's DINO configuration builds its whole domain analytically from the
``namusr_def`` namelist: there is no bathymetry or coordinate file to read.
This module transcribes that construction statement-for-statement so legoESM
can build the SAME domain standalone, and returns it as the
:class:`~legoesm.ocean.fidelity.nemo_io.NemoGrid` that
:func:`~legoesm.ocean.fidelity.nemo_state_bridge.bridge_nemo_to_legoesm_topo`
already consumes.  Feeding the transcribed mesh to that bridge is what makes a
standalone run use the SAME domain conventions (periodic i, full-step z,
NEMO's own masks) as the certified file-read twin, instead of a second,
parallel convention.

Oracle source, all under ``cfgs/DINO`` of the NEMO 5.0.2 tree:

===============================  ==================================  ==========
transcribed here                 NEMO routine                        lines
===============================  ==================================  ==========
:func:`merc_proj`                ``MY_SRC/usrdef_nam.F90``           238-265
:func:`nemo_dino_domain_size`    ``MY_SRC/usrdef_nam.F90``           141-155
horizontal mesh + Coriolis       ``MY_SRC/usrdef_hgr.F90``           78-153
:func:`mi96_1d`                  ``MY_SRC/zgr_lib.F90``              267-324
:func:`depth_to_e3_1d`           ``MY_SRC/zgr_lib.F90``              326-356
:func:`e3_to_depth_1d`           ``MY_SRC/zgr_lib.F90``              391-412
1-D + 3-D ladder assembly        ``MY_SRC/zgr_lib.F90``              124-204
bowl bathymetry + Drake sill     ``MY_SRC/usrdef_zgr.F90``           174-400
``k_bot`` / ``k_top``            ``MY_SRC/usrdef_zgr.F90``           438-484
i-periodic land ring             ``MY_SRC/usrdef_zgr.F90``           150-163
N/S closed rows                  ``WORK/domzgr.F90``                 300-315
t/u/v masks                      ``WORK/dommsk.F90``                 120-153
constants                        ``WORK/phycst.F90``                 25-37, 86-92
===============================  ==================================  ==========

PRECISION AND ARITHMETIC ORDER.  Every routine here is scalar ``float`` +
``math`` (i.e. the platform's libm), evaluated in NEMO's own operand order.
That is deliberate and is what makes the result BIT-EXACT rather than merely
close: the mesh is built once, on the host, before any tracing, so there is
nothing to gain from vectorising it and everything to lose — ``jnp``'s
transcendentals are XLA's, not libm's, and differ from NEMO by ulps.

MEASURED against the DINO ``RUN_TRAJ/mesh_mask.nc`` (52x199x36, fp64) by
``scripts/validate/ocean_fidelity/dino_1226/nemo_dino_mesh_gate.py``: all 44
of the file's domain variables bit-exact (0 cells unequal) EXCEPT ``ff_t`` and
``ff_f``, which differ by at most 3 ulp.  That residual is the NEMO BINARY's,
not ours: ``-O3`` vectorised the two whole-array Coriolis assignments into
glibc's 2-wide vector sine (14 ``_ZGVbN2v_sin`` calls in ``usr_def_hgr``,
against scalar ``asin``/``cos``/``tanh`` for the loop body), and that routine
is accurate to 4 ulp rather than correctly rounded.  No ``ff_t`` value reaches
a tendency on this path: :func:`bridge_nemo_to_legoesm_topo` BUILDS Coriolis
from ``gphit`` (bit-exact) and reads ``ff_t`` at a single site, to check
itself.  (Its beta-plane sibling ``bridge_nemo_to_legoesm`` DOES consume both
arrays, but serves GYRE and never sees this mesh.)  See the gate's
``FF_ULP_WAIVER``.

RELATION TO :mod:`legoesm.ocean.experiments.dino`.  ``dino.py`` carries a
JAX/vectorised bowl (``_smooth_step`` / ``_exp_bathy`` / ``_gauss_ring`` /
``dino_bathymetry``) used for the paper-recipe bathymetry on an ARBITRARY
grid, where the formula must be traceable and differentiable.  This module's
scalar twins exist for the opposite requirement -- bit-exact reproduction of a
specific compiled binary's output -- and the two cannot be merged without
losing one property or the other.  Change one, check the other.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from legoesm.ocean.constants_config import NEMO_OMEGA
from legoesm.ocean.fidelity.nemo_io import NemoGrid

# --- phycst.F90:25-37, 86-92 (the DINO build is not key_cice, so omega is the
# --- sidereal-day expression; NEMO_OMEGA transcribes exactly that).
_RPI = 3.141592653589793          # phycst.F90:25   rpi
_RAD = _RPI / 180.0               # phycst.F90:26   rad
_RA = 6371229.0                   # phycst.F90:37   ra   [m]  # const-ok: NEMO ra
_OMEGA = NEMO_OMEGA               # phycst.F90:91   omega [rad/s]


class NemoDinoNamelist(NamedTuple):
    """``namusr_def`` for a NEMO DINO run — every value an explicit choice.

    Defaults are :data:`NEMO_DINO_R1`, transcribed field-for-field from
    ``cfgs/DINO/RUN_TRAJ/namelist_cfg``, which is the deck every certified
    DINO twin was generated from.  Nothing here is defaulted from legoESM's
    own constants: a run's namelist is a record, not a default.
    """
    rn_e1_deg: float        # resolution [deg lon]
    rn_phi_min: float       # approximate southern T-row latitude [deg]
    rn_phi_max: float       # approximate northern T-row latitude [deg]
    rn_lam_min: float       # approximate western  T-col longitude [deg]
    rn_lam_max: float       # approximate eastern  T-col longitude [deg]
    nn_k: int               # jpk (levels; the deepest is a permanent dummy)
    nn_botcase: int         # 0 flat / 1 bowl-in-cosh / 2 bowl-in-(1-x^4)
    rn_H: float             # basin depth [m]
    rn_hborder: float       # depth at the coast [m]
    rn_distLam: float       # slope length scale [deg lon]
    rn_dzmin: float         # e3 at the surface [m]
    rn_kth: float           # inflexion point of the tanh stretching
    rn_hco: float           # z-coordinate layer thickness [m]
    rn_acr: float           # slope of the tanh stretching
    ln_Iperio: bool         # i-periodic (re-entrant channel)
    rn_cha_min: float       # southern edge of the channel [deg]
    rn_cha_max: float       # northern edge of the channel [deg]
    ln_mid_ridge: bool      # mid-Atlantic ridge
    ln_drake_sill: bool     # circular Drake sill
    rn_ds_depth: float      # sill crest depth [m]
    rn_ds_width: float      # sill width [deg]
    ln_zco_nam: bool        # z-coordinate (full step)


#: ``cfgs/DINO/RUN_TRAJ/namelist_cfg``, ``&namusr_def`` — the deck that
#: produced every certified DINO twin (and ``RUN_FROMREST_Y1``).
NEMO_DINO_R1 = NemoDinoNamelist(
    rn_e1_deg=1.0, rn_phi_min=-70.0, rn_phi_max=70.0,
    rn_lam_min=0.0, rn_lam_max=50.0, nn_k=36,
    nn_botcase=1, rn_H=4000.0, rn_hborder=2000.0, rn_distLam=3.0,
    rn_dzmin=10.0, rn_kth=35.0, rn_hco=1000.0, rn_acr=10.5,
    ln_Iperio=True, rn_cha_min=-65.0, rn_cha_max=-45.0,
    ln_mid_ridge=False, ln_drake_sill=True,
    rn_ds_depth=2500.0, rn_ds_width=4.0,
    ln_zco_nam=True,
)


# ---------------------------------------------------------------------------
# usrdef_nam.F90 — domain size
# ---------------------------------------------------------------------------

def merc_proj(pphi: float, pres: float) -> int:
    """Grid points between the equator and ``pphi`` (usrdef_nam.F90:238-265)."""
    zarg = _RPI / 4.0 - _RPI / 180.0 * pphi / 2.0
    zjeq = abs(180.0 / _RPI * math.log(math.cos(zarg) / math.sin(zarg)) / pres)
    if pphi > 0:
        zjeq = -zjeq
    zjeq = zjeq + 1.0                       # Fortran indexing starts at 1
    # Fortran NINT is round-half-AWAY-from-zero; Python round() is
    # round-half-to-EVEN and would disagree on an exact .5 (not reached by
    # DINO R1's four call sites, but a resolution change would reach it).
    return int(math.floor(zjeq + 0.5)) if zjeq >= 0 else -int(math.floor(-zjeq + 0.5))


def nemo_dino_domain_size(nml: NemoDinoNamelist = NEMO_DINO_R1):
    """``(kpi, kpj, kpk)`` = ``(52, 199, 36)`` for R1 (usrdef_nam.F90:141-155).

    ``kpi``/``kpj`` are the INNER (haloless) sizes.  A NEMO 5 run prints
    ``jpiglo``/``jpjglo`` = these plus ``2*nn_hls``; a NEMO 5 ``mesh_mask.nc``
    is written at the inner size, which is what these match.
    """
    jeq_n = merc_proj(nml.rn_phi_max, nml.rn_e1_deg)
    jeq_s = merc_proj(nml.rn_phi_min, nml.rn_e1_deg)
    kpj = (jeq_s - jeq_n) + 1
    ziglo = (nml.rn_lam_max - nml.rn_lam_min) / nml.rn_e1_deg
    kpi = math.floor(ziglo)
    if abs(float(kpi) - ziglo) > 0.5:
        kpi = kpi + 1
    kpi = kpi + 2                            # volume conservation across res.
    return kpi, kpj, nml.nn_k


# ---------------------------------------------------------------------------
# zgr_lib.F90 — vertical ladder
# ---------------------------------------------------------------------------

def _logcosh(v: float) -> float:
    return math.log(math.cosh(v))


def mi96_1d(pH, pdzmin, ph_co, pkth, kkconst, pacr, jpk):
    """Madec-Imbard-96 tanh-stretched ladder (zgr_lib.F90:267-324).

    Returns ``(w, t)`` depths, length ``jpk``; entries at or below
    ``kkconst`` are left at 0 exactly as NEMO leaves them (the caller
    overwrites those from the coarser ladder).
    """
    jpkm1 = jpk - 1
    za1 = (pdzmin - (pH - ph_co) / float(jpkm1 - kkconst)) / (
        math.tanh((1 - pkth) / pacr)
        - pacr / float(jpkm1 - kkconst)
        * (_logcosh((jpk - kkconst - pkth) / pacr) - _logcosh((1 - pkth) / pacr))
    )
    za0 = pdzmin - za1 * math.tanh((1 - pkth) / pacr)
    zsur = -za0 - za1 * pacr * _logcosh((1 - pkth) / pacr)
    w = np.zeros(jpk, dtype=np.float64)
    t = np.zeros(jpk, dtype=np.float64)
    for jk in range(kkconst + 1, jpk + 1):
        zw = float(jk - kkconst)
        zt = zw + 0.5
        w[jk - 1] = zsur + za0 * zw + za1 * pacr * _logcosh((zw - pkth) / pacr) + ph_co
        t[jk - 1] = zsur + za0 * zt + za1 * pacr * _logcosh((zt - pkth) / pacr) + ph_co
    return w, t


def depth_to_e3_1d(dept, depw):
    """``e3`` from depths (zgr_lib.F90:326-356)."""
    jpk = len(dept)
    e3t = np.zeros(jpk, dtype=np.float64)
    e3w = np.zeros(jpk, dtype=np.float64)
    e3w[0] = 2.0 * (dept[0] - depw[0])
    for jk in range(1, jpk):
        e3w[jk] = dept[jk] - dept[jk - 1]
        e3t[jk - 1] = depw[jk] - depw[jk - 1]
    e3t[jpk - 1] = 2.0 * (dept[jpk - 1] - depw[jpk - 1])
    return e3t, e3w


def e3_to_depth_1d(e3t, e3w):
    """Depths from ``e3`` (zgr_lib.F90:391-412) — the re-anchoring pass."""
    jpk = len(e3t)
    depw = np.zeros(jpk, dtype=np.float64)
    dept = np.zeros(jpk, dtype=np.float64)
    dept[0] = 0.5 * e3w[0]
    for jk in range(1, jpk):
        depw[jk] = depw[jk - 1] + e3t[jk - 1]
        dept[jk] = dept[jk - 1] + e3w[jk]
    return dept, depw


def vertical_ladders(nml: NemoDinoNamelist, jpk: int):
    """The 1-D reference ladder and the 3-D (column-constant) one.

    ``ln_zco`` DINO calls ``zgr_sco_mi96`` on a FLAT bathymetry equal to the
    effective ``Hmax`` (usrdef_zgr.F90:105-116), so the "3-D" scale factors
    are horizontally uniform; the 1-D reference ladder and the 3-D one still
    differ below ``rn_hco`` because the 3-D pass re-anchors at ``kkconst``
    (zgr_lib.F90:161-189).  NEMO's ``e3t_0`` is the one it integrates with.
    """
    # Reference (kkconst = 0) — zgr_lib.F90:161-166.
    depw, dept = mi96_1d(nml.rn_H, nml.rn_dzmin, 0.0, nml.rn_kth, 0,
                         nml.rn_acr, jpk)
    e3t_1d, e3w_1d = depth_to_e3_1d(dept, depw)
    # kkconst = MINLOC(|pdepw_1d - ph_co|) taken on the depths that exist at
    # zgr_lib.F90:169, i.e. BEFORE the e3_to_depth re-anchor at :187.
    kkconst = int(np.argmin(np.abs(depw - nml.rn_hco))) + 1
    # 3-D pass — zgr_lib.F90:173-182.  It runs on a FLAT bathymetry (zflat =
    # zHmax, usrdef_zgr.F90:107-116), which is why NEMO's "3-D" e3t_0 comes out
    # horizontally uniform on this ln_zco card.  NB the ARGUMENT is kkconst-1
    # (=25 for R1) while the levels the caller overwrites are kkconst+1..jpk
    # (=27..36): two different numbers, and mi96_1d's own formulae all use the
    # argument.
    w_s, t_s = mi96_1d(nml.rn_H, e3w_1d[kkconst - 1], depw[kkconst - 1],
                       nml.rn_kth, kkconst - 1, nml.rn_acr, jpk)
    depw3 = depw.copy()
    dept3 = dept.copy()
    for jk in range(kkconst + 1, jpk + 1):
        depw3[jk - 1] = w_s[jk - 1]
        dept3[jk - 1] = t_s[jk - 1]
    e3t_0, e3w_0 = depth_to_e3_1d(dept3, depw3)
    # zgr_lib.F90:187-189 — depths recomputed from SUM(e3), both ladders.
    gdept_1d, gdepw_1d = e3_to_depth_1d(e3t_1d, e3w_1d)
    gdept_0, gdepw_0 = e3_to_depth_1d(e3t_0, e3w_0)
    return dict(e3t_1d=e3t_1d, e3w_1d=e3w_1d, gdept_1d=gdept_1d,
                gdepw_1d=gdepw_1d, e3t_0=e3t_0, e3w_0=e3w_0,
                gdept_0=gdept_0, gdepw_0=gdepw_0, kkconst=kkconst)


# ---------------------------------------------------------------------------
# usrdef_zgr.F90 — bathymetry shape functions (scalar; see module docstring)
# ---------------------------------------------------------------------------

def _smooth_step(x, lft, rgt):
    """usrdef_zgr.F90:672-706."""
    if x < lft:
        return 0.0
    if x <= rgt:
        u = (x - lft) / (rgt - lft)
        return 6 * u ** 5 - 15 * u ** 4 + 10 * u ** 3
    return 1.0


def _exp_bathy(x, lft, rgt, width, dist, taper):
    """usrdef_zgr.F90:613-668."""
    if x < lft:
        return 0.0
    if lft <= x <= lft + taper:
        zstep = _smooth_step(x, lft, lft + taper)
        znorm = 1 + math.exp(-width / dist)
        return (1 - (math.exp(-(x - lft) / dist)) / znorm) * (1 - zstep) + zstep
    if lft + taper < x < rgt - taper:
        return 1.0
    if rgt - taper <= x <= rgt:
        zstep = 1 - _smooth_step(x, rgt - taper, rgt)
        znorm = 1 + math.exp(-width / dist)
        return (1 - (math.exp((x - rgt) / dist)) / znorm) * (1 - zstep) + zstep
    return 0.0


def _gauss_ring(dist, lam0, phi0, prad, lam, phi, dep_top, dep_bot):
    """usrdef_zgr.F90:738-775."""
    zx = lam - lam0
    zy = phi - phi0
    zexp = (-zx ** 2 - zy ** 2 + 2 * prad * math.sqrt(zx ** 2 + zy ** 2)
            - prad ** 2) / dist ** 2
    if dep_bot >= dep_top:
        return (dep_top - dep_bot) * math.exp(zexp) + dep_bot
    return dep_bot


# ---------------------------------------------------------------------------
# the mesh
# ---------------------------------------------------------------------------

def nemo_dino_mesh(nml: NemoDinoNamelist = NEMO_DINO_R1) -> NemoGrid:
    """Build NEMO's DINO mesh analytically as a :class:`NemoGrid`.

    Field-for-field equivalent to ``read_nemo_mesh_mask(mesh_mask.nc,
    nn_hls=0)`` on the run this namelist describes, so the result can be
    handed to :func:`bridge_nemo_to_legoesm_topo` exactly as the file-read
    mesh is.  ``seam_wall_rows`` is ``None`` for the same reason the
    ``nn_hls=0`` reader leaves it ``None``: on a haloless mesh the periodic
    seam wall is carried by the land mask itself (columns 0 and n_lon-1 are
    real land outside the channel), not by a halo probe.

    Raises
    ------
    ValueError
        For any namelist option this transcription does not cover.  DINO's
        namelist has branches (flat / ``1-x**4`` bathymetry, s-coordinate,
        partial steps, the mid-Atlantic ridge, a closed basin) that the R1
        card does not select; running one of them through this mesh would
        silently return the wrong domain, so it is refused instead.
    """
    if nml.nn_botcase != 1:
        raise ValueError(
            f"nn_botcase={nml.nn_botcase} not transcribed; only the bowl-in-cosh "
            "case (1) that DINO R1 selects is (usrdef_zgr.F90:227-284).")
    if not nml.ln_zco_nam:
        raise ValueError(
            "only ln_zco_nam=True (full-step z) is transcribed; the "
            "s-coordinate branch (usrdef_zgr.F90:124-146) is not.")
    if nml.ln_mid_ridge:
        raise ValueError(
            "ln_mid_ridge=True is not transcribed (usrdef_zgr.F90:311-364); "
            "DINO R1 sets it .false.")
    if not nml.ln_Iperio:
        raise ValueError(
            "only the i-periodic (ln_Iperio=True) DINO domain is transcribed; "
            "a closed basin takes a different land ring (usrdef_zgr.F90:150-163 "
            "vs domzgr.F90:304-307).")
    if not nml.ln_drake_sill:
        raise ValueError(
            "ln_drake_sill=False is not transcribed; DINO R1 sets it .true. "
            "(usrdef_zgr.F90:371-383).")

    hgr = nemo_dino_hgr(nml)
    n_lat, n_lon = hgr["gphit"].shape
    jpk = nml.nn_k
    lad = vertical_ladders(nml, jpk)
    bathy, k_top, k_bot = nemo_dino_bathymetry(nml, hgr, lad["gdept_1d"])

    # --- dommsk.F90:120-153 ----------------------------------------------
    kk = np.arange(1, jpk + 1)[None, None, :]
    tmask = ((k_top[:, :, None] != 0) & (kk >= k_top[:, :, None])
             & (kk <= k_bot[:, :, None])).astype(np.float64)
    # umask/vmask = product of neighbouring tmask + lbc_lnk: i wraps
    # (ln_Iperio), j is closed so the north face of the last row is land.
    umask = tmask * np.roll(tmask, -1, axis=1)
    vmask = np.zeros_like(tmask)
    vmask[:-1] = tmask[:-1] * tmask[1:]

    def _lev(a1d):
        return np.broadcast_to(a1d[None, None, :], (n_lat, n_lon, jpk)).copy()

    e3t_0 = _lev(lad["e3t_0"])
    e3w_0 = _lev(lad["e3w_0"])
    # fmask = tmask(i,j)*tmask(i+1,j)*tmask(i,j+1)*tmask(i+1,j+1) (dommsk:152)
    _te = np.roll(tmask, -1, axis=1)
    fmask = np.zeros_like(tmask)
    fmask[:-1] = tmask[:-1] * _te[:-1] * tmask[1:] * _te[1:]
    # e3tw_to_other_e3 (zgr_lib.F90:206-264) averages neighbouring columns;
    # on this horizontally-uniform ladder the averages equal e3t_0 exactly.
    #
    # THE OPTIONAL HALF OF NemoGrid IS NOT OPTIONAL HERE.  Until 2026-09-10
    # these eight fields were left unset, and the analytic domain therefore
    # disagreed with the file-read one on seven z_coord leaves -- gdepw_0,
    # e3w_0, e1e2u, e1e2v, e2u, e1v and the EEN barotropic Coriolis operands
    # -- so a standalone run and the certified twin were not the same
    # experiment, which is the one thing this transcription exists to
    # guarantee.  Since main's `nemo_e3w_source='mesh_reference'` landed it is
    # worse than a fidelity gap: the standalone card ABORTS at step 1 with
    # "requires z_coord.nemo_e3w_0".  Every value below is already checked
    # bit-for-bit against mesh_mask by nemo_dino_mesh_gate.py; they simply
    # were not being handed to the bridge.
    return NemoGrid(
        glamt=hgr["glamt"], gphit=hgr["gphit"],
        e1t=hgr["e1t"], e2t=hgr["e2t"], e1u=hgr["e1u"], e2v=hgr["e2v"],
        ff_t=hgr["ff_t"], ff_f=hgr["ff_f"],
        e3t_1d=lad["e3t_1d"], gdept_1d=lad["gdept_1d"], gdepw_1d=lad["gdepw_1d"],
        tmask=tmask, umask=umask, vmask=vmask,
        e3t_0=e3t_0, gdept_0=_lev(lad["gdept_0"]),
        gdepw_0=_lev(lad["gdepw_0"]), e3w_0=e3w_0,
        gphiv=hgr["gphiv"],
        e2u=hgr["e2u"], e1v=hgr["e1v"], e1f=hgr["e1f"], e2f=hgr["e2f"],
        e3f_0=e3t_0, fmask=fmask,
        seam_wall_rows=None,
        e3u_0=e3t_0, e3v_0=e3t_0,
        hu_0=(e3t_0 * umask).sum(axis=-1), hv_0=(e3t_0 * vmask).sum(axis=-1),
    )


def nemo_dino_hgr(nml: NemoDinoNamelist = NEMO_DINO_R1) -> dict:
    """The horizontal mesh (usrdef_hgr.F90:78-153), as a dict of 2-D arrays.

    Keys are NEMO's own names.  ``glamv``/``glamf``/``gphiu``/``gphif`` and
    the ``e1v``/``e2u``/``e1f``/``e2f`` variants are included even though
    :class:`NemoGrid` does not carry all of them, so a gate can check the
    whole of ``usr_def_hgr``'s output rather than the subset legoESM reads.
    """
    n_lon, n_lat, _jpk = nemo_dino_domain_size(nml)
    jeq_s = merc_proj(nml.rn_phi_min, nml.rn_e1_deg)
    e1deg = nml.rn_e1_deg

    # --- usrdef_hgr.F90:78-119 -------------------------------------------
    zlam0 = nml.rn_lam_min - e1deg * 0.5
    zti = np.arange(n_lon, dtype=np.float64)            # mig(ji,0) - 1
    ztj = np.arange(1, n_lat + 1, dtype=np.float64) - jeq_s
    lam_t = zlam0 + e1deg * zti                          # plamt = plamv
    lam_u = zlam0 + e1deg * (zti + 0.5)                  # plamu = plamf

    def _merc_row(z):
        # 1./rad * ASIN( TANH( rn_e1_deg * rad * z ) )   (usrdef_hgr.F90:106)
        return np.array([(1.0 / _RAD) * math.asin(math.tanh(e1deg * _RAD * float(v)))
                         for v in z], dtype=np.float64)

    phi_t = _merc_row(ztj)                               # pphit = pphiu
    phi_v = _merc_row(ztj + 0.5)                         # pphiv = pphif

    def _e1(phi):
        # ra * rad * COS( rad * phi ) * rn_e1_deg        (usrdef_hgr.F90:111)
        return np.array([_RA * _RAD * math.cos(_RAD * float(p)) * e1deg
                         for p in phi], dtype=np.float64)

    def _ff(phi):
        # 2. * omega * SIN( rad * phi )                  (usrdef_hgr.F90:152)
        return np.array([2.0 * _OMEGA * math.sin(_RAD * float(p)) for p in phi],
                        dtype=np.float64)

    e_t = _e1(phi_t)
    e_v = _e1(phi_v)
    def col(a):      # a latitude-varying 1-D field, spread over columns
        return np.broadcast_to(a[:, None], (n_lat, n_lon)).copy()

    def row(a):      # a longitude-varying 1-D field, spread over rows
        return np.broadcast_to(a[None, :], (n_lat, n_lon)).copy()
    ff_t = col(_ff(phi_t))
    ff_f = col(_ff(phi_v))
    # plamv == plamt and plamf == plamu (zvi == zti, zfi == zui); pphiu ==
    # pphit and pphif == pphiv (zuj == ztj, zfj == zvj) -- usrdef_hgr.F90:96-99.
    return dict(
        glamt=row(lam_t), glamu=row(lam_u), glamv=row(lam_t), glamf=row(lam_u),
        gphit=col(phi_t), gphiu=col(phi_t), gphiv=col(phi_v), gphif=col(phi_v),
        e1t=col(e_t), e1u=col(e_t), e1v=col(e_v), e1f=col(e_v),
        e2t=col(e_t), e2u=col(e_t), e2v=col(e_v), e2f=col(e_v),
        ff_t=ff_t, ff_f=ff_f,
        lam_t=lam_t, lam_u=lam_u, phi_t=phi_t, phi_v=phi_v,
    )


def nemo_dino_bathymetry(nml: NemoDinoNamelist, hgr: dict, gdept_1d):
    """Bowl + Drake sill, and the resulting ``k_top``/``k_bot``.

    usrdef_zgr.F90:174-400 (``zgr_bat``), :438-484 (``zgr_msk_top_bot``),
    :150-163 (the i-periodic land ring), and domzgr.F90:308-314 (the closed
    N/S rows).

    Returns ``(bathy, k_top, k_bot)``.  NEMO's ``mbathy`` is
    ``MAX(k_bot, 1)`` -- note that the N/S closure zeroes ``k_top`` ONLY
    (domzgr.F90:315), which is why ``mbathy`` stays at the bathymetric level
    on the first and last rows while ``tmask`` there is zero.
    """
    lam_t, phi_t, phi_v = hgr["lam_t"], hgr["phi_t"], hgr["phi_v"]
    lam_u = hgr["lam_u"]
    n_lat, n_lon = len(phi_t), len(lam_t)
    jpk = nml.nn_k
    e1deg = nml.rn_e1_deg
    jeq_s = merc_proj(nml.rn_phi_min, e1deg)

    # --- zgr_get_boundaries (usrdef_zgr.F90:388-436), no cdpoint: the basin
    # --- edges are the extreme U longitudes and V latitudes of the INNER
    # --- domain.  (The halo copies of those same values, which NEMO's
    # --- MAXVAL also sees, cannot change an extremum.)  Cross-checked
    # --- against RUN_TRAJ/ocean.output:382-385.
    zminlam = float(lam_u.min())
    zmaxlam = float(lam_u.max())
    zminphi = float(phi_v.min())
    zmaxphi = float(phi_v.max())

    # --- bathymetry (usrdef_zgr.F90:227-284 + 371-383) --------------------
    zwidth = zmaxlam - zminlam
    zdistLam = nml.rn_distLam
    zdistPhi = math.cos(_RAD * zmaxphi) * nml.rn_distLam
    zcha_min, zcha_max = nml.rn_cha_min, nml.rn_cha_max
    zcha_width = zcha_max - zcha_min
    if zcha_min <= nml.rn_phi_min:           # usrdef_zgr.F90:243-246
        zcha_min = zcha_max - zwidth
    taper = zcha_width / 2

    bathy = np.empty((n_lat, n_lon), dtype=np.float64)
    zx_lam = np.array([_exp_bathy(float(lam), zminlam, zmaxlam, zwidth,
                                  zdistLam, taper) for lam in lam_t],
                      dtype=np.float64)
    for jj in range(n_lat):
        p = float(phi_t[jj])
        zy_cha = _exp_bathy(p, zcha_min, zcha_max, zwidth, zdistLam, taper)
        zy = _exp_bathy(p, zminphi, zmaxphi, zwidth, zdistPhi, taper)
        for ji in range(n_lon):
            zx = zx_lam[ji] * (1.0 - zy_cha) + zy_cha
            bathy[jj, ji] = zx * zy * (nml.rn_H - nml.rn_hborder) + nml.rn_hborder

    # Drake sill (usrdef_zgr.F90:371-383).  ``zmidPhi``/``zrad`` use the
    # NAMELIST channel edges, not the (possibly reassigned) locals.
    zmidPhi = (nml.rn_cha_max + nml.rn_cha_min) / 2
    zrad = (nml.rn_cha_max - nml.rn_cha_min) / 2
    for jj in range(n_lat):
        p = float(phi_t[jj])
        for ji in range(n_lon):
            lam = float(lam_t[ji])
            zds_taper = _smooth_step(lam, zminlam, zminlam + nml.rn_ds_width)
            zds = _gauss_ring(nml.rn_ds_width, zminlam, zmidPhi, zrad, lam, p,
                              nml.rn_ds_depth, bathy[jj, ji])
            bathy[jj, ji] = zds_taper * zds + (1.0 - zds_taper) * bathy[jj, ji]

    # --- zgr_msk_top_bot (usrdef_zgr.F90:438-484) ------------------------
    # NEMO's z2d here is an UNINITIALISED automatic array that only the WHERE
    # writes (usrdef_zgr.F90:456,473): every cell whose bathymetry falls
    # outside (gdept_1d(1), gdept_1d(jpk)] keeps whatever was on the stack.
    # DINO R1 never reaches that -- its bathymetry spans [rn_hborder, rn_H] =
    # [2000, 4000] m, inside (5.03, 4253.19] -- but a namelist that moved rn_H,
    # rn_hborder or rn_dzmin would, and NEMO would then produce garbage while
    # this transcription quietly produced land.  Refuse instead of diverging.
    lo, hi = gdept_1d[0], gdept_1d[jpk - 1]
    if not (bathy > lo).all() or not (bathy <= hi).all():
        raise ValueError(
            f"bathymetry spans [{bathy.min():.3f}, {bathy.max():.3f}] m but "
            f"zgr_msk_top_bot only assigns k_bot on ({lo:.3f}, {hi:.3f}] "
            "(usrdef_zgr.F90:473). Outside that band NEMO reads an "
            "uninitialised array, so no transcription can match it.")
    k_bot = np.zeros((n_lat, n_lon), dtype=np.int64)
    for jk in range(1, jpk):                             # 1..jpkm1
        k_bot = np.where((gdept_1d[jk - 1] < bathy) & (bathy <= gdept_1d[jk]),
                         jk, k_bot)
    k_top = np.minimum(1, k_bot)

    # --- i-periodic land ring (usrdef_zgr.F90:150-163) -------------------
    nn_cha_min = jeq_s - merc_proj(nml.rn_cha_min, e1deg) + 1
    nn_cha_max = jeq_s - merc_proj(nml.rn_cha_max, e1deg) + 1
    jg = np.arange(1, n_lat + 1)
    walled = (jg <= nn_cha_min) | (jg >= nn_cha_max)      # (n_lat,)
    ring = np.ones((n_lat, n_lon), dtype=np.int64)
    ring[walled, 0] = 0
    ring[walled, n_lon - 1] = 0
    k_top = k_top * ring
    k_bot = k_bot * ring

    # --- N/S closed: first + last INNER global row (domzgr.F90:308-314).
    # --- Only ``k_top`` is zeroed there, which is enough: dommsk keys off it.
    k_top[0, :] = 0
    k_top[n_lat - 1, :] = 0
    return bathy, k_top, k_bot


# ---------------------------------------------------------------------------
# usrdef_istate.F90 — the initial state
# ---------------------------------------------------------------------------

def nemo_dino_istate(g: NemoGrid, nn_initcase: int = 4):
    """``usr_def_istate``'s T and S for DINO (usrdef_istate.F90:129-183).

    Statement-for-statement transcription of ``nn_initcase = 4`` — the case
    ``cfgs/DINO/RUN_TRAJ/namelist_cfg:58`` selects — returning
    ``(T, S)`` of shape ``(n_lat, n_lon, jpk)`` in NEMO's own array order.

    Three details decide bit-exactness, and each was measured rather than
    assumed:

    * **The depth operand is the 3-D** ``gdept_0``, not ``gdept_1d``.
      ``istate.F90:132-134`` passes ``zgdept = gdept_3d * (1 + r3t(:,:,Kbb))``,
      and a from-rest start has ``ssh = 0`` (``usr_def_istate_ssh``, case 4,
      usrdef_istate.F90:227-228) hence ``r3t = 0``.  ``gdept_0`` is
      ``zgr_lib``'s ``e3 -> depth`` re-integration of the ladder, which is NOT
      the ``mi96`` analytic ``gdept_1d``: substituting ``gdept_1d`` moves the
      initial temperature by up to 7.6e-2 K.
    * **The profile is evaluated with scalar libm** ``math.tanh``, for the
      reason the module docstring gives; ``np.tanh``/``jnp.tanh`` are not
      bit-equal to it.  The profile depends only on depth, so this is 36
      scalar evaluations, once, on the host.
    * **The meridional blend's anchors come from the MODEL FIELDS**, not from
      the namelist: ``zphiMAX = MAXVAL(gphit)`` (69.8517, not ``rn_phi_max``
      = 70) and ``zTbot``/``zSbot`` = ``MINVAL(profile + 100*(1-tmask))``,
      i.e. the profile at the deepest WET level, not at the deepest reference
      level (usrdef_istate.F90:151-155).  Both are global reductions
      (``mpp_max`` on the negated pair, :159-167); this is the single-domain
      form, which is the same number.

    The ``jk`` loop stops at ``jpkm1`` (:172), so the deepest level keeps the
    un-blended ``profile * tmask``; on DINO that level is dry everywhere, so
    it is zero either way — transcribed as written rather than relied upon.

    ``nn_pert_seed`` (:178-183) is NOT transcribed.  Its default is 0
    (usrdef_nam.F90:81) and neither ``namelist_cfg`` nor ``namelist_ref``
    sets it, so on every deck here the 1e-10 K ensemble perturbation is a
    no-op.  This function takes no seed argument and therefore cannot honour
    a non-zero one: a deck that sets ``nn_pert_seed`` would need it added.

    MEASURED against ``RUN_FROMREST_KT1``'s ``tb``/``sb`` (the untouched
    initial condition, since the Euler first step leaves ``Kbb`` alone):
    all 372528 cells bit-exact, wet and dry.  Gated by
    ``scripts/validate/ocean_fidelity/dino_1226/nemo_dino_istate_gate.py``.
    """
    if nn_initcase != 4:
        raise ValueError(
            f"nn_initcase={nn_initcase} is not transcribed; DINO R1 selects 4 "
            "(namelist_cfg:58). The other cases are usrdef_istate.F90:65-128 "
            "and :184-192.")
    tmask = np.asarray(g.tmask, dtype=np.float64)
    gphit = np.asarray(g.gphit, dtype=np.float64)
    pdept = np.asarray(g.gdept_0, dtype=np.float64)
    if pdept.shape != tmask.shape:
        raise ValueError(
            f"gdept_0 {pdept.shape} and tmask {tmask.shape} disagree.")

    # The profile is a function of depth alone, and DINO's gdept_0 is
    # horizontally uniform (full-step z, ln_zco_nam), so evaluate the 36
    # scalar values once and broadcast.  Refuse a column-varying ladder
    # rather than silently evaluating only column (0, 0) of it.
    if not np.array_equal(pdept, np.broadcast_to(pdept[0, 0][None, None, :],
                                                 pdept.shape)):
        raise ValueError(
            "nemo_dino_istate needs a horizontally uniform gdept_0 (full-step "
            "z); this mesh varies gdept_0 by column, which would need the "
            "profile evaluated per column.")
    z1d = pdept[0, 0]

    th = math.tanh   # scalar libm, as in NEMO's compiled loop

    def _t(z):       # usrdef_istate.F90:135-140
        return ((16. - 12. * th((z - 400) / 700))
                * (-th((500. - z) / 150.) + 1.) / 2.
                + (15. * (1. - th((z - 50.) / 1500.))
                   - 1.4 * th((z - 100.) / 100.)
                   + 7. * (1500. - z) / 1500.)
                * (-th((z - 500.) / 150.) + 1.) / 2.)

    def _s(z):       # usrdef_istate.F90:142-148
        return ((36.25 - 1.13 * th((z - 305) / 460))
                * (-th((500. - z) / 150.) + 1.) / 2
                + (35.55 + 1.25 * (5000. - z) / 5000.
                   - 1.62 * th((z - 60.) / 650.)
                   + 0.2 * th((z - 35.) / 100.)
                   + 0.2 * th((z - 1000.) / 5000.))
                * (-th((z - 500.) / 150.) + 1.) / 2)

    T = np.array([_t(float(z)) for z in z1d])[None, None, :] * tmask
    S = np.array([_s(float(z)) for z in z1d])[None, None, :] * tmask

    # :151-155 -- anchors from the model fields (global reductions)
    zphiMAX = float(gphit.max())
    zTbot = float((T + 100 * (1. - tmask)).min())
    zSbot = float((S + 100 * (1. - tmask)).min())
    z1_phiMAX = 1. / zphiMAX

    # :172-175 -- NEMO's operand order is ((x - bot) * dphi) * (1/phiMAX),
    # which is NOT the same rounding as (x - bot) * (dphi / phiMAX).
    zdphi = (zphiMAX - np.abs(gphit))[:, :, None]
    jpkm1 = tmask.shape[-1] - 1
    sl = (slice(None), slice(None), slice(0, jpkm1))
    T[sl] = ((T[sl] - zTbot) * zdphi * z1_phiMAX + zTbot) * tmask[sl]
    S[sl] = ((S[sl] - zSbot) * zdphi * z1_phiMAX + zSbot) * tmask[sl]
    return T, S


__all__ = (
    "NemoDinoNamelist", "NEMO_DINO_R1", "merc_proj", "nemo_dino_domain_size",
    "mi96_1d", "depth_to_e3_1d", "e3_to_depth_1d", "vertical_ladders",
    "nemo_dino_hgr", "nemo_dino_bathymetry", "nemo_dino_mesh",
    "nemo_dino_istate",
)
