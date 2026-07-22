"""
JAX translation of MLSolarRadiationMod Fortran module.

Solar radiation transfer through the multilayer canopy.
Provides one public driver (:func:`SolarRadiation`) and two private
solvers (:func:`_Norman`, :func:`_TwoStream`).

Original Fortran module: MLSolarRadiationMod
Fortran lines 1-490

Differentiability notes
-----------------------
* All ``import numpy as np`` and ``import math`` removed — JAX used throughout.
* All ``np.asarray()`` calls removed — JAX arrays used directly.
* All ``float()`` wrappers on JAX scalars removed.
* All ``math.*`` calls (cos, exp, sqrt, log) replaced by ``jnp.*``.
* Numpy intermediate working arrays (``_avmu``, ``_betad``, etc.) replaced
  by ``jnp.zeros(...)``; element assignments use ``.at[].set()``.
* Tridiagonal coefficient lists hold ``jnp.zeros(())`` JAX scalars so the
  Thomas algorithm is fully differentiable at trace time.
* ``int()`` wrappers on ``ncan[p]``, ``ntop[p]``, ``nbot[p]`` are kept —
  these must be concrete Python ints for loop bounds and static slices.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import rpi as pi  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import ivis, numrad  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.pftconMod import pftcon  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import J_to_umol, chil_max, chil_min, kb_max  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import (  # noqa: F401
    GridInfo,  # noqa: F401
    leaf_optics_type,
    light_type,
)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varpar import isha, isun, nlevmlcan  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLMathToolsMod import tridiag  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLpftconMod import MLpftcon  # noqa: F401

# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def SolarRadiation(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo | None" = None,
) -> mlcanopy_type:
    """
    Solar radiation transfer through the multilayer canopy.

    Mirrors Fortran subroutine ``SolarRadiation`` (lines 28-195).

    References
    ----------
    Bonan (2019) *Climate Change and Terrestrial Ecosystem Modeling*,
    Chapter 14.
    Bonan et al. (2021) *Agric. For. Met.* 306, 108435,
    supplemental eqs. (1)-(5).

    Execution sequence
    ------------------
    1. For each patch, zero all layer optical variables.
    2. Loop over leaf layers (``ntop`` down to ``nbot``): compute
       leaf/stem reflectance (``rho``), transmittance (``tau``),
       scattering (``omega``), leaf angle distribution (``chil``,
       ``phi1``, ``phi2``, ``gdir``), direct beam extinction
       coefficient ``kb``, clumping factor, single-layer direct
       transmittance ``tb``, diffuse transmittance ``td`` (9-angle
       numerical integration), cumulative direct beam transmittance
       ``tbi``, sunlit fraction ``fracsun``, and two-stream parameters
       ``avmu``, ``betad``, ``betab``.
    3. Compute ``tbi(p,0)`` (transmittance onto the ground).
    4. Dispatch to :func:`_Norman` (``light_type == 1``) or
       :func:`_TwoStream` (``light_type == 2``).
    5. Convert PAR absorbed by leaves from W/m2 to umol/m2/s via
       ``J_to_umol``.

    Optical properties follow ``leaf_optics_type``:

    - **0** (constant with height): weighted average of leaf and stem
      properties by ``wl = dlai/dpai``, ``ws = dsai/dpai``.
    - **1** (vary with height): fatal — vertical profiles not
      implemented.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Multilayer canopy container; optical diagnostics,
            flux profiles, and absorbed radiation fields are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    # ------------------------------------------------------------------
    # Unpack inputs (Fortran associate block, lines 65-95)
    # ------------------------------------------------------------------
    xl = pftcon.xl
    rhol = pftcon.rhol
    taul = pftcon.taul
    rhos = pftcon.rhos
    taus = pftcon.taus
    clump_fac = MLpftcon.clump_fac

    solar_zen = mlcanopy_inst.solar_zen_forcing
    ncan = mlcanopy_inst.ncan_canopy
    ntop = mlcanopy_inst.ntop_canopy
    nbot = mlcanopy_inst.nbot_canopy
    dlai = mlcanopy_inst.dlai_profile
    dsai = mlcanopy_inst.dsai_profile
    dpai = mlcanopy_inst.dpai_profile

    fracsun = mlcanopy_inst.fracsun_profile
    kb = mlcanopy_inst.kb_profile
    tb = mlcanopy_inst.tb_profile
    td = mlcanopy_inst.td_profile
    tbi = mlcanopy_inst.tbi_profile
    apar = mlcanopy_inst.apar_leaf

    # ------------------------------------------------------------------
    # Working JAX arrays for optical properties
    # ------------------------------------------------------------------
    n_idx = bounds.endp + 1
    n_lev = nlevmlcan + 1
    n_rad = numrad + 1
    _avmu = jnp.zeros((n_idx, n_lev))
    _betad = jnp.zeros((n_idx, n_lev, n_rad))
    _betab = jnp.zeros((n_idx, n_lev, n_rad))
    _cf_ic = jnp.zeros((n_idx, n_lev))
    _rho_arr = jnp.zeros((n_idx, n_lev, n_rad))
    _tau_arr = jnp.zeros((n_idx, n_lev, n_rad))
    _om_arr = jnp.zeros((n_idx, n_lev, n_rad))

    # ------------------------------------------------------------------
    # Calculate canopy layer optical properties — Fortran lines 100-182
    # ------------------------------------------------------------------
    # Pre-materialise index arrays as numpy so int() calls below are
    # always concrete, even when this function is traced by jax.grad/jit.
    # When grid is provided (diff mode), ncan/ntop/nbot are already
    # concrete Python ints from GridInfo — no np.asarray() needed.
    if grid is None:
        _ncan_np = np.asarray(ncan)
        _ntop_np = np.asarray(ntop)
        _nbot_np = np.asarray(nbot)
    _itype_np = np.asarray(patch.itype)

    for fp in range(1, num_filter + 1):
        p = int(filter_patch[fp - 1]) if grid is None else grid.p
        pft = int(_itype_np[p])
        zen = solar_zen[p]
        cos_zen = jnp.cos(zen)
        _ncan = int(_ncan_np[p]) if grid is None else grid.ncan
        _ntop = int(_ntop_np[p]) if grid is None else grid.ntop
        _nbot = int(_nbot_np[p]) if grid is None else grid.nbot

        # Per-patch output arrays (JAX)
        _kb_p = jnp.zeros(_ncan + 2)
        _fracsun_p = jnp.zeros(_ncan + 2)
        _tb_p = jnp.zeros(_ncan + 2)
        _td_p = jnp.zeros(_ncan + 2)
        _tbi_p = jnp.zeros(_ncan + 2)  # index 0 = ground, 1.._ncan = canopy

        # Extract scalar PFT values once
        if leaf_optics_type == 0:
            _xl_pft = xl[pft]
            _cf_pft = clump_fac[pft]
            _rhol_ib = [rhol[pft, ib] for ib in range(n_rad)]
            _taul_ib = [taul[pft, ib] for ib in range(n_rad)]
            _rhos_ib = [rhos[pft, ib] for ib in range(n_rad)]
            _taus_ib = [taus[pft, ib] for ib in range(n_rad)]
        else:
            endrun(msg=" ERROR: SolarRadiation: need to specify vertical profile for rho & tau")
            _xl_pft = 0.01
            _cf_pft = 1.0
            _rhol_ib = _taul_ib = _rhos_ib = _taus_ib = [1.0e-6] * n_rad

        # Layer-by-layer optical properties — Fortran: do ic = ntop, nbot, -1
        for ic in range(_ntop, _nbot - 1, -1):

            dpai_ic = dpai[p, ic]
            dlai_ic = dlai[p, ic]
            dsai_ic = dsai[p, ic]
            # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
            dpai_safe = jnp.maximum(dpai_ic, 1.0e-30)
            wl = dlai_ic / dpai_safe
            ws = dsai_ic / dpai_safe

            # Reflectance, transmittance, scattering — Fortran lines 121-132
            for ib in range(1, numrad + 1):
                r = jnp.maximum(_rhol_ib[ib] * wl + _rhos_ib[ib] * ws, 1.0e-6)
                t = jnp.maximum(_taul_ib[ib] * wl + _taus_ib[ib] * ws, 1.0e-6)
                _rho_arr = _rho_arr.at[p, ic, ib].set(r)
                _tau_arr = _tau_arr.at[p, ic, ib].set(t)
                _om_arr = _om_arr.at[p, ic, ib].set(r + t)

            # Leaf angle distribution — Fortran lines 134-141
            chil_ic = jnp.clip(_xl_pft, chil_min, chil_max)
            chil_ic = jnp.where(jnp.abs(chil_ic) <= 0.01, 0.01, chil_ic)

            # Ross-Goudriaan phi1, phi2, gdir — Fortran lines 143-148
            p1 = 0.5 - 0.633 * chil_ic - 0.330 * chil_ic * chil_ic
            p2 = 0.877 * (1.0 - 2.0 * p1)
            gd = p1 + p2 * cos_zen

            # Direct beam extinction coefficient — Fortran lines 150-152
            # jnp.maximum avoids NaN grad: cos_zen=0 at horizon gives gd/0=inf
            # in the True branch of jnp.minimum, producing 0*inf=NaN in backward.
            cos_zen_safe = jnp.maximum(cos_zen, 1.0e-10)
            kb_ic = jnp.minimum(gd / cos_zen_safe, kb_max)
            _kb_p = _kb_p.at[ic].set(kb_ic)

            # Clumping factor — Fortran lines 154-159
            cf = _cf_pft
            _cf_ic = _cf_ic.at[p, ic].set(cf)

            # Direct beam single-layer transmittance — Fortran line 161
            _tb_p = _tb_p.at[ic].set(jnp.exp(-kb_ic * dpai_ic * cf))

            # Diffuse transmittance (9-angle integration) — Fortran lines 163-170
            td_ic = jnp.zeros(())
            for j in range(1, 10):
                angle = (5.0 + (j - 1) * 10.0) * pi / 180.0
                gdirj = p1 + p2 * jnp.cos(jnp.asarray(angle))
                td_ic = td_ic + (
                    jnp.exp(-gdirj / jnp.cos(jnp.asarray(angle)) * dpai_ic * cf)
                    * jnp.sin(jnp.asarray(angle))
                    * jnp.cos(jnp.asarray(angle))
                )
            _td_p = _td_p.at[ic].set(td_ic * 2.0 * (10.0 * pi / 180.0))

            # Cumulative direct beam transmittance tbi — Fortran lines 172-177
            if ic == _ntop:
                _tbi_p = _tbi_p.at[ic].set(1.0)
            else:
                _tbi_p = _tbi_p.at[ic].set(
                    _tbi_p[ic + 1] * jnp.exp(-_kb_p[ic + 1] * dpai[p, ic + 1] * _cf_ic[p, ic + 1])
                )

            # Sunlit fraction — Fortran lines 179-188
            tbi_ic = _tbi_p[ic]
            # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
            # When kb*dpai=0, the exp term also =0 so fracsun_ic=0 regardless of denom.
            kb_ic_safe = jnp.maximum(kb_ic * dpai_ic, 1.0e-30)
            fracsun_ic = tbi_ic / kb_ic_safe * (1.0 - jnp.exp(-kb_ic * cf * dpai_ic))
            if grid is None and jnp.any(fracsun_ic <= 0.0):
                endrun(msg=" ERROR: SolarRadiation: fracsun is too small")
            if grid is None and jnp.any((1.0 - fracsun_ic) <= 0.0):
                endrun(msg=" ERROR: SolarRadiation: fracsha is too small")
            _fracsun_p = _fracsun_p.at[ic].set(fracsun_ic)

            # Two-stream avmu — Fortran lines 190-194
            # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
            p1_safe = jnp.maximum(p1, 1.0e-10)
            p2_safe = jnp.maximum(p2, 1.0e-10)
            avmu_ic = (1.0 - p1_safe / p2_safe * jnp.log((p1_safe + p2_safe) / p1_safe)) / p2_safe
            _avmu = _avmu.at[p, ic].set(avmu_ic)

            # betad, betab — Fortran lines 196-209
            for ib in range(1, numrad + 1):
                om = _om_arr[p, ic, ib]
                r_ic = _rho_arr[p, ic, ib]
                t_ic = _tau_arr[p, ic, ib]
                _betad = _betad.at[p, ic, ib].set(
                    0.5 / om * (r_ic + t_ic + (r_ic - t_ic) * ((1.0 + chil_ic) / 2.0) ** 2)
                )
                tmp0 = gd + p2 * cos_zen
                tmp1 = p1 * cos_zen
                # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
                # tmp0 = gd + p2*cos_zen, tmp1 = p1*cos_zen — both non-negative physically
                tmp0_safe = jnp.maximum(tmp0, 1.0e-10)
                tmp1_safe = jnp.maximum(tmp1, 1.0e-10)
                tmp2 = 1.0 - tmp1_safe / tmp0_safe * jnp.log((tmp1_safe + tmp0_safe) / tmp1_safe)
                asu = 0.5 * om * gd / tmp0_safe * tmp2
                _betab = _betab.at[p, ic, ib].set(
                    (1.0 + avmu_ic * kb_ic) / (om * avmu_ic * kb_ic) * asu
                )

        # tbi onto ground — Fortran lines 211-213
        _tbi_p = _tbi_p.at[0].set(
            _tbi_p[_nbot] * jnp.exp(-_kb_p[_nbot] * dpai[p, _nbot] * _cf_ic[p, _nbot])
        )

        # Write-back — already JAX arrays, no conversion needed
        _sl = slice(1, _ncan + 1)
        _sl0 = slice(0, _ncan + 1)
        kb = kb.at[p, _sl].set(_kb_p[1 : _ncan + 1])
        fracsun = fracsun.at[p, _sl].set(_fracsun_p[1 : _ncan + 1])
        tb = tb.at[p, _sl].set(_tb_p[1 : _ncan + 1])
        td = td.at[p, _sl].set(_td_p[1 : _ncan + 1])
        tbi = tbi.at[p, _sl0].set(_tbi_p[0 : _ncan + 1])

    # ------------------------------------------------------------------
    # Commit optical properties before radiative transfer
    # ------------------------------------------------------------------
    mlcanopy_inst = mlcanopy_inst._replace(
        fracsun_profile=fracsun,
        kb_profile=kb,
        tb_profile=tb,
        td_profile=td,
        tbi_profile=tbi,
    )

    # ------------------------------------------------------------------
    # Radiative transfer solver — Fortran lines 216-222
    # ------------------------------------------------------------------
    if light_type == 1:
        mlcanopy_inst = _Norman(
            bounds,
            num_filter,
            filter_patch,
            _rho_arr,
            _tau_arr,
            _om_arr,
            mlcanopy_inst,
            grid=grid,
        )
    elif light_type == 2:
        mlcanopy_inst = _TwoStream(
            bounds,
            num_filter,
            filter_patch,
            _om_arr,
            _avmu,
            _betad,
            _betab,
            _cf_ic,
            mlcanopy_inst,
            grid=grid,
        )
    else:
        endrun(msg=" ERROR: SolarRadiation: light_type not valid")

    # ------------------------------------------------------------------
    # APAR: W/m2 → umol/m2/s — Fortran lines 224-229 (vectorized)
    # ------------------------------------------------------------------
    swleaf = mlcanopy_inst.swleaf_leaf
    apar = mlcanopy_inst.apar_leaf
    for fp in range(1, num_filter + 1):
        p = int(filter_patch[fp - 1]) if grid is None else grid.p
        _ncan = int(_ncan_np[p]) if grid is None else grid.ncan
        _sl = slice(1, _ncan + 1)
        apar = apar.at[p, _sl, :].set(swleaf[p, _sl, :, ivis] * J_to_umol)

    mlcanopy_inst = mlcanopy_inst._replace(apar_leaf=apar)
    return mlcanopy_inst


# ---------------------------------------------------------------------------
# Private: Norman (1979) radiative transfer
# ---------------------------------------------------------------------------


def _Norman(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    rho: Array,
    tau: Array,
    omega: Array,
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo | None" = None,
) -> mlcanopy_type:
    """
    Norman (1979) tridiagonal radiative transfer through the canopy.

    Mirrors Fortran subroutine ``Norman`` (private, lines 232-370).

    For each waveband ``ib`` and each patch ``p``, sets up a tridiagonal
    system of ``2*(nbot-to-ntop layers + soil)`` equations for the
    upward and downward diffuse fluxes at each layer interface and
    solves it via :func:`MLMathToolsMod.tridiag`. The ordering of
    equations follows the Fortran exactly: soil upward (m=1), soil
    downward (m=2), then for ``ic = nbot, ..., ntop-1`` upward (odd)
    and downward (even), finally top-layer upward and downward.

    After solving, absorbed direct-beam and diffuse fluxes per layer
    are partitioned into sunlit (``swleaf[..,isun,..]``) and shaded
    (``swleaf[..,isha,..]``) components per unit leaf area. Energy
    conservation is checked to within 1e-3 W/m2.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches.
        filter_patch: Patch filter.
        rho: Leaf/stem reflectance ``(p, ic, ib)``.
        tau: Leaf/stem transmittance ``(p, ic, ib)``.
        omega: Scattering coefficient ``(p, ic, ib)``.
        mlcanopy_inst: Canopy container; radiation flux fields updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    neq = (nlevmlcan + 1) * 2  # Fortran: parameter neq = (nlevmlcan+1)*2

    swskyb = mlcanopy_inst.swskyb_forcing
    swskyd = mlcanopy_inst.swskyd_forcing
    ncan = mlcanopy_inst.ncan_canopy
    ntop = mlcanopy_inst.ntop_canopy
    nbot = mlcanopy_inst.nbot_canopy
    albsoib = mlcanopy_inst.albsoib_soil
    albsoid = mlcanopy_inst.albsoid_soil
    dpai = mlcanopy_inst.dpai_profile
    fracsun = mlcanopy_inst.fracsun_profile
    tb = mlcanopy_inst.tb_profile
    td = mlcanopy_inst.td_profile
    tbi = mlcanopy_inst.tbi_profile

    swveg = mlcanopy_inst.swveg_canopy
    swvegsun = mlcanopy_inst.swvegsun_canopy
    swvegsha = mlcanopy_inst.swvegsha_canopy
    albcan = mlcanopy_inst.albcan_canopy
    swsoi = mlcanopy_inst.swsoi_soil
    swleaf = mlcanopy_inst.swleaf_leaf
    swupw = mlcanopy_inst.swupw_profile
    swdwn = mlcanopy_inst.swdwn_profile
    swbeam = mlcanopy_inst.swbeam_profile

    # Pre-materialise ncan/ntop/nbot as numpy so int() calls below are
    # always concrete inside jax.grad/jit tracing.
    # When grid is provided (diff mode), use pre-extracted concrete ints.
    if grid is None:
        _ncan_np = np.asarray(ncan)
        _ntop_np = np.asarray(ntop)
        _nbot_np = np.asarray(nbot)

    for ib in range(1, numrad + 1):  # Fortran: do ib = 1, numrad
        for fp in range(1, num_filter + 1):
            p = int(filter_patch[fp - 1]) if grid is None else grid.p
            _ncan = int(_ncan_np[p]) if grid is None else grid.ncan
            _ntop = int(_ntop_np[p]) if grid is None else grid.ntop
            _nbot = int(_nbot_np[p]) if grid is None else grid.nbot

            # Scalar inputs — JAX scalars directly
            _swskyb_ib = swskyb[p, ib]
            _swskyd_ib = swskyd[p, ib]
            _albsoib_ib = albsoib[p, ib]
            _albsoid_ib = albsoid[p, ib]

            # Zero swleaf for all layers (bulk — 1 JAX op)
            swleaf = swleaf.at[p, 1 : _ncan + 1, :, ib].set(0.0)

            # ----------------------------------------------------------------
            # Build tridiagonal system — Fortran lines 284-340
            # Lists hold JAX traced scalars; Thomas algorithm is differentiable
            # ----------------------------------------------------------------
            atri = [jnp.zeros(())] * (neq + 1)
            btri = [jnp.zeros(())] * (neq + 1)
            ctri = [jnp.zeros(())] * (neq + 1)
            dtri = [jnp.zeros(())] * (neq + 1)

            m = 0

            # Soil: upward flux — Fortran lines 287-290
            m += 1
            atri[m] = jnp.zeros(())
            btri[m] = jnp.ones(())
            ctri[m] = -_albsoid_ib
            dtri[m] = _swskyb_ib * tbi[p, 0] * _albsoib_ib

            # Soil: downward flux — Fortran lines 292-301
            td_nb = td[p, _nbot]
            rho_nb = rho[p, _nbot, ib]
            tau_nb = tau[p, _nbot, ib]
            tb_nb = tb[p, _nbot]
            tbi_nb = tbi[p, _nbot]
            refld = (1.0 - td_nb) * rho_nb
            trand = (1.0 - td_nb) * tau_nb + td_nb
            aic = refld - trand * trand / refld
            bic = trand / refld
            m += 1
            atri[m] = -aic
            btri[m] = jnp.ones(())
            ctri[m] = -bic
            dtri[m] = _swskyb_ib * tbi_nb * (1.0 - tb_nb) * (tau_nb - rho_nb * bic)

            # Leaf layers except top — Fortran lines 303-326
            for ic in range(_nbot, _ntop):  # Fortran: do ic = nbot, ntop-1

                # Upward flux
                td_ic = td[p, ic]
                rho_ic = rho[p, ic, ib]
                tau_ic = tau[p, ic, ib]
                refld = (1.0 - td_ic) * rho_ic
                trand = (1.0 - td_ic) * tau_ic + td_ic
                fic = refld - trand * trand / refld
                eic = trand / refld
                m += 1
                atri[m] = -eic
                btri[m] = jnp.ones(())
                ctri[m] = -fic
                dtri[m] = _swskyb_ib * tbi[p, ic] * (1.0 - tb[p, ic]) * (rho_ic - tau_ic * eic)

                # Downward flux
                ic1 = ic + 1
                td_ic1 = td[p, ic1]
                rho_ic1 = rho[p, ic1, ib]
                tau_ic1 = tau[p, ic1, ib]
                refld = (1.0 - td_ic1) * rho_ic1
                trand = (1.0 - td_ic1) * tau_ic1 + td_ic1
                aic = refld - trand * trand / refld
                bic = trand / refld
                m += 1
                atri[m] = -aic
                btri[m] = jnp.ones(())
                ctri[m] = -bic
                dtri[m] = _swskyb_ib * tbi[p, ic1] * (1.0 - tb[p, ic1]) * (tau_ic1 - rho_ic1 * bic)

            # Top layer: upward flux — Fortran lines 328-337
            ic = _ntop
            td_ic = td[p, ic]
            rho_ic = rho[p, ic, ib]
            tau_ic = tau[p, ic, ib]
            refld = (1.0 - td_ic) * rho_ic
            trand = (1.0 - td_ic) * tau_ic + td_ic
            fic = refld - trand * trand / refld
            eic = trand / refld
            m += 1
            atri[m] = -eic
            btri[m] = jnp.ones(())
            ctri[m] = -fic
            dtri[m] = _swskyb_ib * tbi[p, ic] * (1.0 - tb[p, ic]) * (rho_ic - tau_ic * eic)

            # Top layer: downward flux — Fortran lines 339-343
            m += 1
            atri[m] = jnp.zeros(())
            btri[m] = jnp.ones(())
            ctri[m] = jnp.zeros(())
            dtri[m] = _swskyd_ib

            # Solve — Fortran line 345
            utri = tridiag(atri, btri, ctri, dtri, m)

            # ----------------------------------------------------------------
            # Unpack solution into JAX arrays
            # ----------------------------------------------------------------
            swupw_new = jnp.zeros(_ncan + 2)
            swdwn_new = jnp.zeros(_ncan + 2)
            m_sol = 0
            m_sol += 1
            swupw_new = swupw_new.at[0].set(utri[m_sol])
            m_sol += 1
            swdwn_new = swdwn_new.at[0].set(utri[m_sol])
            for ic in range(_nbot, _ntop + 1):
                m_sol += 1
                swupw_new = swupw_new.at[ic].set(utri[m_sol])
                m_sol += 1
                swdwn_new = swdwn_new.at[ic].set(utri[m_sol])

            # ----------------------------------------------------------------
            # Compute fluxes — Fortran lines 365-430
            # ----------------------------------------------------------------

            # Ground absorption — Fortran lines 368-372
            _swbeam_0 = tbi[p, 0] * _swskyb_ib
            _swsoi_ib = _swbeam_0 * (1.0 - _albsoib_ib) + swdwn_new[0] * (1.0 - _albsoid_ib)

            # Per-layer accumulators (JAX arrays)
            swbeam_new = jnp.zeros(_ncan + 2)
            swleaf_sun_new = jnp.zeros(_ncan + 2)
            swleaf_sha_new = jnp.zeros(_ncan + 2)
            swbeam_new = swbeam_new.at[0].set(_swbeam_0)
            _swveg_acc = jnp.zeros(())
            _swvegsun_acc = jnp.zeros(())
            _swvegsha_acc = jnp.zeros(())

            for ic in range(_nbot, _ntop + 1):  # Fortran: do ic = nbot, ntop
                _swbeam_ic = tbi[p, ic] * _swskyb_ib
                _om_ic = omega[p, ic, ib]
                _swabsb_ic = _swbeam_ic * (1.0 - tb[p, ic]) * (1.0 - _om_ic)

                icm1 = 0 if ic == _nbot else ic - 1
                _swabsd_ic = (swdwn_new[ic] + swupw_new[icm1]) * (1.0 - td[p, ic]) * (1.0 - _om_ic)

                _fs = fracsun[p, ic]
                _swsha = _swabsd_ic * (1.0 - _fs)
                _swsun = _swabsd_ic * _fs + _swabsb_ic

                _dpai_ic = dpai[p, ic]
                # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
                fs_safe = jnp.maximum(_fs, 1.0e-30)
                fsha_safe = jnp.maximum(1.0 - _fs, 1.0e-30)
                dpai_safe = jnp.maximum(_dpai_ic, 1.0e-30)
                swleaf_sun_new = swleaf_sun_new.at[ic].set(_swsun / (fs_safe * dpai_safe))
                swleaf_sha_new = swleaf_sha_new.at[ic].set(_swsha / (fsha_safe * dpai_safe))
                swbeam_new = swbeam_new.at[ic].set(_swbeam_ic)
                _swveg_acc = _swveg_acc + _swabsb_ic + _swabsd_ic
                _swvegsun_acc = _swvegsun_acc + _swsun
                _swvegsha_acc = _swvegsha_acc + _swsha

            # Albedo — Fortran lines 410-414
            _suminc = _swskyb_ib + _swskyd_ib
            # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
            _suminc_safe = jnp.maximum(_suminc, 1.0e-30)
            _albcan_ib = jnp.where(_suminc > 0.0, swupw_new[_ntop] / _suminc_safe, 0.0)

            # Conservation checks — Fortran lines 416-426
            _sumref = _albcan_ib * _suminc
            _sumabs = _suminc - _sumref
            _err = _sumabs - (_swveg_acc + _swsoi_ib)
            if grid is None and jnp.abs(_err) > 1.0e-3:
                endrun(msg="ERROR: Norman: total solar conservation error")
            _err2 = (_swvegsun_acc + _swvegsha_acc) - _swveg_acc
            if grid is None and jnp.abs(_err2) > 1.0e-3:
                endrun(msg="ERROR: Norman: sunlit/shade solar conservation error")

            # Bulk JAX write-back
            _sl = slice(0, _ncan + 1)
            _sl_layers = slice(_nbot, _ntop + 1)
            swupw = swupw.at[p, _sl, ib].set(swupw_new[: _ncan + 1])
            swdwn = swdwn.at[p, _sl, ib].set(swdwn_new[: _ncan + 1])
            swbeam = swbeam.at[p, _sl, ib].set(swbeam_new[: _ncan + 1])
            swleaf = swleaf.at[p, _sl_layers, isun, ib].set(swleaf_sun_new[_nbot : _ntop + 1])
            swleaf = swleaf.at[p, _sl_layers, isha, ib].set(swleaf_sha_new[_nbot : _ntop + 1])
            swsoi = swsoi.at[p, ib].set(_swsoi_ib)
            swveg = swveg.at[p, ib].set(_swveg_acc)
            swvegsun = swvegsun.at[p, ib].set(_swvegsun_acc)
            swvegsha = swvegsha.at[p, ib].set(_swvegsha_acc)
            albcan = albcan.at[p, ib].set(_albcan_ib)

        # end patch loop
    # end waveband loop

    return mlcanopy_inst._replace(
        swveg_canopy=swveg,
        swvegsun_canopy=swvegsun,
        swvegsha_canopy=swvegsha,
        albcan_canopy=albcan,
        swsoi_soil=swsoi,
        swleaf_leaf=swleaf,
        swupw_profile=swupw,
        swdwn_profile=swdwn,
        swbeam_profile=swbeam,
    )


# ---------------------------------------------------------------------------
# Private: two-stream radiative transfer
# ---------------------------------------------------------------------------


def _TwoStream(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    omega: Array,
    avmu: Array,
    betad: Array,
    betab: Array,
    clump_fac_ic: Array,
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo | None" = None,
) -> mlcanopy_type:
    """
    Two-stream radiative transfer integrated over each canopy layer.

    Mirrors Fortran subroutine ``TwoStream`` (private, lines 372-490).

    References
    ----------
    Bonan et al. (2021) *Agric. For. Met.* 306, 108435,
    supplemental eqs. (1)-(5).

    The algorithm proceeds in two sweeps:

    **Bottom-to-top sweep**: For each layer compute analytic two-stream
    solutions for unit direct beam (``unitb = 1``) and unit diffuse
    (``unitd = 1``) forcing. Boundary condition at the bottom is the
    soil albedo; at each subsequent layer the upward flux from the
    layer below is used. Store per-layer scattered fluxes
    ``iupwb0``, ``iupwb``, ``idwnb`` (direct) and ``iupwd0``,
    ``iupwd``, ``idwnd`` (diffuse), plus absorbed terms partitioned
    into sunlit/shaded components.

    **Top-to-bottom sweep**: Propagate the actual incident fluxes
    downward to compute ``swbeam``, ``swdwn``, ``swupw``, and per-leaf
    absorptions ``swleaf``. Accumulate canopy totals and check energy
    conservation to within 1e-6 W/m2.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches.
        filter_patch: Patch filter.
        omega: Scattering coefficient ``(p, ic, ib)``.
        avmu: Average inverse diffuse optical depth ``(p, ic)``.
        betad: Diffuse upscatter parameter ``(p, ic, ib)``.
        betab: Direct beam upscatter parameter ``(p, ic, ib)``.
        clump_fac_ic: Foliage clumping index ``(p, ic)``.
        mlcanopy_inst: Canopy container; radiation fields updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    swskyb = mlcanopy_inst.swskyb_forcing
    swskyd = mlcanopy_inst.swskyd_forcing
    ncan = mlcanopy_inst.ncan_canopy
    ntop = mlcanopy_inst.ntop_canopy
    nbot = mlcanopy_inst.nbot_canopy
    albsoib = mlcanopy_inst.albsoib_soil
    albsoid = mlcanopy_inst.albsoid_soil
    dpai = mlcanopy_inst.dpai_profile
    fracsun = mlcanopy_inst.fracsun_profile
    kb = mlcanopy_inst.kb_profile
    tbi = mlcanopy_inst.tbi_profile

    swveg = mlcanopy_inst.swveg_canopy
    swvegsun = mlcanopy_inst.swvegsun_canopy
    swvegsha = mlcanopy_inst.swvegsha_canopy
    albcan = mlcanopy_inst.albcan_canopy
    swsoi = mlcanopy_inst.swsoi_soil
    swleaf = mlcanopy_inst.swleaf_leaf
    swupw = mlcanopy_inst.swupw_profile
    swdwn = mlcanopy_inst.swdwn_profile
    swbeam = mlcanopy_inst.swbeam_profile

    n_lev = nlevmlcan + 1
    n_rad = numrad + 1
    nleaf_p1 = 3  # nleaf + 1 = 3

    # Pre-materialise ncan/ntop/nbot as numpy so int() calls below are
    # always concrete inside jax.grad/jit tracing.
    # When grid is provided (diff mode), use pre-extracted concrete ints.
    if grid is None:
        _ncan_np = np.asarray(ncan)
        _ntop_np = np.asarray(ntop)
        _nbot_np = np.asarray(nbot)

    for fp in range(1, num_filter + 1):
        p = int(filter_patch[fp - 1]) if grid is None else grid.p
        _ncan = int(_ncan_np[p]) if grid is None else grid.ncan
        _ntop = int(_ntop_np[p]) if grid is None else grid.ntop
        _nbot = int(_nbot_np[p]) if grid is None else grid.nbot

        # Per-patch output JAX arrays
        _swleaf_p = jnp.zeros((n_lev, nleaf_p1, n_rad))
        _swveg_p = jnp.zeros(n_rad)
        _swvegsun_p = jnp.zeros(n_rad)
        _swvegsha_p = jnp.zeros(n_rad)
        _albcan_p = jnp.zeros(n_rad)
        _swsoi_p = jnp.zeros(n_rad)
        _swupw_p = jnp.zeros((n_lev, n_rad))
        _swdwn_p = jnp.zeros((n_lev, n_rad))
        _swbeam_p = jnp.zeros((n_lev, n_rad))

        for ib in range(1, numrad + 1):

            albb = albsoib[p, ib]
            albd = albsoid[p, ib]

            # Per-layer work arrays (JAX)
            n = _ncan + 2
            _iupwb0 = jnp.zeros(n)
            _iupwb = jnp.zeros(n)
            _idwnb = jnp.zeros(n)
            _iabsb_sun = jnp.zeros(n)
            _iabsb_sha = jnp.zeros(n)
            _iupwd0 = jnp.zeros(n)
            _iupwd = jnp.zeros(n)
            _idwnd = jnp.zeros(n)
            _iabsd_sun = jnp.zeros(n)
            _iabsd_sha = jnp.zeros(n)

            # ----------------------------------------------------------
            # Bottom-to-top sweep — Fortran lines 444-540
            # ----------------------------------------------------------
            for ic in range(_nbot, _ntop + 1):
                om_ic = omega[p, ic, ib]
                av_ic = avmu[p, ic]
                kb_ic = kb[p, ic]
                cf_ic = clump_fac_ic[p, ic]
                dp_ic = dpai[p, ic]
                bd_ic = betad[p, ic, ib]
                bb_ic = betab[p, ic, ib]
                tbi_ic = tbi[p, ic]

                b = (1.0 - (1.0 - bd_ic) * om_ic) / av_ic
                c = bd_ic * om_ic / av_ic
                # jnp.maximum guards sqrt(0) and /0 in backward pass (NaN-safe grads)
                h = jnp.sqrt(jnp.maximum(b * b - c * c, 1.0e-30))
                u = (h - b - c) / (2.0 * h)
                v = (h + b + c) / (2.0 * h)
                # Guard h²−kb² denominator for NaN-safe grads when h ≈ kb
                _h2_kb2 = h * h - kb_ic * kb_ic
                _h2_kb2_sign = jnp.where(_h2_kb2 < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
                _h2_kb2_safe = _h2_kb2_sign * jnp.maximum(jnp.abs(_h2_kb2), 1.0e-30)
                d = om_ic * kb_ic / _h2_kb2_safe  # unitb=1
                g1 = (bb_ic * kb_ic - b * bb_ic - c * (1.0 - bb_ic)) * d
                g2 = ((1.0 - bb_ic) * kb_ic + c * bb_ic + b * (1.0 - bb_ic)) * d
                s1 = jnp.exp(-h * cf_ic * dp_ic)
                s2 = jnp.exp(-kb_ic * cf_ic * dp_ic)

                # Direct beam terms (unitb = 1)
                num1 = v * (g1 + g2 * albd + albb) * s2
                num2 = g2 * (u + v * albd) * s1
                den1 = v * (v + u * albd) / s1
                den2 = u * (u + v * albd) * s1
                # Guard Wronskian-like denominator for NaN-safe backward pass
                _d1d2 = den1 - den2
                _d1d2_sgn = jnp.where(_d1d2 < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
                _d1d2_saf = _d1d2_sgn * jnp.maximum(jnp.abs(_d1d2), 1.0e-30)
                n2b = (num1 - num2) / _d1d2_saf
                n1b = (g2 - n2b * u) / v

                # Guard kb±h denominators — degenerate when kb≈h (already h≠0 from sqrt guard)
                _kbph = kb_ic + h  # always > 0
                _kbmh = kb_ic - h
                _kbmh_sgn = jnp.where(_kbmh < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
                _kbmh_saf = _kbmh_sgn * jnp.maximum(jnp.abs(_kbmh), 1.0e-30)
                _2kb_saf = jnp.maximum(2.0 * kb_ic, 1.0e-30)
                a1b = (
                    -g1 * (1.0 - s2 * s2) / _2kb_saf
                    + n1b * u * (1.0 - s2 * s1) / _kbph
                    + n2b * v * (1.0 - s2 / s1) / _kbmh_saf
                ) * tbi_ic
                a2b = (
                    g2 * (1.0 - s2 * s2) / _2kb_saf
                    - n1b * v * (1.0 - s2 * s1) / _kbph
                    - n2b * u * (1.0 - s2 / s1) / _kbmh_saf
                ) * tbi_ic

                iupwb0_ic = -g1 + n1b * u + n2b * v
                iupwb_ic = -g1 * s2 + n1b * u * s1 + n2b * v / s1
                idwnb_ic = g2 * s2 - n1b * v * s1 - n2b * u / s1
                abs_b = (1.0 - s2) - iupwb0_ic + iupwb_ic - idwnb_ic
                abs_b_sun = (1.0 - om_ic) * ((1.0 - s2) + cf_ic / av_ic * (a1b + a2b))
                _iupwb0 = _iupwb0.at[ic].set(iupwb0_ic)
                _iupwb = _iupwb.at[ic].set(iupwb_ic)
                _idwnb = _idwnb.at[ic].set(idwnb_ic)
                _iabsb_sun = _iabsb_sun.at[ic].set(abs_b_sun)
                _iabsb_sha = _iabsb_sha.at[ic].set(abs_b - abs_b_sun)

                # Diffuse terms (unitd = 1) — reuse guards from direct-beam block
                num1 = (u + v * albd) * s1
                n2d = num1 / _d1d2_saf
                n1d = -(1.0 + n2d * u) / v

                a1d = (
                    n1d * u * (1.0 - s2 * s1) / _kbph + n2d * v * (1.0 - s2 / s1) / _kbmh_saf
                ) * tbi_ic
                a2d = (
                    -n1d * v * (1.0 - s2 * s1) / _kbph - n2d * u * (1.0 - s2 / s1) / _kbmh_saf
                ) * tbi_ic

                iupwd0_ic = n1d * u + n2d * v
                iupwd_ic = n1d * u * s1 + n2d * v / s1
                idwnd_ic = -n1d * v * s1 - n2d * u / s1
                abs_d = 1.0 - iupwd0_ic + iupwd_ic - idwnd_ic
                abs_d_sun = (1.0 - om_ic) * cf_ic / av_ic * (a1d + a2d)
                _iupwd0 = _iupwd0.at[ic].set(iupwd0_ic)
                _iupwd = _iupwd.at[ic].set(iupwd_ic)
                _idwnd = _idwnd.at[ic].set(idwnd_ic)
                _iabsd_sun = _iabsd_sun.at[ic].set(abs_d_sun)
                _iabsd_sha = _iabsd_sha.at[ic].set(abs_d - abs_d_sun)

                # Update albedos for next layer
                albb = iupwb0_ic
                albd = iupwd0_ic

            # ----------------------------------------------------------
            # Top-to-bottom sweep — Fortran lines 536-570
            # ----------------------------------------------------------
            dir_ = swskyb[p, ib]
            dif = swskyd[p, ib]

            for ic in range(_ntop, _nbot - 1, -1):
                _swbeam_p = _swbeam_p.at[ic, ib].set(dir_)
                _swdwn_p = _swdwn_p.at[ic, ib].set(dif)
                _swupw_p = _swupw_p.at[ic, ib].set(_iupwd0[ic] * dif + _iupwb0[ic] * dir_)

                fs = fracsun[p, ic]
                dp_ic = dpai[p, ic]
                # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
                fs_safe = jnp.maximum(fs, 1.0e-30)
                fsha_safe = jnp.maximum(1.0 - fs, 1.0e-30)
                dpai_safe = jnp.maximum(dp_ic, 1.0e-30)
                _swleaf_p = _swleaf_p.at[ic, isun, ib].set(
                    (_iabsb_sun[ic] * dir_ + _iabsd_sun[ic] * dif) / (fs_safe * dpai_safe)
                )
                _swleaf_p = _swleaf_p.at[ic, isha, ib].set(
                    (_iabsb_sha[ic] * dir_ + _iabsd_sha[ic] * dif) / (fsha_safe * dpai_safe)
                )

                kb_ic = kb[p, ic]
                cf_ic = clump_fac_ic[p, ic]
                dif_new = dir_ * _idwnb[ic] + dif * _idwnd[ic]
                dir_ = dir_ * jnp.exp(-kb_ic * cf_ic * dp_ic)
                dif = dif_new

            # Ground fluxes
            _swbeam_p = _swbeam_p.at[0, ib].set(dir_)
            _swdwn_p = _swdwn_p.at[0, ib].set(dif)
            _swupw_p = _swupw_p.at[0, ib].set(albsoid[p, ib] * dif + albsoib[p, ib] * dir_)
            _swsoi_p = _swsoi_p.at[ib].set(
                dir_ * (1.0 - albsoib[p, ib]) + dif * (1.0 - albsoid[p, ib])
            )

            # Canopy albedo
            suminc = swskyb[p, ib] + swskyd[p, ib]
            sumref = _iupwb0[_ntop] * swskyb[p, ib] + _iupwd0[_ntop] * swskyd[p, ib]
            # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
            suminc_safe = jnp.maximum(suminc, 1.0e-30)
            _albcan_p = _albcan_p.at[ib].set(jnp.where(suminc > 0.0, sumref / suminc_safe, 0.0))

            # Sum vegetation absorption
            swveg_ib = jnp.zeros(())
            svs_ib = jnp.zeros(())
            svsh_ib = jnp.zeros(())
            for ic in range(_nbot, _ntop + 1):
                fs = fracsun[p, ic]
                dp_ic = dpai[p, ic]
                sun = _swleaf_p[ic, isun, ib] * fs * dp_ic
                sha = _swleaf_p[ic, isha, ib] * (1.0 - fs) * dp_ic
                swveg_ib = swveg_ib + sun + sha
                svs_ib = svs_ib + sun
                svsh_ib = svsh_ib + sha
            _swveg_p = _swveg_p.at[ib].set(swveg_ib)
            _swvegsun_p = _swvegsun_p.at[ib].set(svs_ib)
            _swvegsha_p = _swvegsha_p.at[ib].set(svsh_ib)

            # Conservation check
            sumabs = swveg_ib + _swsoi_p[ib]
            if grid is None and jnp.abs(suminc - (sumabs + _albcan_p[ib] * suminc)) >= 1.0e-6:
                endrun(msg="ERROR: TwoStream: total solar radiation conservation error")

        # Bulk JAX write-back for this patch
        _sl_lev = slice(0, _ncan + 1)
        _sl_canopy = slice(1, _ncan + 1)
        _sl_rad = slice(1, numrad + 1)
        swbeam = swbeam.at[p, _sl_lev, _sl_rad].set(_swbeam_p[_sl_lev, _sl_rad])
        swdwn = swdwn.at[p, _sl_lev, _sl_rad].set(_swdwn_p[_sl_lev, _sl_rad])
        swupw = swupw.at[p, _sl_lev, _sl_rad].set(_swupw_p[_sl_lev, _sl_rad])
        swsoi = swsoi.at[p, _sl_rad].set(_swsoi_p[_sl_rad])
        albcan = albcan.at[p, _sl_rad].set(_albcan_p[_sl_rad])
        swveg = swveg.at[p, _sl_rad].set(_swveg_p[_sl_rad])
        swvegsun = swvegsun.at[p, _sl_rad].set(_swvegsun_p[_sl_rad])
        swvegsha = swvegsha.at[p, _sl_rad].set(_swvegsha_p[_sl_rad])
        swleaf = swleaf.at[p, _sl_canopy, isun, _sl_rad].set(_swleaf_p[_sl_canopy, isun, _sl_rad])
        swleaf = swleaf.at[p, _sl_canopy, isha, _sl_rad].set(_swleaf_p[_sl_canopy, isha, _sl_rad])

    return mlcanopy_inst._replace(
        swveg_canopy=swveg,
        swvegsun_canopy=swvegsun,
        swvegsha_canopy=swvegsha,
        albcan_canopy=albcan,
        swsoi_soil=swsoi,
        swleaf_leaf=swleaf,
        swupw_profile=swupw,
        swdwn_profile=swdwn,
        swbeam_profile=swbeam,
    )
