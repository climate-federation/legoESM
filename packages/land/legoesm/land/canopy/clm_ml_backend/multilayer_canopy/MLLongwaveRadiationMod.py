"""
JAX translation of MLLongwaveRadiationMod Fortran module.

Longwave radiation transfer through the multilayer canopy.
Provides one public driver (:func:`LongwaveRadiation`) and one
private solver (:func:`_Norman`).

Original Fortran module: MLLongwaveRadiationMod
Fortran lines 1-195

Differentiability notes
-----------------------
* All ``np.asarray()`` calls removed — JAX arrays used directly.
* All ``np.`` array operations replaced by ``jnp.``.
* ``float()`` wrappers on JAX scalars removed.
* The tridiagonal system is assembled using Python lists of JAX scalars
  (Thomas algorithm in ``_tridiag_py``).  The list-based Python loop
  runs at trace time, so all values inside the lists are JAX traced
  scalars and the solver is fully differentiable.  TODO: replace with
  ``jax.lax.scan`` for a fully JIT-compilable version.
"""

from __future__ import annotations

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp


def _tridiag_py(a, b, c, r, n):
    """Thomas algorithm using pure Python lists — no JAX, no numpy."""
    gam = [0.0] * (n + 1)
    u = [0.0] * (n + 1)
    bet = b[1]
    u[1] = r[1] / bet
    for j in range(2, n + 1):
        gam[j] = c[j - 1] / bet
        bet = b[j] - a[j] * gam[j]
        u[j] = (r[j] - a[j] * u[j - 1]) / bet
    for j in range(n - 1, 0, -1):
        u[j] = u[j] - gam[j + 1] * u[j + 1]
    return u


from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: E402, F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import sb  # noqa: E402, F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: E402, F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type  # noqa: E402, F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch  # noqa: E402, F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: E402, F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import emg  # noqa: E402, F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import GridInfo, longwave_type  # noqa: E402, F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varpar import isha, isun, nlevmlcan  # noqa: E402, F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLMathToolsMod import tridiag  # noqa: E402, F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLpftconMod import MLpftcon  # noqa: E402, F401

# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def LongwaveRadiation(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    grid: GridInfo = None,
) -> mlcanopy_type:
    """
    Longwave radiation transfer through the multilayer canopy.

    Mirrors Fortran subroutine ``LongwaveRadiation`` (lines 30-45).

    Dispatches to :func:`_Norman` when ``longwave_type == 1``; any
    other value is fatal.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Multilayer canopy container; longwave flux
            fields are updated by the solver.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    if longwave_type == 1:
        return _Norman(bounds, num_filter, filter_patch, mlcanopy_inst, grid=grid)
    else:
        endrun(msg=" ERROR: LongwaveRadiation: longwave_type not valid")
        return mlcanopy_inst  # Unreachable; satisfies type checker


# ---------------------------------------------------------------------------
# Private: Norman (1979) longwave radiative transfer
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0, 1, 2))
def _Norman_patch(
    p: int,
    ntop: int,
    nbot: int,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """JIT-compiled Norman longwave solver for a single patch.

    p, ntop, nbot are static Python ints — they are used as Python loop
    bounds and slice indices (e.g. ``jnp.arange(nbot, ntop+1)``) which
    must be concrete under ``jax.jit``.  Recompilation only happens when
    the canopy structure changes (rare for a fixed site).
    """
    neq: int = (nlevmlcan + 1) * 2
    emleaf = MLpftcon.emleaf

    pft = patch.itype[p]  # dynamic gather — valid in JIT
    em_leaf = emleaf[pft]
    rho = 1.0 - em_leaf
    tau = 0.0

    _tleaf_sun = mlcanopy_inst.tleaf_leaf[p, :, isun]
    _tleaf_sha = mlcanopy_inst.tleaf_leaf[p, :, isha]
    _fracsun = mlcanopy_inst.fracsun_profile[p]
    _td = mlcanopy_inst.td_profile[p]
    _dpai = mlcanopy_inst.dpai_profile[p]
    _lwsky_p = mlcanopy_inst.lwsky_forcing[p]
    _tg_p = mlcanopy_inst.tg_soil[p]

    # Layer emission — JAX vectorised
    ics = jnp.arange(nbot, ntop + 1)  # static size: ntop-nbot+1
    lw_source = jnp.zeros(nlevmlcan + 2)
    lw_sun = em_leaf * sb * _tleaf_sun[ics] ** 4
    lw_sha = em_leaf * sb * _tleaf_sha[ics] ** 4
    fs_arr = _fracsun[ics]
    lw_source = lw_source.at[ics].set(
        (lw_sun * fs_arr + lw_sha * (1.0 - fs_arr)) * (1.0 - _td[ics])
    )

    atri = [jnp.zeros(())] * (neq + 1)
    btri = [jnp.zeros(())] * (neq + 1)
    ctri = [jnp.zeros(())] * (neq + 1)
    dtri = [jnp.zeros(())] * (neq + 1)

    m = 0

    m += 1
    atri[m] = jnp.zeros(())
    btri[m] = jnp.ones(())
    ctri[m] = -(1.0 - emg)
    dtri[m] = emg * sb * _tg_p**4

    td_nb = _td[nbot]
    refld = (1.0 - td_nb) * rho
    trand = (1.0 - td_nb) * tau + td_nb
    aic = refld - trand * trand / refld
    bic = trand / refld
    m += 1
    atri[m] = -aic
    btri[m] = jnp.ones(())
    ctri[m] = -bic
    dtri[m] = (1.0 - bic) * lw_source[nbot]

    for ic in range(nbot, ntop):  # static loop — unrolled at trace time
        td_ic = _td[ic]
        refld = (1.0 - td_ic) * rho
        trand = (1.0 - td_ic) * tau + td_ic
        fic = refld - trand * trand / refld
        eic = trand / refld
        m += 1
        atri[m] = -eic
        btri[m] = jnp.ones(())
        ctri[m] = -fic
        dtri[m] = (1.0 - eic) * lw_source[ic]

        ic1 = ic + 1
        td_ic1 = _td[ic1]
        refld = (1.0 - td_ic1) * rho
        trand = (1.0 - td_ic1) * tau + td_ic1
        aic = refld - trand * trand / refld
        bic = trand / refld
        m += 1
        atri[m] = -aic
        btri[m] = jnp.ones(())
        ctri[m] = -bic
        dtri[m] = (1.0 - bic) * lw_source[ic1]

    ic = ntop
    td_ic = _td[ic]
    refld = (1.0 - td_ic) * rho
    trand = (1.0 - td_ic) * tau + td_ic
    fic = refld - trand * trand / refld
    eic = trand / refld
    m += 1
    atri[m] = -eic
    btri[m] = jnp.ones(())
    ctri[m] = -fic
    dtri[m] = (1.0 - eic) * lw_source[ic]

    m += 1
    atri[m] = jnp.zeros(())
    btri[m] = jnp.ones(())
    ctri[m] = jnp.zeros(())
    dtri[m] = _lwsky_p

    utri = _tridiag_py(atri, btri, ctri, dtri, m)

    lwupw_new = jnp.zeros(nlevmlcan + 2)
    lwdwn_new = jnp.zeros(nlevmlcan + 2)
    k = 0
    k += 1
    lwupw_new = lwupw_new.at[0].set(utri[k])
    k += 1
    lwdwn_new = lwdwn_new.at[0].set(utri[k])
    for ic in range(nbot, ntop + 1):  # static loop
        k += 1
        lwupw_new = lwupw_new.at[ic].set(utri[k])
        k += 1
        lwdwn_new = lwdwn_new.at[ic].set(utri[k])

    _lwsoi_p = lwdwn_new[0] - lwupw_new[0]

    icm1s = jnp.where(ics == nbot, 0, ics - 1)
    lwabs_arr = (
        em_leaf * (lwdwn_new[ics] + lwupw_new[icm1s]) * (1.0 - _td[ics]) - 2.0 * lw_source[ics]
    )
    dpai_ics = _dpai[ics]
    lwleaf_arr = lwabs_arr / dpai_ics
    _lwveg_p = jnp.sum(lwabs_arr)

    lwleaf_new = jnp.zeros(nlevmlcan + 2)
    lwleaf_new = lwleaf_new.at[ics].set(lwleaf_arr)

    _lwup_p = lwupw_new[ntop]

    # Conservation check — JIT-compatible via debug.callback
    err = (_lwsky_p - _lwup_p) - (_lwveg_p + _lwsoi_p)
    jax.debug.callback(
        lambda e: (
            endrun(msg="ERROR: Norman: total longwave conservation error")
            if abs(float(e)) > 1.0e-3
            else None
        ),
        err,
    )

    _sl = slice(0, ntop + 1)
    return mlcanopy_inst._replace(
        lwup_canopy=mlcanopy_inst.lwup_canopy.at[p].set(_lwup_p),
        lwveg_canopy=mlcanopy_inst.lwveg_canopy.at[p].set(_lwveg_p),
        lwsoi_soil=mlcanopy_inst.lwsoi_soil.at[p].set(_lwsoi_p),
        lwleaf_leaf=(
            mlcanopy_inst.lwleaf_leaf.at[p, _sl, isun]
            .set(lwleaf_new[_sl])
            .at[p, _sl, isha]
            .set(lwleaf_new[_sl])
        ),
        lwupw_profile=mlcanopy_inst.lwupw_profile.at[p, _sl].set(lwupw_new[_sl]),
        lwdwn_profile=mlcanopy_inst.lwdwn_profile.at[p, _sl].set(lwdwn_new[_sl]),
    )


def _Norman(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    grid: GridInfo = None,
) -> mlcanopy_type:
    """
    Norman (1979) tridiagonal longwave radiative transfer.

    Mirrors Fortran subroutine ``Norman`` (private, lines 47-195).

    Reference: Bonan (2019) *Climate Change and Terrestrial Ecosystem
    Modeling*, Chapter 14.

    The algorithm mirrors the solar :func:`MLSolarRadiationMod._Norman`
    structure but operates on a single spectral band (no waveband loop)
    and includes leaf thermal emission as an additional source term.

    **Optical properties** (Fortran lines 97-101):

    .. code-block:: none

        omega = 1 - emleaf(pft)    [scattering coefficient]
        rho   = omega              [intercepted radiation fully reflected]
        tau   = 0                  [no transmission of intercepted radiation]

    **Layer emission** (Fortran lines 103-107):

    .. code-block:: none

        lw_source(ic) = [emleaf*sb*T_sun^4 * fracsun
                         + emleaf*sb*T_sha^4 * (1-fracsun)]
                        * (1 - td(ic))

    **Tridiagonal system** (Fortran lines 109-158): same equation
    ordering as the solar Norman solver — soil upward (m=1), soil
    downward (m=2), then for ``ic = nbot, ..., ntop-1`` upward (odd)
    and downward (even), finally top-layer upward and downward — with
    leaf emission entering the right-hand-side ``dtri`` terms.

    **Leaf absorption** (Fortran lines 170-181): absorbed flux per
    unit leaf area is the same for sunlit and shaded fractions:

    .. code-block:: none

        lwabs = emleaf * (lwdwn(ic) + lwupw(icm1)) * (1-td(ic))
                - 2 * lw_source(ic)
        lwleaf(ic, isun) = lwleaf(ic, isha) = lwabs / dpai(ic)

    Energy conservation is checked to within 1e-3 W/m2.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; the following output fields
            are updated: ``lwup_canopy``, ``lwveg_canopy``,
            ``lwsoi_soil``, ``lwleaf_leaf``, ``lwupw_profile``,
            ``lwdwn_profile``.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    """Dispatch to JIT'd per-patch Norman kernel.

    ntop and nbot are extracted concretely here (outside JIT) and passed
    as static arguments to ``_Norman_patch``, enabling the Python loop
    bounds and ``jnp.arange(nbot, ntop+1)`` inside the kernel.
    """
    for fp in range(num_filter):
        p = grid.p if grid is not None else filter_patch[fp]
        ntop = grid.ntop if grid is not None else int(mlcanopy_inst.ntop_canopy[p])
        nbot = grid.nbot if grid is not None else int(mlcanopy_inst.nbot_canopy[p])
        mlcanopy_inst = _Norman_patch(p, ntop, nbot, mlcanopy_inst)
    return mlcanopy_inst
