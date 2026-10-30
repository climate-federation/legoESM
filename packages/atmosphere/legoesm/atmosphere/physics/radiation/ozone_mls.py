"""SAM RCEMIP ozone: gSAM's MLS standard profile as model-layer means.

ORACLE.  gSAM 1.8.8 ``SRC/RAD_RRTM/rad.f90`` ``tracesini`` (lines 1219-1431):
gSAM's RRTM radiation reads the mid-latitude-summer (MLS) trace-gas
soundings from ``RUNDATA/rrtmg_lw.nc`` (``Pressure``, ``AbsorberAmountMLS``),
integrates the ozone PATH from the surface up through the model interfaces
plus one extra layer above the model top (top interface
``min(1e-4 hPa, 0.25*0.5*p_top)``), with the sounding linear in pressure
between its nodes and held constant beyond its ends, and differences the
paths so each layer gets its mass-weighted MEAN mixing ratio.  The namelist
factor is 1 for RCEMIP.  :func:`mls_ozone_layer_vmr` ports that algorithm in
the oracle's operation order and is checked against the running oracle in
``tests/atmosphere/test_ozone_mls_oracle.py``.

The bundled table is ``rrtmg_lw.nc`` (sha256 4458c778...93d9) at full
precision: ``Pressure`` [hPa] and the O3 column of ``AbsorberAmountMLS``
[mol/mol], 59 nodes, surface first.  gSAM zeroes entries > 2 (none for O3).

This is the SAM/gSAM MLS ozone (what ``nxco2/factor=1`` RCEMIP uses) — NOT the
RCEMIP-protocol analytic ozone (Wing et al. 2018, ``g1·p^g2·exp(-p/g3)``).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from legoesm import constants

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "SAM/gSAM RCEMIP mid-latitude-summer (MLS) standard-atmosphere ozone: "
        "gSAM tracesini path-differenced layer-mean O3 VMR on the model "
        "layers (plus the layer above the model top) as the RRTMGP ozone "
        "absorber."
    ),
    "inputs": {"p_half": "Pa (interface pressure, index 0 = model top)"},
    "outputs": {"o3_vmr": "mol/mol (layer-mean ozone volume mixing ratio)"},
    "sign_convention": (
        "o3_vmr>0; a positive mole fraction peaking in the lower stratosphere; "
        "the sounding is held at its end values beyond [0.00966, 1053.63] hPa. "
        "Diagnoses an absorber field only (no budget)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "AFGL mid-latitude-summer standard atmosphere (Anderson et al. 1986); "
        "gSAM 1.8.8 RAD_RRTM/rad.f90 tracesini with RUNDATA/rrtmg_lw.nc "
        "AbsorberAmountMLS (O3)."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ozone_mls_oracle.py: layer means match the "
        "running gSAM tracesini to 1e-10 on three grids; tests/unit/"
        "test_mls_ozone.py: column path is grid-invariant."
    ),
}

# --- MLS sounding, gSAM RUNDATA/rrtmg_lw.nc (surface -> top) ---
# Pressure [hPa]
_MLS_P_HPA = (
    1053.63, 862.642, 706.272, 578.246, 473.428, 387.61, 317.348, 259.823,
    212.725, 174.164, 142.594, 116.746, 95.5835, 78.2571, 64.0715, 52.4573,
    42.9484, 35.1632, 28.7892, 23.5706, 19.298, 15.7998, 12.9358, 10.591,
    8.67114, 7.09933, 5.81244, 4.75882, 3.89619, 3.18993, 2.6117, 2.13828,
    1.75067, 1.43333, 1.17351, 0.960789, 0.786628, 0.644036, 0.527292,
    0.43171, 0.353455, 0.289384, 0.236928, 0.19398, 0.158817, 0.130029,
    0.106458, 0.0871608, 0.0713612, 0.0584256, 0.0478349, 0.0391639,
    0.0320647, 0.0262523, 0.0214936, 0.0175975, 0.0144076, 0.0117959,
    0.00965769,
)
# O3 volume mixing ratio [mol/mol] (AbsorberAmountMLS, absorber "O3")
_MLS_O3_VMR = (
    1.7351146061445684e-08, 1.8594253528211756e-08, 2.315958209318113e-08,
    2.977607283479713e-08, 3.6160692454954786e-08, 4.03765567598473e-08,
    4.0419242276651843e-08, 3.972259096079735e-08, 3.927787357530583e-08,
    3.9965961952221263e-08, 4.324244438418727e-08, 6.223225792007951e-08,
    1.4326748886742276e-07, 3.833234821927476e-07, 8.636481488650343e-07,
    1.4865046967488144e-06, 2.2224979491558506e-06, 3.2582534827057087e-06,
    4.65608101977412e-06, 6.263463249715041e-06, 7.694613833863467e-06,
    8.616951869010989e-06, 9.334983949150036e-06, 9.766387973227697e-06,
    9.964502341540206e-06, 9.854079301724168e-06, 9.468470341438533e-06,
    8.864065331820452e-06, 8.06590770780332e-06, 7.132535402553212e-06,
    6.153457476755871e-06, 5.252318231359983e-06, 4.500507279439954e-06,
    3.832681305095953e-06, 3.242876157647763e-06, 2.735217566682869e-06,
    2.314578125921976e-06, 1.9813845468617263e-06, 1.715660728078452e-06,
    1.501086740653173e-06, 1.3296361983515985e-06, 1.1900842378111121e-06,
    1.0664678127621685e-06, 9.650660136603498e-07, 8.805738209341674e-07,
    8.113999380401784e-07, 7.392325282634074e-07, 6.791270017275985e-07,
    6.257790755193067e-07, 5.79229656546301e-07, 5.371014609318492e-07,
    4.6701738877289174e-07, 4.0963751966080874e-07, 3.409109923886526e-07,
    2.711398120755099e-07, 2.152125676507284e-07, 1.8114569410404434e-07,
    1.5325376845061487e-07, 1.595950432797674e-07,
)
# --- gSAM tracesini extra layer (rad.f90: tmp_presi(nz+1) = MIN(1.e-4, 0.25*tmp_pres(nz))) ---
_TOP_INTERFACE_CAP_HPA = 1.0e-4

# Keep NumPy constants: this module can be imported lazily while a step is
# being traced, and JAX arrays built at import would cache tracers.
_P_TABLE = np.asarray(_MLS_P_HPA, dtype=np.float64)
_O3_TABLE = np.asarray(_MLS_O3_VMR, dtype=np.float64)


def mls_ozone_layer_vmr(p_half: jnp.ndarray):
    """gSAM ``tracesini`` ozone on the layers bounded by ``p_half``.

    Parameters
    ----------
    p_half : jnp.ndarray
        ``(ncol, nlev+1)`` interface pressure [Pa], index 0 = model top.

    Returns
    -------
    o3_layers : jnp.ndarray
        ``(ncol, nlev)`` layer-mean O3 VMR [mol/mol], index 0 = model top.
    o3_above : jnp.ndarray
        ``(ncol,)`` mean O3 VMR of gSAM's extra layer above the model top,
        between ``p_half[:, 0]`` and ``min(1e-4 hPa, p_half[:, 0]/8)``.  Not
        finite when ``p_half[:, 0] == 0`` (that layer then has no thickness).

    The operations follow the oracle's order (surface-up cumulative path,
    ``(dp/ggr)*tr`` per piece, difference times ``ggr/dp``) with gSAM's
    ``ggr`` (``constants.g_gsam``; it cancels analytically), so round-off
    matches.
    """
    p_half = jnp.asarray(p_half)
    dt = p_half.dtype
    pm = jnp.asarray(_P_TABLE, dtype=dt)
    tr = jnp.asarray(_O3_TABLE, dtype=dt)
    ggr = jnp.asarray(constants.g_gsam, dtype=dt)
    n_press = _P_TABLE.shape[0]

    presi = p_half[:, ::-1] / 100.0                  # (ncol, nz) hPa, surface first
    top = jnp.minimum(_TOP_INTERFACE_CAP_HPA, 0.25 * (0.5 * presi[:, -1]))
    tmp_presi = jnp.concatenate([presi, top[:, None]], axis=1)   # (ncol, nz+1)

    def step(path, bounds):
        a, b = bounds                    # tmp_presi(k-1) (lower), tmp_presi(k) (upper)
        # below the sounding: concentration at its bottom node
        path = path + jnp.where(a > pm[0], (a - jnp.maximum(b, pm[0])) / ggr * tr[0], 0.0)
        for m in range(1, n_press):
            plow = jnp.minimum(a, jnp.maximum(b, pm[m - 1]))
            pupp = jnp.minimum(a, jnp.maximum(b, pm[m]))
            pmid = 0.5 * (plow + pupp)
            wgtlow = (pmid - pm[m]) / (pm[m - 1] - pm[m])
            wgtupp = (pm[m - 1] - pmid) / (pm[m - 1] - pm[m])
            path = path + jnp.where(
                plow > pupp,
                (plow - pupp) / ggr * (wgtlow * tr[m - 1] + wgtupp * tr[m]), 0.0)
        # above the sounding: concentration at its top node
        path = path + jnp.where(b < pm[n_press - 1],
                                (jnp.minimum(a, pm[n_press - 1]) - b) / ggr * tr[n_press - 1],
                                0.0)
        return path, path

    _, paths = jax.lax.scan(step, jnp.zeros(p_half.shape[0], dtype=dt),
                            (tmp_presi[:, :-1].T, tmp_presi[:, 1:].T))
    trpath = jnp.concatenate([jnp.zeros((1, p_half.shape[0]), dtype=dt), paths])  # (nz+1, ncol)
    # Model layers and the extra layer are formed separately so a zero-thickness
    # extra layer (p_top = 0) cannot leak a non-finite value into the layers'
    # gradient.
    nz = presi.shape[1]
    godp = ggr / (presi[:, :-1] - presi[:, 1:])      # (ncol, nz-1)
    o3_layers = (trpath[1:nz] - trpath[:nz - 1]).T * godp
    o3_above = (trpath[nz] - trpath[nz - 1]) * (ggr / (presi[:, -1] - top))
    return o3_layers[:, ::-1], o3_above
