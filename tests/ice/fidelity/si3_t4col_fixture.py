"""Loader for the tier-4 SI3 column oracle fixtures (NEMO 5.0.1 SI3, ORCA1_T4COL, kt=17281).

Each ``fixtures/si3_t4col/<name>.npz`` holds a few recorded columns: every input
(``<stage>:in:<f>``) and NEMO output (``<stage>:nemo:<f>``) of ice_thd_zdf, ice_thd_dh and
ice_thd_sal, the main-branch port result at extraction (``<stage>:main:<f>``), and the
per-cell tolerance ``<stage>:tol:<f>`` = max(20-seed 1-ulp spread, 1e-12) of the pick-15
relative error.  Provenance (record path, recorder cfg, harness SHA) is in ``<name>.json``.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from legoesm.ice import bitz_lipscomb as bl

FIXDIR = Path(__file__).parent / "fixtures" / "si3_t4col"


def load(name: str) -> dict:
    with np.load(FIXDIR / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def col(fx: dict, key: str) -> np.ndarray:
    """(ncell,) for 1-D fields; (ncell, nlay) for layered ones."""
    v = fx[key]
    return v[:, 0] if v.ndim == 2 and v.shape[1] == 1 else v


def dt(fx: dict) -> float:
    return float(fx["dh:nemo:rDt_ice"][0])


def forcing(n: int, **kw) -> bl.SI3SurfaceForcing:
    z = jnp.zeros(n)
    f = {k: z for k in bl.SI3SurfaceForcing._fields if bl.SI3SurfaceForcing._field_defaults.get(k, 0) is not None}
    f.update({k: jnp.asarray(v) for k, v in kw.items()})
    return bl.SI3SurfaceForcing(**f)


def zdf_args(fx: dict):
    i = lambda k: jnp.asarray(col(fx, f"zdf:in:{k}"))  # noqa: E731
    n = i("h_i").shape[0]
    fz = forcing(n, qns_ice=i("qns_ice"), qsr_ice=i("qsr_ice"), dqns_ice=i("dqns_ice"),
                 qtr_ice_top=i("qtr_ice_top"), t_bottom=i("t_bo"))
    return (i("e_i"), i("e_s"), i("sz_i"), i("h_i"), i("h_s"), i("t_su"), fz)


def dh_args(fx: dict):
    """(state, zdf_result, forcing) exactly as the tier-4 literal row feeds _dh_step."""
    i = lambda k: jnp.asarray(col(fx, f"dh:in:{k}"))  # noqa: E731
    n = i("h_i").shape[0]
    st = bl.SI3ColumnArrays(i("a_i"), i("h_i"), i("h_s"), i("t_su"), i("e_i"), i("e_s"),
                            i("s_i"), i("sz_i"), jnp.zeros(n))
    zr = bl.SI3ZDFResult(i("t_su"), i("t_i"), i("t_s"), i("e_i"), i("e_s"), i("qns_ice"),
                         i("qtr_ice_bot"), i("qcn_ice_top"), i("qcn_ice_bot"), jnp.zeros(n, jnp.int32))
    kw = dict(qsr_ice=i("qsr_ice"), qtr_ice_top=i("qtr_ice_top"), t_bottom=i("t_bo"), sss=i("sss"),
              evaporation=i("evap_ice"), snow_precipitation=i("sprecip"), qprec_ice=i("qprec_ice"),
              qsb_ice_bottom=i("qsb_ice_bot"), fhld=i("fhld"), qns_ice=i("qns_ice"))
    if "sst" in bl.SI3SurfaceForcing._fields:
        kw["sst"] = i("sst")
    return st, zr, forcing(n, **kw)


def rel(a, b) -> np.ndarray:
    """pick 15 per cell: max over layers of |a-b|/max(|a|,|b|), 0 where both are 0."""
    a = np.asarray(a, float).reshape(len(a), -1)
    b = np.asarray(b, float).reshape(len(b), -1)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    den = np.maximum(np.abs(a), np.abs(b))
    return np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0).max(axis=1)


def within(fx: dict, stage: str, field: str, port) -> np.ndarray:
    """Per-cell pass mask of the port value against NEMO at the stored tolerance."""
    return rel(port, col(fx, f"{stage}:nemo:{field}")) <= fx[f"{stage}:tol:{field}"]


def within_flux(fx: dict, stage: str, field: str, acc) -> np.ndarray:
    """Accumulator pass mask (vote W1a): per-cell spread OR |diff| <= 1e-12 x record-wide max contribution.

    The absolute floor admits NEMO's FMA-contraction residuals (~1e-22 kg m-2 s-1 where the
    port gives exactly 0); it is 1e-12 of the largest contribution of that field in the record.
    """
    nemo = col(fx, f"{stage}:nemo:{field}")
    floor = 1e-12 * float(fx[f"{stage}:scale:{field}"])
    return within(fx, stage, field, acc) | (np.abs(np.asarray(acc) - nemo) <= floor)
