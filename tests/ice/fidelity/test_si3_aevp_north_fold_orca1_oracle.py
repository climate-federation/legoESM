"""ORCA1 oracle for the SI3 C-grid aEVP north-fold lateral boundary (tier-4 U9, defect D1).

Fixture ``fixtures/orca1_t4/aevp_fold_band.{npz,json}``: the top band of NEMO
5.0.1 SI3 ORCA1 (F-pivot fold, nn_hls=2) at kt=17281, full width with NEMO's
recorded halos -- every aEVP input plus NEMO's subcycle-1 outputs.  Provenance
(record path, build job, assembly, spread) is in the JSON.

Extended band rows (axis 1): 0-1 recorded rows below the band (south pad),
2-9 the eight top inner rows (9 = last global row), 10-11 NEMO's north halo.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice import dynamics as dyn

_FIX = Path(__file__).parent / "fixtures" / "orca1_t4" / "aevp_fold_band.npz"
_HALO = 2
# Rows scored after one subcycle: the band's bottom 3 inner rows are polluted
# by the zero south pad (each subcycle reads ~2 rows), so score the top five.
_SCORED_ROWS = slice(5, 10)
# Tolerance = 10x the field's 20-seed NEMO spread (vote V5 a).
_SPREAD_FACTOR = 10.0


@pytest.fixture(scope="module")
def band():
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    data = np.load(_FIX)
    prov = json.loads(_FIX.with_suffix(".json").read_text())
    return data, prov


def _inputs(data, north_fold, n_subcycles=1):
    g = lambda k: jnp.asarray(data["in_" + k])  # noqa: E731
    s = lambda k: float(data["sc_" + k])  # noqa: E731
    metrics = dyn.SI3CGridMetrics(
        e1t=g("e1t"),
        e2t=g("e2t"),
        e1u=g("e1u"),
        e2u=g("e2u"),
        e1v=g("e1v"),
        e2v=g("e2v"),
        e1f=g("e1f"),
        e2f=g("e2f"),
        area_t=g("e1e2t"),
        area_u=g("e1e2u"),
        area_v=g("e1e2v"),
        area_f=g("e1e2f"),
    )
    # key_qco depth macros: h = h_0 * (1 + r3)
    depth = {p: g(f"h{p}_0") * (1.0 + g(f"r3{p}_Kmm")) for p in "tuv"}
    forcing = dyn.SI3CGridAEVPForcing(
        concentration_t=g("at_i"),
        ice_volume_t=g("vt_i"),
        snow_volume_t=g("vt_s"),
        pond_volume_t=g("vt_ip"),
        lid_volume_t=g("vt_il"),
        air_stress_u_t=g("utau_ice"),
        air_stress_v_t=g("vtau_ice"),
        drag_io_t=g("drag_io"),
        ocean_u_u=g("u_oce"),
        ocean_v_v=g("v_oce"),
        ssh_t=g("ssh_m"),
        coriolis_t=g("ff_t"),
        tmask_t=g("tmask1"),
        umask_u=g("umask1"),
        vmask_v=g("vmask1"),
        fast_tmask=g("fast_tmask"),
        depth_t=depth["t"],
        depth_u=depth["u"],
        depth_v=depth["v"],
        iceberg_tmask=g("icb_tmask"),
        iceberg_umask=g("icb_umask"),
        iceberg_vmask=g("icb_vmask"),
    )
    state = dyn.SI3CGridAEVPState(
        u_ice_u=g("u_ice"),
        v_ice_v=g("v_ice"),
        stress1_t=g("stress1_i"),
        stress2_t=g("stress2_i"),
        stress12_f=g("stress12_i"),
    )
    config = dyn.SI3CGridAEVPConfig(
        scheme="si3_aevp",
        staggering="si3_c_grid",
        dt_s=s("rDt_ice"),
        n_subcycles=n_subcycles,
        eccentricity=s("rn_ecc"),
        creep_limit_s_inv=s("rn_creepl"),
        strength_parameter_pa=s("rn_pstar"),
        strength_decay=s("rn_crhg"),
        rho_snow=s("rhos"),
        rho_ice=s("rhoi"),
        rho_water=s("rhow"),
        rho_ocean=s("rho0"),
        gravity=s("grav"),
        rn_ishlat=s("rn_ishlat"),
        halo_width=_HALO,
        category_count=1,
        landfast=True,
        landfast_depth_fraction=s("rn_lf_depfra"),
        landfast_basal_friction_n_m3=s("rn_lf_bfr"),
        landfast_relaxation_s_inv=s("rn_lf_relax"),
        landfast_tensile_fraction=s("rn_lf_tensile"),
        convergence_check=0,
        north_fold=north_fold,
    )
    return state, forcing, metrics, config


def _rel(a, b):
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise FloatingPointError("non-finite operand in the oracle score")
    den = np.maximum(np.abs(a), np.abs(b))
    return np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0)


def _worst(data, north_fold):
    out = dyn.si3_cgrid_aevp_solver(*_inputs(data, north_fold), limit_sharpness_scale=math.inf)  # NEMO hard limits
    pairs = {
        "sub1_u_ice": out.u_ice_u,
        "sub1_v_ice": out.v_ice_v,
        "sub1_s1": out.stress1_t,
        "sub1_s2": out.stress2_t,
        "sub1_s12": out.stress12_f,
    }
    inner_i = slice(_HALO, -_HALO)
    return {
        k: float(
            _rel(
                np.asarray(p)[inner_i, _SCORED_ROWS], data["out_" + k][inner_i, _SCORED_ROWS]
            ).max()
        )
        for k, p in pairs.items()
    }


# (fixture key, point type, sign, compare E-W halo): the lbc'ed fields whose
# recorded halos NEMO wrote.  e1u/e1f E-W halos carry NEMO geometry, not lbc.
_KNOWN = (
    ("in_e1t", "T", 1.0, True),
    ("in_e1u", "U", 1.0, False),
    ("in_e1v", "V", 1.0, True),
    ("in_e1f", "F", 1.0, False),
    ("out_sub1_u_ice", "U", -1.0, True),
    ("out_sub1_v_ice", "V", -1.0, True),
)


@pytest.mark.parametrize("key,point,sign,ew", _KNOWN)
def test_fold_halo_reproduces_nemo_recorded_halo(band, key, point, sign, ew):
    data, _ = band
    recorded = data[key]
    rebuilt = np.asarray(dyn.si3_orca_f_pivot_halo(jnp.asarray(recorded), point, sign))
    north = (slice(None), slice(-_HALO, None))
    np.testing.assert_array_equal(rebuilt[north], recorded[north])
    if ew:
        cols = np.r_[0:_HALO, -_HALO:0]
        inner_j = slice(_HALO, -_HALO)
        np.testing.assert_array_equal(rebuilt[cols, inner_j], recorded[cols, inner_j])


def test_band_solve_matches_nemo_subcycle1(band):
    data, prov = band
    worst = _worst(data, "orca_f_pivot")
    tol = {k: _SPREAD_FACTOR * v for k, v in prov["spread_all_max"].items()}
    assert all(worst[k] <= tol[k] for k in worst), (worst, tol)


def test_periodic_halo_is_detected_by_the_band_check(band):
    # Planted violation: the doubly periodic lbc must fail the same check.
    data, prov = band
    worst = _worst(data, "none")
    tol = {k: _SPREAD_FACTOR * v for k, v in prov["spread_all_max"].items()}
    assert worst["sub1_u_ice"] > 1.0e3 * tol["sub1_u_ice"]
    assert worst["sub1_v_ice"] > 1.0e3 * tol["sub1_v_ice"]


def test_score_rejects_non_finite():
    with pytest.raises(FloatingPointError):
        _rel(np.array([np.nan]), np.array([1.0]))


def test_default_is_orca_fold_and_unknown_raises(band):
    data, _ = band
    assert dyn.SI3CGridAEVPConfig._field_defaults["north_fold"] == "orca_f_pivot"
    state, forcing, metrics, config = _inputs(data, "tripole_typo")
    with pytest.raises(ValueError, match="unknown SI3 north_fold"):
        dyn.si3_cgrid_aevp_solver(state, forcing, metrics, config)
    with pytest.raises(ValueError, match="unknown SI3 lbc point type"):
        dyn.si3_orca_f_pivot_halo(jnp.zeros((8, 8)), "W", 1.0)


@pytest.mark.parametrize("point,sign", (("T", 1.0), ("U", -1.0), ("V", -1.0), ("F", 1.0)))
def test_fold_halo_gradients(point, sign):
    value = jax.random.normal(jax.random.PRNGKey(0), (10, 9), jnp.float64)
    check_grads(
        lambda x: dyn.si3_orca_f_pivot_halo(x, point, sign),
        (value,),
        order=2,
        modes=("fwd", "rev"),
    )


def test_solver_gradient_through_fold_matches_finite_difference(band):
    data, _ = band
    state, forcing, metrics, config = _inputs(data, "orca_f_pivot", n_subcycles=2)
    top = (slice(_HALO, -_HALO), _SCORED_ROWS)

    def loss(scale):
        s = state._replace(u_ice_u=state.u_ice_u * scale, v_ice_v=state.v_ice_v * scale)
        out = dyn.si3_cgrid_aevp_solver(s, forcing, metrics, config, differentiable=True)
        return jnp.sum(out.v_ice_v[top] ** 2) + jnp.sum(out.u_ice_u[top] ** 2)

    grad = float(jax.grad(loss)(1.0))
    eps = 1.0e-6
    fd = (float(loss(1.0 + eps)) - float(loss(1.0 - eps))) / (2.0 * eps)
    assert np.isfinite(grad) and grad != 0.0
    assert abs(grad / fd - 1.0) < 1.0e-5, (grad, fd)
