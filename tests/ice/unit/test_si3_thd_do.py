"""O14 ``ice_thd_do`` tests against a standalone scalar NumPy oracle."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ice import bitz_lipscomb as bl  # noqa: E402
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG  # noqa: E402


# Frozen independently from NEMO 5.0.2 phycst/eos and the admitted namelist.
T0 = 273.15
RHOI = 917.0
RCPI = 2096.7
RCP = 3991.86795711963
RLFUS = 333360.1
RTMLT = 0.054
SINEW = 0.75
EPS10 = 1.0e-10
EPS20 = 1.0e-20


def _inputs(*, vanished: bool = False):
    a = np.array([[0.62, 0.12], [0.0, 0.0] if vanished else [0.15, 0.30], [0.42, 0.20]])
    v = np.array([[0.58, 0.36], [0.0, 0.0] if vanished else [0.12, 0.75], [0.31, 0.44]])
    e = np.empty((3, 2, 2))
    szv = np.empty_like(e)
    for i in range(3):
        for k in range(2):
            for cat in range(2):
                e[i, k, cat] = (2.40e8 + 2.0e6 * k + 3.0e5 * cat) * v[i, cat] / 2.0
                szv[i, k, cat] = (4.0 + 0.4 * k + 0.2 * cat) * v[i, cat] / 2.0
    qlead = np.array([-1.0e6, -5.0e5 if vanished else 25.0, -7.5e5])
    ht_i_new = np.array([0.1, 0.1 if vanished else 0.0, 0.1])
    return {
        "a_i": a,
        "v_i": v,
        "sv_i": np.array([[2.4, 1.1], [0.0, 0.0] if vanished else [0.5, 2.0], [1.2, 1.8]]),
        "e_i": e,
        "szv_i": szv,
        "qlead": qlead,
        "t_bo": np.array([271.4, 271.6, 271.2]),
        "sfx_opw": np.array([0.3, -0.2, 0.1]),
        "wfx_opw": np.array([0.4, 0.2, -0.1]),
        "ht_i_new": ht_i_new,
        "fraz_frac": np.zeros(3),
        "hfx_thd": np.array([1.0, 2.0, 3.0]),
        "hfx_opw": np.array([-1.0, -2.0, -3.0]),
        "rn_amax_2d": np.full(3, 0.99999),
        "sss_m": np.array([34.2, 33.8, 35.0]),
        "hi_max": np.array([0.0, 0.2, math.inf]),
    }


def _remap_scalar(h_old, c_old, nlay):
    h0 = [0.0]
    c0 = [0.0]
    for h, c in zip(h_old, c_old):
        h0.append(h0[-1] + h)
        c0.append(c0[-1] + c)
    hnew = h0[-1] / nlay
    h1 = [k * hnew for k in range(nlay + 1)]
    c1 = [0.0] * (nlay + 1)
    for old in range(1, len(h_old) + 1):
        for new in range(1, nlay):
            if h1[new] <= h0[old] and h1[new] > h0[old - 1]:
                c1[new] = (
                    c0[old - 1] * (h0[old] - h1[new])
                    + c0[old] * (h1[new] - h0[old - 1])
                ) / (h0[old] - h0[old - 1])
    c1[-1] = c0[-1]
    return np.array([max(c1[k] - c1[k - 1], 0.0) / max(hnew, EPS20) for k in range(1, nlay + 1)])


def _scalar_oracle(x):
    out = {name: np.array(x[name], dtype=np.float64, copy=True) for name in (
        "a_i", "v_i", "sv_i", "e_i", "szv_i", "sfx_opw", "wfx_opw", "hfx_thd", "hfx_opw"
    )}
    out["at_i"] = np.array([sum(row) for row in x["a_i"]], dtype=np.float64)
    nlay = x["e_i"].shape[-2]
    for i, qlead in enumerate(x["qlead"]):
        if not qlead < 0.0:
            continue
        old_a = x["a_i"][i].copy()
        old_v = x["v_i"][i].copy()
        e = np.array([
            [x["e_i"][i, k, c] / old_v[c] * nlay if old_v[c] > 0.0 else 0.0 for c in range(2)]
            for k in range(nlay)
        ])
        sz = np.array([
            [x["szv_i"][i, k, c] / old_v[c] * nlay if old_v[c] > 0.0 else 0.0 for c in range(2)]
            for k in range(nlay)
        ])
        zs = SINEW * x["sss_m"][i]
        ztm = -RTMLT * zs
        tbo_c = x["t_bo"][i] - T0
        ze = RHOI * (RCPI * (ztm - tbo_c) + RLFUS * max(0.0, 1.0 - ztm / min(tbo_c, -EPS10)) - RCP * ztm)
        zei = -ze / RHOI
        zew = RCP * tbo_c
        zde = zei - zew
        zfmdt = -qlead / zde
        zv = -zfmdt / RHOI
        out["hfx_thd"][i] += zfmdt * zew / 21600.0
        out["hfx_opw"][i] -= zfmdt * zde / 21600.0
        out["wfx_opw"][i] -= zv * RHOI / 21600.0
        out["sfx_opw"][i] -= zv * RHOI * zs / 21600.0
        za = zv / x["ht_i_new"][i]
        available = max(0.0, x["rn_amax_2d"][i] - out["at_i"][i])
        if za > available:
            da_res = za - available
            dv_res = da_res * x["ht_i_new"][i]
            za = max(0.0, za - da_res)
            zv = max(0.0, zv - dv_res)
        else:
            dv_res = 0.0
        selected = next(c for c in range(2) if x["ht_i_new"][i] > x["hi_max"][c] and x["ht_i_new"][i] <= x["hi_max"][c + 1])
        out["a_i"][i, selected] += za
        out["v_i"][i, selected] += zv
        if old_a[selected] > 0.0:
            e[:, selected] = (ze * zv + e[:, selected] * old_v[selected]) / max(out["v_i"][i, selected], EPS20)
            sz[:, selected] = (zs * zv + sz[:, selected] * old_v[selected]) / max(out["v_i"][i, selected], EPS20)
        else:
            e[:, selected] = ze
            sz[:, selected] = zs
        total_area = sum(out["a_i"][i])
        for c in range(2):
            new_bottom = dv_res * out["a_i"][i, c] / max(total_area, EPS20) if total_area >= EPS20 else 0.0
            lateral_v = out["v_i"][i, c]
            h_old = [lateral_v / nlay] * nlay + [new_bottom]
            e_old = [e[k, c] * lateral_v / nlay for k in range(nlay)] + [ze * new_bottom]
            sz_old = [sz[k, c] * lateral_v / nlay for k in range(nlay)] + [zs * new_bottom]
            out["v_i"][i, c] += new_bottom
            out["sv_i"][i, c] += zs * (out["v_i"][i, c] - old_v[c])
            out["e_i"][i, :, c] = _remap_scalar(h_old, e_old, nlay) * out["v_i"][i, c] / nlay
            out["szv_i"][i, :, c] = _remap_scalar(h_old, sz_old, nlay) * out["v_i"][i, c] / nlay
    return out


def _call(x):
    return bl.si3_ice_thd_do(
        **{k: jnp.asarray(v, dtype=jnp.float64) for k, v in x.items()},
        nn_icesal=4,
        rn_sinew=SINEW,
        r1_Dt_ice=1.0 / 21600.0,
        r1_nlay_i=0.5,
        constants=NEMO_SI3_CONSTANTS_CONFIG,
    )


def test_scalar_oracle_category_choice_fluxes_and_all_state_fields():
    x = _inputs()
    want = _scalar_oracle(x)
    got = jax.jit(_call)(x)
    for name in got._fields:
        np.testing.assert_allclose(np.asarray(getattr(got, name)), want[name], rtol=2e-14, atol=2e-12)
    assert np.all(np.asarray(got.a_i)[[0, 2], 0] > x["a_i"][[0, 2], 0])
    assert np.array_equal(np.asarray(got.a_i)[1], x["a_i"][1])


def test_real_eager_equals_jit_with_runtime_arguments_and_vanished_categories():
    for vanished in (False, True):
        x = _inputs(vanished=vanished)
        compiled = jax.jit(_call)(x)
        with jax.disable_jit():
            eager = _call(x)
        for a, b in zip(compiled, eager):
            assert np.array_equal(np.asarray(a).view(np.uint64), np.asarray(b).view(np.uint64))
            assert np.all(np.isfinite(np.asarray(a)))


@pytest.mark.parametrize(
    ("name", "index", "eps"),
    [
        ("qlead", (0,), 1.0e-2),
        ("a_i", (0, 1), 1.0e-6),
        ("v_i", (0, 1), 1.0e-6),
        ("sv_i", (0, 1), 1.0e-6),
        ("e_i", (0, 0, 1), 1.0e2),
        ("szv_i", (0, 0, 1), 1.0e-6),
    ],
)
def test_gradients_finite_and_match_central_difference_away_from_switches(
    name, index, eps
):
    base = _inputs(vanished=True)
    base = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in base.items()}

    def loss(value):
        out = _call({**base, name: value})
        return (
            jnp.sum(out.a_i)
            + jnp.sum(out.v_i)
            + 0.1 * jnp.sum(out.sv_i)
            + 1.0e-10 * jnp.sum(out.e_i)
            + 0.01 * jnp.sum(out.szv_i)
        )

    value = base[name]
    grad = jax.jit(jax.grad(loss))(value)
    assert np.all(np.isfinite(np.asarray(grad)))
    direction = jnp.zeros_like(value).at[index].set(1.0)
    central = (loss(value + eps * direction) - loss(value - eps * direction)) / (2.0 * eps)
    analytic = jnp.sum(grad * direction)
    assert math.isclose(
        float(analytic), float(central), rel_tol=2.0e-6, abs_tol=2.0e-10
    )


def test_column_step_vanished_ice_has_finite_concentration_and_thickness_gradients():
    from legoesm.ice.c1d_omip_l3 import build_c1d_omip_l3_card

    card = build_c1d_omip_l3_card()
    z1 = jnp.zeros(1, dtype=jnp.float64)
    z3 = jnp.zeros((1, 3), dtype=jnp.float64)
    state = bl.SI3ColumnArrays(z1, z1, z1, z1, z3, z3, z1, z3, z1)
    forcing = bl.SI3SurfaceForcing(
        qns_ice=z1,
        qsr_ice=z1,
        dqns_ice=z1,
        qtr_ice_top=z1,
        t_bottom=jnp.asarray([271.35], dtype=jnp.float64),
        sss=jnp.asarray([34.0], dtype=jnp.float64),
        evaporation=z1,
        snow_precipitation=z1,
        qprec_ice=z1,
        qcn_ice_bottom=z1,
        qsb_ice_bottom=z1,
        fhld=z1,
        qlead=z1,
    )

    def loss(concentration, h_ice):
        trace = bl.si3_column_step_arrays(
            state._replace(concentration=concentration, h_ice=h_ice),
            forcing,
            card.dt_seconds,
            NEMO_SI3_CONSTANTS_CONFIG,
            si3=card.config.si3,
            t_i_entry=z3,
            t_s_entry=z3,
            _zdf_dh_override=(
                state,
                state._replace(concentration=concentration, h_ice=h_ice),
                state.S_bulk,
                state.S_layers,
            ),
        )
        return jnp.sum(trace.post_do.concentration + trace.post_do.h_ice)

    grads = jax.jit(jax.grad(loss, argnums=(0, 1)))(z1, z1)
    for grad in grads:
        assert np.all(np.isfinite(np.asarray(grad)))


def test_no_defaults_shape_refusals_and_object_code_tie_roles():
    x = _inputs()
    with pytest.raises(TypeError):
        bl.si3_ice_thd_do(
            **{k: jnp.asarray(v) for k, v in x.items()},
            nn_icesal=4,
            rn_sinew=SINEW,
            r1_Dt_ice=1.0 / 21600.0,
            r1_nlay_i=0.5,
        )  # constants is mandatory
    with pytest.raises(ValueError, match="nn_icesal"):
        bl.si3_ice_thd_do(
            **{k: jnp.asarray(v) for k, v in x.items()},
            nn_icesal=3,
            rn_sinew=SINEW,
            r1_Dt_ice=1.0 / 21600.0,
            r1_nlay_i=0.5,
            constants=NEMO_SI3_CONSTANTS_CONFIG,
        )
    with pytest.raises(ValueError, match="hi_max shape"):
        _call({**x, "hi_max": np.array([0.0, math.inf])})

    # Exact O14 object: all executing MIN/MAX sites keep src on a numeric tie.
    # Bit-distinct signed zeros make the survivor observable; swapping operands
    # is the plant and flips every result bit.
    dst = jnp.asarray([-0.0, 0.0], dtype=jnp.float64)
    src = jnp.asarray([0.0, -0.0], dtype=jnp.float64)
    sign_bit = np.uint64(1 << 63)
    expected = np.asarray([np.uint64(0), sign_bit], dtype=np.uint64)
    for helper in (bl._maxsd, bl._minsd):
        got = np.asarray(jax.jit(helper)(dst, src)).view(np.uint64)
        swapped = np.asarray(jax.jit(lambda d, s: helper(s, d))(dst, src)).view(np.uint64)
        np.testing.assert_array_equal(got, expected)
        assert np.all(swapped != got)
