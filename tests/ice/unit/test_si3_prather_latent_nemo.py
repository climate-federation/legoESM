"""Known-answer tests for the SI3 Prather latent deviations (oracle audit tier 4, U10 D3).

The tier-4 ORCA1 record (NEMO 5.0.1, ln_pnd=F) triggers neither the pond
branch nor a negative concentration, so these fixtures are small synthetic
columns whose expected values are transliterated from the cited NEMO lines.
Exact equality: the same IEEE operations run on both sides, and each case is
built so that only the branch under test changes the result.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import (
    ICE_ADV2D_TRACERS,
    apply_si3_prather_source_corrections,
)
from legoesm.ice.transport import advect_si3_prather_2d



@pytest.fixture(autouse=True)
def _fp64_policy():
    """fp64 storage + libm policy for this module only (restored afterwards)."""

    old_x64, old_policy = bool(jax.config.jax_enable_x64), get_policy()
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        yield
    finally:
        set_policy(old_policy)
        jax.config.update("jax_enable_x64", old_x64)


_TR = ICE_ADV2D_TRACERS
_N = 7  # 3x3 interior + halo 2
_C = 3  # centre cell (interior)
_HALO = 2
_HBIG_A = 0.15  # icedyn_adv_pra.F90:1035,1043,1050
_EPSI20 = 1.0e-20  # icedyn_adv_pra.F90:1034


def _background():
    """Uniform thick-ice background (no Hbig/Hsnow/zapneg branch binds)."""

    value = {name: 0.0 for name in _TR}
    value.update(v_i=1.8, v_s=0.18, a_i=0.9, oa_i=0.9, a_ip=0.2, v_ip=0.01)
    for k in (1, 2, 3):
        value[f"e_s_l{k:02d}"] = 1.0e7
        value[f"e_i_l{k:02d}"] = 3.0e8
        value[f"szv_i_l{k:02d}"] = 3.0
    return np.stack(
        [np.full((_N, _N), value[name], dtype=np.float64) for name in _TR], axis=-1
    )


_IN = (slice(_HALO, -_HALO), slice(_HALO, -_HALO))  # halos are a periodic refresh


def _correct(entry, transported, ponds_active):
    return np.asarray(
        apply_si3_prather_source_corrections(
            jnp.asarray(entry),
            jnp.asarray(transported),
            tracer_names=_TR,
            nlay_i=3,
            nlay_s=3,
            cell_area_m2=1.0,
            halo_width=_HALO,
            ponds_active=ponds_active,
            entry_intensive_contents=jnp.asarray(entry),
            contents_are_intensive=True,
        )
    )


def _pond_case():
    entry = _background()
    transported = entry.copy()
    i = _TR.index
    # Transported centre: thick pond (h_ip = 0.04/0.05 = 0.8 m > entry 9-point
    # max 0.05 m) on a low pond fraction (a_ip 0.05 < 0.15); ice itself thick.
    transported[_C, _C, i("a_ip")] = 0.05
    transported[_C, _C, i("v_ip")] = 0.04
    return entry, transported


def test_pond_hbig_skipped_when_ponds_off():
    """icedyn_adv_pra.F90:1033: ponds off -> a_ip untouched by Hbig_pra."""

    entry, transported = _pond_case()
    out = _correct(entry, transported, ponds_active=False)
    a_ip = _TR.index("a_ip")
    assert out[_C, _C, a_ip] == transported[_C, _C, a_ip]
    np.testing.assert_array_equal(out[_IN], transported[_IN])


def test_pond_hbig_applied_when_level_ponds_on():
    """icedyn_adv_pra.F90:1033-1037 (ln_pnd_LEV): a_ip = v_ip / zhip_max."""

    entry, transported = _pond_case()
    i = _TR.index
    h_ip_max = max(_EPSI20, entry[_C, _C, i("v_ip")] / entry[_C, _C, i("a_ip")])
    zhip = transported[_C, _C, i("v_ip")] / max(_EPSI20, transported[_C, _C, i("a_ip")])
    assert zhip > h_ip_max and transported[_C, _C, i("a_ip")] < _HBIG_A
    expected = transported.copy()
    expected[_C, _C, i("a_ip")] = transported[_C, _C, i("v_ip")] / h_ip_max
    out = _correct(entry, transported, ponds_active=True)
    np.testing.assert_array_equal(out[_IN], expected[_IN])


def _negative_area_case():
    entry = _background()
    transported = entry.copy()
    i = _TR.index
    # Centre after transport: v_i > 0 but a_i < 0 (Prather overshoot), no snow,
    # no pond.  ice_var_zapneg (icevar.F90:760-837): :760 leaves a_i (v_i > 0);
    # :768/:790/:797 zap salt/enthalpy (a_i <= 0); :809/:813 zap v_i/v_s;
    # :836 WHERE(pa_i < 0) pa_i = 0 ; :835 oa_i < 0 -> 0 ; :837 a_ip < 0 -> 0.
    for name in _TR:
        transported[_C, _C, i(name)] = 0.0
    transported[_C, _C, i("v_i")] = 2.0e-4
    transported[_C, _C, i("a_i")] = -1.0e-3
    return entry, transported


def test_zapneg_final_negative_area_reset():
    entry, transported = _negative_area_case()
    expected = transported.copy()
    expected[_C, _C, :] = 0.0  # every centre field zapped, a_i by :836
    for ponds_active in (False, True):
        out = _correct(entry, transported, ponds_active=ponds_active)
        np.testing.assert_array_equal(out[_IN], expected[_IN])


def test_zapneg_area_reset_has_finite_gradient():
    """Hard NEMO clamp (vote V2a): d a_i_out / d a_i = 0 below 0, 1 above."""

    entry, transported = _negative_area_case()
    i = _TR.index

    def centre_area(a):
        x = jnp.asarray(transported).at[_C, _C, i("a_i")].set(a)
        out = apply_si3_prather_source_corrections(
            jnp.asarray(entry), x, tracer_names=_TR, nlay_i=3, nlay_s=3,
            cell_area_m2=1.0, halo_width=_HALO, ponds_active=False,
            entry_intensive_contents=jnp.asarray(entry), contents_are_intensive=True,
        )
        return out[_C, _C, i("a_i")]

    for a, slope in ((-1.0e-3, 0.0), (0.5, 1.0)):
        g = float(jax.grad(centre_area)(a))
        h = 1.0e-7
        fd = (float(centre_area(a + h)) - float(centre_area(a - h))) / (2 * h)
        assert np.isfinite(g) and g == slope
        assert abs(fd - slope) < 1.0e-6


def _kernel_inputs():
    shape = (7, 7, 3)
    idx = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    content = jnp.asarray(1.0 + (idx % 37) / 53.0)
    area = jnp.full(shape[:2], 9.0e6)
    u = jnp.asarray(((idx[..., 0] % 7) - 3) * 0.05)
    v = jnp.asarray(((idx[..., 0] % 5) - 2) * 0.03)
    wet = jnp.ones(shape[:2], dtype=bool)
    moments = tuple(jnp.asarray(content * 0.01 * (m + 1)) for m in range(5))
    return content, moments, u, v, area, wet


def _kernel(dx, dy, subcycles):
    content, moments, u, v, area, wet = _kernel_inputs()
    return advect_si3_prather_2d(
        content, moments, u, v, area, wet, 1800.0, dx=dx, dy=dy,
        ice_step_index=1, halo_width=2, ice_volume_index=0,
        concentration_index=1, subcycles=subcycles,
    )


def test_array_spacings_require_resolved_subcycles():
    """CFL uses r1_e1u/r1_e2v (icedyn_adv_pra.F90:119-120), not e1v/e2u."""

    arr = jnp.full((7, 7), 3000.0)
    with pytest.raises(ValueError, match="cannot select the CFL"):
        _kernel(arr, arr, None)
    with pytest.raises(ValueError, match="shape"):
        _kernel(jnp.full((6, 7), 3000.0), arr, 1)


def test_uniform_array_spacings_match_scalar_path():
    scalar = _kernel(3000.0, 3000.0, 1)
    arr = jnp.full((7, 7), 3000.0)
    array = _kernel(arr, arr, 1)
    np.testing.assert_array_equal(np.asarray(array[0]), np.asarray(scalar[0]))
    for a, b in zip(array[1], scalar[1]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert _kernel(3000.0, 3000.0, None)[2] == 1  # scalar CFL path unchanged
