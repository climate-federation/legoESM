from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.ocean import freshwater as freshwater_module
from legoesm.ocean.forcing import nemo_surface as surface_module
from legoesm.ocean.forcing.nemo_fld_read import (
    nemo_fld_model_time,
    nemo_fld_record_centres,
    nemo_fld_select_bracket,
    nemo_fld_time_interpolate,
)
from legoesm.ocean.forcing.nemo_surface import (
    NemoSurfaceMeans,
    nemo_orca2_s2,
    nemo_sbc_ssm_step,
    nemo_s2_to_ocean_interfaces,
)


def _bits(value):
    return np.asarray(value, dtype=np.float64).view(np.uint64)


def _zeros(shape=(2, 3)):
    return jnp.zeros(shape, dtype=jnp.float64)


def _ssm_kwargs(value):
    zero = jnp.zeros_like(value)
    one = jnp.ones_like(value)
    return dict(
        nit000=1,
        forcing_stride=2,
        restart_means_present=False,
        use_conservative_temperature=False,
        remove_inverse_barometer=False,
        u_before=value,
        v_before=value + 1.0,
        temperature_now=value + 2.0,
        salinity_now=value + 30.0,
        ssh_now=value / 10.0,
        ssh_inverse_barometer=zero,
        e3t_reference=10.0 * one,
        r3t_now=value / 100.0,
        surface_mask=one,
        shortwave_fraction_top=value / 20.0,
    )


def _s2_kwargs(sss_mean=None):
    shape = (2, 3)
    one = jnp.ones(shape, dtype=jnp.float64)
    return dict(
        kt=1,
        forcing_stride=2,
        post_ice_utau=jnp.asarray([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]),
        post_ice_vtau=jnp.asarray([[0.7, 0.8, -0.9], [1.0, -1.1, 1.2]]),
        post_ice_utau_east=jnp.asarray([[0.2, 0.3, 0.4], [-0.5, 0.6, 0.7]]),
        post_ice_vtau_north=jnp.asarray([[0.8, -0.9, 1.0], [1.1, 1.2, -1.3]]),
        post_ice_taum=0.25 * one,
        post_ice_qsr=120.0 * one,
        post_ice_qns=-35.0 * one,
        post_ice_emp=1.0e-5 * one,
        post_ice_sfx=-2.0e-6 * one,
        runoff_now=jnp.asarray([[1.0e-5, 0.0, 2.0e-5], [3.0e-5, 4.0e-5, 0.0]]),
        prior_runoff_mass=jnp.zeros(shape, dtype=jnp.float64),
        prior_runoff_temperature_content=jnp.zeros(shape, dtype=jnp.float64),
        prior_runoff_salinity_content=jnp.zeros(shape, dtype=jnp.float64),
        sst_mean=jnp.asarray([[-0.0, -2.0, 3.0], [4.0, 5.0, 6.0]]),
        sss_mean=(
            jnp.asarray([[34.5, 35.2, 33.0], [36.1, 32.0, 35.0]])
            if sss_mean is None
            else sss_mean
        ),
        sss_target=jnp.asarray([[34.0, 35.0, 33.5], [35.0, 33.0, 35.0]]),
        runoff_mouth_mask=jnp.asarray([[0.0, 0.5, 0.0], [0.25, 0.0, 0.0]]),
        surface_mask=jnp.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
        tmask_east=jnp.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        tmask_north=jnp.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        umask=jnp.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        vmask=jnp.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        freshwater_budget_correction=jnp.asarray(2.0e-7),
        runoff_factor=jnp.asarray(1.0),
        reference_density=jnp.asarray(constants.rho_ocean_nemo),
        restoring_mm_per_day=jnp.asarray(-166.67),
        restoring_bound_mm_per_day=jnp.asarray(4.0),
        seconds_per_day=jnp.asarray(86400.0),
        salinity_floor=jnp.asarray(1.0e-20),
        heat_capacity=jnp.asarray(constants.c_p_seawater),
    )


def _scalar_s2(kwargs):
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in kwargs.items()}
    shape = arrays["post_ice_qns"].shape
    out = {name: np.empty(shape, dtype=np.float64) for name in (
        "rnf", "rnf_t", "rnf_s", "erp", "qrp", "qns", "emp", "utau_u", "vtau_v"
    )}
    for index in np.ndindex(shape):
        mask = np.float64(arrays["surface_mask"][index])
        rnf = np.float64(arrays["runoff_factor"] * arrays["runoff_now"][index])
        rnf = np.float64(rnf * mask)
        rnf_t = np.maximum(arrays["sst_mean"][index], np.float64(0.0))
        rnf_t = np.float64(rnf_t * rnf)
        inverse_density = np.float64(np.float64(1.0) / arrays["reference_density"])
        rnf_t = np.float64(rnf_t * inverse_density)
        rnf_s = np.float64(np.float64(0.0) * rnf)
        rnf_s = np.float64(rnf_s * inverse_density)
        zsrp = np.float64(
            arrays["restoring_mm_per_day"] / arrays["seconds_per_day"]
        )
        bound = np.float64(
            arrays["restoring_bound_mm_per_day"] / arrays["seconds_per_day"]
        )
        coefice = np.float64(
            np.float64(1.0)
            - np.float64(np.float64(2.0) * arrays["runoff_mouth_mask"][index])
        )
        delta = np.float64(arrays["sss_mean"][index] - arrays["sss_target"][index])
        numerator = np.float64(np.float64(zsrp * coefice) * delta)
        denominator = np.maximum(
            arrays["sss_mean"][index], arrays["salinity_floor"]
        )
        erp0 = np.float64(np.float64(numerator / denominator) * mask)
        magnitude = np.minimum(bound, np.abs(erp0))
        erp = np.float64(magnitude if erp0 >= np.float64(0.0) else -magnitude)
        heat = np.float64(arrays["heat_capacity"] * arrays["sst_mean"][index])
        qrp = np.float64(np.float64(-erp) * heat)
        emp = np.float64(arrays["post_ice_emp"][index] + erp)
        correction = np.float64(arrays["freshwater_budget_correction"] * mask)
        emp = np.float64(emp + correction)
        correction_heat = np.float64(
            np.float64(
                np.float64(
                    arrays["freshwater_budget_correction"]
                    * arrays["heat_capacity"]
                )
                * arrays["sst_mean"][index]
            )
            * mask
        )
        qns = np.float64(arrays["post_ice_qns"][index] + qrp)
        qns = np.float64(qns - correction_heat)
        u_sum = np.float64(
            arrays["post_ice_utau"][index] + arrays["post_ice_utau_east"][index]
        )
        u_mean = np.float64(np.float64(0.5) * u_sum)
        u_unmask = np.float64(np.float64(2.0) - arrays["umask"][index])
        u_wet = np.maximum(mask, arrays["tmask_east"][index])
        utau_u = np.float64(np.float64(u_mean * u_unmask) * u_wet)
        v_sum = np.float64(
            arrays["post_ice_vtau"][index] + arrays["post_ice_vtau_north"][index]
        )
        v_mean = np.float64(np.float64(0.5) * v_sum)
        v_unmask = np.float64(np.float64(2.0) - arrays["vmask"][index])
        v_wet = np.maximum(mask, arrays["tmask_north"][index])
        vtau_v = np.float64(np.float64(v_mean * v_unmask) * v_wet)
        for name, value in (
            ("rnf", rnf), ("rnf_t", rnf_t), ("rnf_s", rnf_s),
            ("erp", erp), ("qrp", qrp), ("qns", qns), ("emp", emp),
            ("utau_u", utau_u), ("vtau_v", vtau_v),
        ):
            out[name][index] = value
    return out


def test_fld_read_temporal_record_arithmetic_and_jit_bits():
    edges = jnp.asarray([-5, 4, 13, 22], dtype=jnp.int64)
    centres = nemo_fld_record_centres(edges, jnp.asarray(0.0))
    np.testing.assert_array_equal(np.asarray(centres), np.asarray([0, 9, 18]))
    before, after = nemo_fld_select_bracket(jnp.asarray(9), centres)
    assert (int(before), int(after)) == (1, 2)
    model_time = nemo_fld_model_time(
        nsec_year=100,
        nsec_1jan000=1000,
        forcing_stride=2,
        timestep_seconds=5.0,
        time_offset=0.0,
    )
    assert int(model_time) == 1103
    first = jnp.asarray([1.25, -3.0], dtype=jnp.float64)
    second = jnp.asarray([7.0, 5.5], dtype=jnp.float64)
    eager_call = lambda a, b: nemo_fld_time_interpolate(
        a, b, model_seconds=7, before_seconds=1, after_seconds=11
    )
    with jax.disable_jit():
        eager = eager_call(first, second)
    compiled = jax.jit(eager_call)(first, second)
    for lhs, rhs in zip(eager, compiled, strict=True):
        np.testing.assert_array_equal(_bits(lhs), _bits(rhs))


def test_sbc_ssm_nn_fsbc2_cadence_and_restart_arm():
    z = _zeros()
    prior = NemoSurfaceMeans(z, z, z, z, z, z, z)
    values = [jnp.full_like(z, float(step)) for step in range(1, 5)]
    states = []
    for kt, value in enumerate(values, start=1):
        prior = nemo_sbc_ssm_step(prior, kt=kt, **_ssm_kwargs(value))
        states.append(prior)
    np.testing.assert_array_equal(np.asarray(states[0].ssu), np.asarray(values[0]))
    np.testing.assert_array_equal(np.asarray(states[1].ssu), np.asarray(values[1]))
    np.testing.assert_array_equal(
        np.asarray(states[2].ssu), np.asarray((values[1] + values[2]) * 0.5)
    )
    np.testing.assert_array_equal(np.asarray(states[3].ssu), np.asarray(values[3]))

    def scan_step(carry, item):
        kt, value = item
        return nemo_sbc_ssm_step(carry, kt=kt, **_ssm_kwargs(value)), None

    compiled_state, _ = jax.jit(
        lambda state, samples: jax.lax.scan(
            scan_step, state, (jnp.arange(1, 5), samples)
        )
    )(NemoSurfaceMeans(z, z, z, z, z, z, z), jnp.stack(values))
    for eager_value, compiled_value in zip(states[-1], compiled_state, strict=True):
        np.testing.assert_array_equal(_bits(eager_value), _bits(compiled_value))

    restart_prior = NemoSurfaceMeans(*[jnp.full_like(z, 8.0) for _ in range(7)])
    kwargs = _ssm_kwargs(jnp.full_like(z, 2.0))
    kwargs["restart_means_present"] = True
    restarted = nemo_sbc_ssm_step(restart_prior, kt=1, **kwargs)
    np.testing.assert_array_equal(np.asarray(restarted.ssu), np.asarray(5.0 * jnp.ones_like(z)))


def test_s2_scalar_transcription_is_bit_exact_under_jit_and_truly_eager():
    kwargs = _s2_kwargs()
    expected = _scalar_s2(kwargs)
    fn = lambda *values: nemo_orca2_s2(**dict(zip(kwargs, values, strict=True)))
    values = tuple(kwargs.values())
    with jax.disable_jit():
        eager = fn(*values)
    compiled = jax.jit(fn)(*values)
    observed = {
        "rnf": compiled.runoff.mass_flux,
        "rnf_t": compiled.runoff.temperature_content,
        "rnf_s": compiled.runoff.salinity_content,
        "erp": compiled.erp,
        "qrp": compiled.qrp,
        "qns": compiled.qns,
        "emp": compiled.emp,
        "utau_u": compiled.utau_u,
        "vtau_v": compiled.vtau_v,
    }
    eager_observed = {
        "rnf": eager.runoff.mass_flux,
        "rnf_t": eager.runoff.temperature_content,
        "rnf_s": eager.runoff.salinity_content,
        "erp": eager.erp,
        "qrp": eager.qrp,
        "qns": eager.qns,
        "emp": eager.emp,
        "utau_u": eager.utau_u,
        "vtau_v": eager.vtau_v,
    }
    for name in expected:
        np.testing.assert_array_equal(_bits(observed[name]), _bits(expected[name]))
        np.testing.assert_array_equal(_bits(eager_observed[name]), _bits(expected[name]))
    assert not np.signbit(np.asarray(compiled.runoff.temperature_content)[0, 0])
    assert not np.signbit(np.asarray(compiled.erp)[1, 2])

    off_kwargs = _s2_kwargs()
    off_kwargs["kt"] = 2
    off_kwargs["prior_runoff_mass"] = jnp.full((2, 3), 0.125, dtype=jnp.float64)
    off_kwargs["prior_runoff_temperature_content"] = jnp.full(
        (2, 3), 0.25, dtype=jnp.float64
    )
    off_kwargs["prior_runoff_salinity_content"] = jnp.full(
        (2, 3), 0.5, dtype=jnp.float64
    )
    off = jax.jit(lambda: nemo_orca2_s2(**off_kwargs))()
    np.testing.assert_array_equal(np.asarray(off.runoff.mass_flux), np.full((2, 3), 0.125))
    np.testing.assert_array_equal(
        np.asarray(off.runoff.temperature_content), np.full((2, 3), 0.25)
    )
    np.testing.assert_array_equal(
        np.asarray(off.runoff.salinity_content), np.full((2, 3), 0.5)
    )
    np.testing.assert_array_equal(_bits(off.qns), _bits(off_kwargs["post_ice_qns"]))
    np.testing.assert_array_equal(_bits(off.emp), _bits(off_kwargs["post_ice_emp"]))


def test_s2_finite_difference_gradient():
    base = jnp.asarray([[34.5, 35.2, 33.0], [36.1, 32.0, 35.1]], dtype=jnp.float64)

    def objective(sss):
        result = nemo_orca2_s2(**_s2_kwargs(sss))
        return jnp.sum(result.qns) + 1.0e5 * jnp.sum(result.emp)

    direction = jnp.asarray([[0.2, -0.1, 0.0], [0.05, 0.1, -0.2]], dtype=jnp.float64)
    tangent = jnp.vdot(jax.grad(objective)(base), direction)
    epsilon = jnp.asarray(1.0e-5, dtype=jnp.float64)
    finite_difference = (
        objective(base + epsilon * direction) - objective(base - epsilon * direction)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-7, atol=2.0e-8)


def test_production_binding_and_existing_top_cell_runoff_path(monkeypatch):
    baseline = nemo_orca2_s2(**_s2_kwargs())
    original = surface_module.nemo_orca2_surface_runoff

    def planted(*args, **kwargs):
        result = original(*args, **kwargs)
        return freshwater_module.NemoRunoffSurface(
            result.mass_flux.at[0, 0].set(jnp.nextafter(result.mass_flux[0, 0], jnp.inf)),
            result.temperature_content,
            result.salinity_content,
        )

    monkeypatch.setattr(surface_module, "nemo_orca2_surface_runoff", planted)
    changed = nemo_orca2_s2(**_s2_kwargs())
    assert _bits(changed.runoff.mass_flux)[0, 0] != _bits(baseline.runoff.mass_flux)[0, 0]

    forcing, freshwater = nemo_s2_to_ocean_interfaces(changed)
    np.testing.assert_array_equal(_bits(forcing.q_solar), _bits(changed.qsr))
    np.testing.assert_array_equal(_bits(forcing.sw_down), _bits(changed.qsr))
    h = jnp.ones(changed.emp.shape + (3,), dtype=jnp.float64)
    temperature = jnp.broadcast_to(changed.qns[..., None] * 0.0 + 2.0, h.shape)
    f_rate = freshwater_module.net_freshwater_flux(freshwater) / constants.rho_ocean_nemo
    entry, _ = freshwater_module.real_freshwater_entry(
        freshwater, f_rate, h, jnp.ones_like(changed.emp), constants.rho_ocean_nemo,
        temperature, runoff_spread_m=None,
    )
    assert np.all(np.asarray(entry[..., 1:]) == 0.0)


def test_literal_physical_inputs_have_no_defaults():
    functions = (
        nemo_fld_model_time,
        nemo_fld_time_interpolate,
        nemo_sbc_ssm_step,
        nemo_orca2_s2,
    )
    for function in functions:
        for parameter in inspect.signature(function).parameters.values():
            assert parameter.default is inspect.Parameter.empty
