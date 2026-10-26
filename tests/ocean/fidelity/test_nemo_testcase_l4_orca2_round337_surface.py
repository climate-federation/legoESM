from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean import freshwater as freshwater_module
from legoesm.ocean import bulk_flux_omip as bulk_module
from legoesm.ocean.forcing import sss_restoring as restoring_module
from legoesm.ocean.forcing.nemo_fld_read import (
    nemo_fld_model_time,
    nemo_fld_record_centres,
    nemo_fld_select_bracket,
    nemo_fld_time_interpolate,
)
from legoesm.ocean.forcing.nemo_surface import (
    NemoSurfaceMeans,
    NemoTStressLbcPlan,
    nemo_orca2_s2,
    nemo_sbc_ssm_step,
    nemo_t_stress_lbc_neighbors,
    nemo_t_stress_to_uv,
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
        prior_restoring_volume_flux=jnp.zeros(shape, dtype=jnp.float64),
        prior_restoring_heat_flux=jnp.zeros(shape, dtype=jnp.float64),
        prior_restoring_coefficient=jnp.ones(shape, dtype=jnp.float64),
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
    """Independent scalar execution of sbcrnf/sbcssr/sbcfwb/sbcmod statements."""
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in kwargs.items()}
    shape = arrays["post_ice_qns"].shape
    out = {name: np.empty(shape, dtype=np.float64) for name in (
        "rnf", "rnf_t", "rnf_s", "coefice", "erp", "qrp", "qns", "emp",
        "utau_u", "vtau_v"
    )}
    update = (int(arrays["kt"]) - 1) % int(arrays["forcing_stride"]) == 0
    for index in np.ndindex(shape):
        mask = np.float64(arrays["surface_mask"][index])
        if update:
            # sbcrnf.f90: the runoff fields are assigned only on an sbc step.
            rnf = np.float64(arrays["runoff_factor"] * arrays["runoff_now"][index])
            rnf = np.float64(rnf * mask)
            sst_value = np.float64(arrays["sst_mean"][index])
            rnf_t = sst_value if sst_value > np.float64(0.0) else np.float64(0.0)
            rnf_t = np.float64(rnf_t * rnf)
            inverse_density = np.float64(np.float64(1.0) / arrays["reference_density"])
            rnf_t = np.float64(rnf_t * inverse_density)
            rnf_s = np.float64(np.float64(0.0) * rnf)
            rnf_s = np.float64(rnf_s * inverse_density)

            # sbcssr.f90:130-139, in written statement order.
            two_rnf = np.float64(np.float64(2.0) * arrays["runoff_mouth_mask"][index])
            damping = np.float64(np.float64(1.0) - two_rnf)
            coefice = np.float64(arrays["prior_restoring_coefficient"][index] * damping)
            zsrp = np.float64(arrays["restoring_mm_per_day"] / arrays["seconds_per_day"])
            bound = np.float64(arrays["restoring_bound_mm_per_day"] / arrays["seconds_per_day"])
            delta = np.float64(arrays["sss_mean"][index] - arrays["sss_target"][index])
            numerator = np.float64(np.float64(zsrp * coefice) * delta)
            first = np.float64(arrays["sss_mean"][index])
            second = np.float64(arrays["salinity_floor"])
            denominator = first if first > second else second
            erp0 = np.float64(np.float64(numerator / denominator) * mask)
            absolute = np.float64(abs(erp0))
            magnitude = bound if bound < absolute else absolute
            erp = np.float64(magnitude if erp0 >= np.float64(0.0) else -magnitude)
            heat = np.float64(np.float64(erp * arrays["heat_capacity"]) * arrays["sst_mean"][index])
            emp = np.float64(arrays["post_ice_emp"][index] + erp)
            qns = np.float64(arrays["post_ice_qns"][index] - heat)
            qrp = np.float64(np.float64(0.0) - heat)

            # sbcfwb.f90: forcing-cadence correction, after restoring.
            correction = np.float64(arrays["freshwater_budget_correction"] * mask)
            emp = np.float64(emp + correction)
            correction_heat = np.float64(
                np.float64(
                    np.float64(arrays["freshwater_budget_correction"] * arrays["heat_capacity"])
                    * arrays["sst_mean"][index]
                ) * mask
            )
            qns = np.float64(qns - correction_heat)
        else:
            rnf = np.float64(arrays["prior_runoff_mass"][index])
            rnf_t = np.float64(arrays["prior_runoff_temperature_content"][index])
            rnf_s = np.float64(arrays["prior_runoff_salinity_content"][index])
            coefice = np.float64(arrays["prior_restoring_coefficient"][index])
            erp = np.float64(arrays["prior_restoring_volume_flux"][index])
            qrp = np.float64(arrays["prior_restoring_heat_flux"][index])
            emp = np.float64(arrays["post_ice_emp"][index])
            qns = np.float64(arrays["post_ice_qns"][index])
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
            ("rnf", rnf), ("rnf_t", rnf_t), ("rnf_s", rnf_s), ("coefice", coefice),
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
    assert (int(before), int(after)) == (0, 1)
    before, after = nemo_fld_select_bracket(jnp.asarray(10), centres)
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
    with jax.disable_jit():
        for kt, value in enumerate(values, start=1):
            kwargs = _ssm_kwargs(value)
            kwargs["remove_inverse_barometer"] = True
            kwargs["ssh_inverse_barometer"] = jnp.full_like(value, 0.25)
            prior = nemo_sbc_ssm_step(prior, kt=kt, **kwargs)
            states.append(prior)
    np.testing.assert_array_equal(np.asarray(states[0].ssu), np.asarray(values[0]))
    np.testing.assert_array_equal(np.asarray(states[1].ssu), np.asarray(values[1]))
    np.testing.assert_array_equal(
        np.asarray(states[2].ssu), np.asarray((values[1] + values[2]) * 0.5)
    )
    np.testing.assert_array_equal(np.asarray(states[3].ssu), np.asarray(values[3]))

    def scan_step(carry, item):
        kt, value = item
        kwargs = _ssm_kwargs(value)
        kwargs["remove_inverse_barometer"] = True
        kwargs["ssh_inverse_barometer"] = jnp.full_like(value, 0.25)
        return nemo_sbc_ssm_step(carry, kt=kt, **kwargs), None

    with jax.disable_jit():
        eager_state, _ = jax.lax.scan(
            scan_step,
            NemoSurfaceMeans(z, z, z, z, z, z, z),
            (jnp.arange(1, 5), jnp.stack(values)),
        )
    compiled_state, _ = jax.jit(
        lambda state, samples: jax.lax.scan(
            scan_step, state, (jnp.arange(1, 5), samples)
        )
    )(NemoSurfaceMeans(z, z, z, z, z, z, z), jnp.stack(values))
    for eager_value, compiled_value in zip(eager_state, compiled_state, strict=True):
        np.testing.assert_array_equal(_bits(eager_value), _bits(compiled_value))

    restart_prior = NemoSurfaceMeans(*[jnp.full_like(z, 8.0) for _ in range(7)])
    kwargs = _ssm_kwargs(jnp.full_like(z, 2.0))
    kwargs["restart_means_present"] = True
    restarted = nemo_sbc_ssm_step(restart_prior, kt=1, **kwargs)
    np.testing.assert_array_equal(np.asarray(restarted.ssu), np.asarray(5.0 * jnp.ones_like(z)))

    # sbcssm.f90:152 is (ssh_m + ssh) - ssh_ib, not ssh_m + (ssh - ssh_ib).
    association = _ssm_kwargs(jnp.zeros_like(z))
    association.update(
        forcing_stride=3,
        restart_means_present=True,
        remove_inverse_barometer=True,
        ssh_now=jnp.full_like(z, -1.0e16),
        ssh_inverse_barometer=jnp.ones_like(z),
    )
    association_prior = restart_prior._replace(ssh=jnp.full_like(z, 1.0e16))
    with jax.disable_jit():
        grouped = nemo_sbc_ssm_step(association_prior, kt=1, **association)
    np.testing.assert_array_equal(
        _bits(grouped.ssh), _bits(jnp.full_like(z, -1.0 / 3.0))
    )


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
        "coefice": compiled.restoring_coefficient,
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
        "coefice": eager.restoring_coefficient,
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
    off_kwargs["prior_restoring_volume_flux"] = jnp.full(
        (2, 3), 0.75, dtype=jnp.float64
    )
    off_kwargs["prior_restoring_heat_flux"] = jnp.full(
        (2, 3), 1.25, dtype=jnp.float64
    )
    off_kwargs["prior_restoring_coefficient"] = jnp.full(
        (2, 3), 0.625, dtype=jnp.float64
    )
    off_kwargs["post_ice_qns"] = jnp.full((2, 3), -0.0, dtype=jnp.float64)
    off_kwargs["post_ice_emp"] = jnp.full((2, 3), -0.0, dtype=jnp.float64)
    off = jax.jit(lambda: nemo_orca2_s2(**off_kwargs))()
    np.testing.assert_array_equal(np.asarray(off.runoff.mass_flux), np.full((2, 3), 0.125))
    np.testing.assert_array_equal(
        np.asarray(off.runoff.temperature_content), np.full((2, 3), 0.25)
    )
    np.testing.assert_array_equal(
        np.asarray(off.runoff.salinity_content), np.full((2, 3), 0.5)
    )
    np.testing.assert_array_equal(np.asarray(off.erp), np.full((2, 3), 0.75))
    np.testing.assert_array_equal(np.asarray(off.qrp), np.full((2, 3), 1.25))
    np.testing.assert_array_equal(
        np.asarray(off.restoring_coefficient), np.full((2, 3), 0.625)
    )
    np.testing.assert_array_equal(_bits(off.qns), _bits(off_kwargs["post_ice_qns"]))
    np.testing.assert_array_equal(_bits(off.emp), _bits(off_kwargs["post_ice_emp"]))


def test_sbcssr_object_ties_and_coefficient_state_are_bit_pinned(monkeypatch):
    shape = (2, 3)
    max_tie = _s2_kwargs(jnp.zeros(shape, dtype=jnp.float64))
    max_tie["sss_target"] = jnp.ones(shape, dtype=jnp.float64)
    max_tie["salinity_floor"] = jnp.asarray(-0.0, dtype=jnp.float64)
    max_tie["runoff_mouth_mask"] = jnp.zeros(shape, dtype=jnp.float64)
    max_tie["surface_mask"] = jnp.ones(shape, dtype=jnp.float64)
    literal_max = nemo_orca2_s2(**max_tie)
    assert np.all(np.asarray(literal_max.erp) < 0.0)
    real_max = restoring_module._fortran_max
    monkeypatch.setattr(restoring_module, "_fortran_max", jnp.maximum)
    planted_max = nemo_orca2_s2(**max_tie)
    assert np.all(np.asarray(planted_max.erp) > 0.0)
    monkeypatch.setattr(restoring_module, "_fortran_max", real_max)

    # The three MINSD sites differ in tie operand, but a finite tie cannot
    # change bits: bound and ABS(zerp) are both nonnegative and equal.  Check
    # every peeled/unrolled site for even and odd local-i extents at the real
    # positive bound; this replaces the old unreachable negative-zero bound.
    for width in (3, 4):
        bound = np.float64(4.0 / 86400.0)
        absolute = jnp.full((2, width), bound, dtype=jnp.float64)
        expected_bound_sites = (
            np.asarray([True, False, True])
            if width == 3
            else np.asarray([False, True, False, True])
        )
        np.testing.assert_array_equal(
            np.asarray(restoring_module._sbcssr_bound_wins_tie(width)),
            expected_bound_sites,
        )
        tied = restoring_module._sbcssr_bound_magnitude(
            jnp.asarray(bound), absolute
        )
        np.testing.assert_array_equal(_bits(tied), _bits(absolute))
        np.testing.assert_array_equal(
            _bits(absolute), _bits(jnp.full_like(absolute, bound))
        )

    first_kwargs = _s2_kwargs()
    first = nemo_orca2_s2(**first_kwargs)
    second_kwargs = _s2_kwargs()
    second_kwargs.update(
        kt=3,
        prior_restoring_coefficient=first.restoring_coefficient,
        prior_restoring_volume_flux=first.erp,
        prior_restoring_heat_flux=first.qrp,
    )
    second = nemo_orca2_s2(**second_kwargs)
    damping = 1.0 - 2.0 * np.asarray(first_kwargs["runoff_mouth_mask"])
    expected = np.asarray(first.restoring_coefficient) * damping
    np.testing.assert_array_equal(_bits(second.restoring_coefficient), _bits(expected))


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


def _production_surface_case():
    from legoesm.ocean.bulk_flux_omip import (
        NemoS0Inputs,
        NemoS2Inputs,
        NemoSbcIceIfInputs,
    )

    shape = (2, 3)
    full_shape = (4, 5)
    zero = jnp.zeros(full_shape, dtype=jnp.float64)
    one = jnp.ones(shape, dtype=jnp.float64)
    value = jnp.arange(20, dtype=jnp.float64).reshape(full_shape) / 10.0
    s0 = NemoS0Inputs(
        prior=NemoSurfaceMeans(zero, zero, zero, zero, zero, zero, zero),
        kt=1,
        **_ssm_kwargs(value),
    )
    s2_kwargs = _s2_kwargs()
    center = jnp.arange(6, dtype=jnp.int64).reshape(shape)
    stress_lbc = NemoTStressLbcPlan(
        center_index=center,
        east_index=jnp.asarray([[1, 2, 0], [4, 5, 3]], dtype=jnp.int64),
        north_index=jnp.asarray([[3, 4, 5], [5, 4, 3]], dtype=jnp.int64),
        center_sign=one,
        east_sign=one,
        north_sign=jnp.asarray([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
    )
    s2_values = {
        name: (stress_lbc if name == "stress_lbc" else s2_kwargs[name])
        for name in NemoS2Inputs._fields
    }
    s2 = NemoS2Inputs(**s2_values)
    ice_if = NemoSbcIceIfInputs(
        fdta_before=jnp.full(shape, 0.25, dtype=jnp.float64),
        fdta_after=jnp.full(shape, 0.75, dtype=jnp.float64),
        isecsbc=jnp.asarray(0, dtype=jnp.int64),
        nrec_sec_before=jnp.asarray(-10, dtype=jnp.int64),
        nrec_sec_after=jnp.asarray(10, dtype=jnp.int64),
        # Deliberately stale: the composed path must replace these with S0.
        sss_m=jnp.full(full_shape, -999.0, dtype=jnp.float64),
        sst_m=jnp.full(full_shape, 999.0, dtype=jnp.float64),
        ts_surface=jnp.full(full_shape, -2.0, dtype=jnp.float64),
        ff_f=jnp.ones(full_shape, dtype=jnp.float64),
        tmask_surface=jnp.ones(full_shape, dtype=jnp.float64),
        smask0=one,
    )
    bulk_args = (
        6.0 * one,
        -2.0 * one,
        286.0 * one,
        0.006 * one,
        100800.0 * one,
        None,
        None,
        None,
        125.0 * one,
        310.0 * one,
        2.3e-5 * one,
        3.0e-6 * one,
        s2.surface_mask,
        0.066,
        1.0,
        1.0,
        5,
    )
    return bulk_args, s0, ice_if, s2


def test_production_binding_and_existing_top_cell_runoff_path(monkeypatch):
    from legoesm.ocean.bulk_flux_omip import nemo_blk_oce_ncar

    bulk_args, s0, ice_if, s2 = _production_surface_case()
    with pytest.raises(ValueError, match="S0 and S2 together"):
        nemo_blk_oce_ncar(
            *bulk_args, nn_ice=1, ice_if=ice_if, s0=s0, s2=None)
    with pytest.raises(ValueError, match="requires composed nn_ice=1 S3"):
        nemo_blk_oce_ncar(
            *bulk_args, nn_ice=0, ice_if=None, s0=s0, s2=s2)
    shadowed = list(bulk_args)
    shadowed[5:8] = [s0.temperature_now, s0.u_before, s0.v_before]
    with pytest.raises(ValueError, match="produces sst_c"):
        nemo_blk_oce_ncar(
            *shadowed, nn_ice=1, ice_if=ice_if, s0=s0, s2=s2)
    run = lambda: nemo_blk_oce_ncar(
        *bulk_args, nn_ice=1, ice_if=ice_if, s0=s0, s2=s2)
    with jax.disable_jit():
        baseline = run()
    compiled = jax.jit(run)()
    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(baseline),
        jax.tree_util.tree_leaves(compiled),
        strict=True,
    ):
        np.testing.assert_array_equal(_bits(eager_leaf), _bits(compiled_leaf))

    # S3 consumes the same-step S0 means, not the deliberately stale means in
    # the caller's ice bundle, and sits between the bulk and S2 calls.
    interior = (slice(1, -1), slice(1, -1))
    s1_args = list(bulk_args)
    s1_args[5:8] = [
        baseline.means.sst[interior],
        baseline.means.ssu[interior],
        baseline.means.ssv[interior],
    ]
    raw_bulk = nemo_blk_oce_ncar(
        *s1_args, nn_ice=0, ice_if=None, s0=None, s2=None)
    cover = bulk_module.nemo_fld_time_interp(*ice_if[:5])
    expected_s3 = bulk_module.nemo_sbc_ice_if(
        cover,
        baseline.means.sss,
        baseline.means.sst,
        ice_if.ts_surface,
        ice_if.ff_f,
        ice_if.tmask_surface,
        ice_if.smask0,
        raw_bulk.qsr,
        raw_bulk.qns,
        nn_hls=1,
    )
    np.testing.assert_array_equal(_bits(baseline.bulk.qsr), _bits(expected_s3.qsr))
    np.testing.assert_array_equal(_bits(baseline.bulk.qns), _bits(expected_s3.qns))

    # The native-face stresses must use the just-computed stress after its
    # NEMO T-vector LBC, never caller-supplied east/north stress values.
    lbc = nemo_t_stress_lbc_neighbors(
        baseline.bulk.utau, baseline.bulk.vtau, s2.stress_lbc
    )
    expected_u, expected_v = nemo_t_stress_to_uv(
        *lbc,
        s2.surface_mask,
        s2.tmask_east,
        s2.tmask_north,
        s2.umask,
        s2.vmask,
    )
    np.testing.assert_array_equal(_bits(baseline.s2.utau_u), _bits(expected_u))
    np.testing.assert_array_equal(_bits(baseline.s2.vtau_v), _bits(expected_v))
    caller_u, caller_v = nemo_t_stress_to_uv(
        lbc[0],
        lbc[1],
        jnp.full_like(lbc[2], 1234.0),
        jnp.full_like(lbc[3], -5678.0),
        s2.surface_mask,
        s2.tmask_east,
        s2.tmask_north,
        s2.umask,
        s2.vmask,
    )
    assert np.any(_bits(baseline.s2.utau_u) != _bits(caller_u))
    assert np.any(_bits(baseline.s2.vtau_v) != _bits(caller_v))

    original_s3 = bulk_module.nemo_sbc_ice_if

    def planted_s3(*args, **kwargs):
        result = original_s3(*args, **kwargs)
        return result._replace(qns=result.qns.at[0, 0].add(1.0))

    monkeypatch.setattr(bulk_module, "nemo_sbc_ice_if", planted_s3)
    with jax.disable_jit():
        changed_s3 = run()
    assert _bits(changed_s3.bulk.qns)[0, 0] != _bits(baseline.bulk.qns)[0, 0]
    assert _bits(changed_s3.s2.qns)[0, 0] != _bits(baseline.s2.qns)[0, 0]
    monkeypatch.setattr(bulk_module, "nemo_sbc_ice_if", original_s3)

    original_ssm = bulk_module.nemo_sbc_ssm_step

    def planted_ssm(*args, **kwargs):
        result = original_ssm(*args, **kwargs)
        return result._replace(
            sst=result.sst.at[0, 0].set(jnp.nextafter(result.sst[0, 0], jnp.inf))
        )

    monkeypatch.setattr(bulk_module, "nemo_sbc_ssm_step", planted_ssm)
    with jax.disable_jit():
        changed_s0 = run()
    assert _bits(changed_s0.means.sst)[0, 0] != _bits(baseline.means.sst)[0, 0]
    monkeypatch.setattr(bulk_module, "nemo_sbc_ssm_step", original_ssm)

    original_s2 = bulk_module.nemo_orca2_s2

    def planted_s2(*args, **kwargs):
        result = original_s2(*args, **kwargs)
        return result._replace(
            qns=result.qns.at[0, 0].add(jnp.asarray(1.0, dtype=jnp.float64))
        )

    monkeypatch.setattr(bulk_module, "nemo_orca2_s2", planted_s2)
    with jax.disable_jit():
        changed_s2 = run()
    assert _bits(changed_s2.s2.qns)[0, 0] != _bits(baseline.s2.qns)[0, 0]
    assert _bits(changed_s2.surface_forcing.q_net)[0, 0] != _bits(
        baseline.surface_forcing.q_net
    )[0, 0]

    forcing, freshwater = baseline.surface_forcing, baseline.freshwater_forcing
    np.testing.assert_array_equal(_bits(forcing.q_solar), _bits(baseline.s2.qsr))
    np.testing.assert_array_equal(_bits(forcing.sw_down), _bits(baseline.s2.qsr))
    h = jnp.ones(baseline.s2.emp.shape + (3,), dtype=jnp.float64)
    temperature = jnp.broadcast_to(baseline.s2.qns[..., None] * 0.0 + 2.0, h.shape)
    f_rate = freshwater_module.net_freshwater_flux(freshwater) / constants.rho_ocean_nemo
    entry, _ = freshwater_module.real_freshwater_entry(
        freshwater, f_rate, h, jnp.ones_like(baseline.s2.emp), constants.rho_ocean_nemo,
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
