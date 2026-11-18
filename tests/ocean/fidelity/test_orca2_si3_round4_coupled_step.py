from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.field import Field
from legoesm.ice.si3_finalize import (
    SI3StressInputs,
    SI3StressOutputs,
    SI3UpdateFluxOutputs,
)
from legoesm.ice.si3_o17 import SI3O17LiveForcing, SI3O17SurfaceForcing
from legoesm.ocean import bulk_flux_omip as bulk
from legoesm.ocean.forcing.nemo_surface import (
    NemoFwbCase2Carry,
    NemoFwbCase2Result,
    NemoRunoffRestoringResult,
    NemoSurfaceMeans,
)
from legoesm.ocean.freshwater import FreshwaterForcing, NemoRunoffSurface
from legoesm.ocean.state import OceanSurfaceForcing


class _Aggregated(NamedTuple):
    at_i: jax.Array
    hm_i: jax.Array


class _IceState(NamedTuple):
    u_ice: jax.Array
    v_ice: jax.Array
    aggregated: _Aggregated
    snwice_mass_b: jax.Array


class _IceStep(NamedTuple):
    ice: _IceState
    aevp: jax.Array


class _AEVPForcing(NamedTuple):
    ocean_u_u: jax.Array
    ocean_v_v: jax.Array


class _Core(NamedTuple):
    state: _IceStep
    final: object
    stress_inputs: SI3StressInputs


class _Final(NamedTuple):
    outputs: SI3UpdateFluxOutputs


class _OceanStep(NamedTuple):
    state_after: object


def _field(data, name):
    return Field(jnp.asarray(data, dtype=jnp.float64), name=name)


def _fixture(monkeypatch):
    yx = (2, 3)
    xy = (3, 2)
    zero_yx = jnp.zeros(yx, dtype=jnp.float64)
    one_yx = jnp.ones(yx, dtype=jnp.float64)
    zero_xy = jnp.zeros(xy, dtype=jnp.float64)
    one_xy = jnp.ones(xy, dtype=jnp.float64)
    state = SimpleNamespace(
        T=_field(jnp.zeros(yx + (1,)), "T"),
        S=_field(jnp.ones(yx + (1,)), "S"),
        u=_field(jnp.zeros((2, 4, 1)), "u"),
        v=_field(jnp.zeros((3, 3, 1)), "v"),
        u_before=None,
        v_before=None,
        eta=_field(zero_yx, "eta"),
        H_bathy=_field(one_yx, "H"),
        land_mask=_field(one_yx, "mask"),
    )
    # SimpleNamespace is static only outside the compiled smoke.  Use the
    # production state pytree shape for the JIT/gradient check below.
    class _OceanState(NamedTuple):
        T: Field
        S: Field
        u: Field
        v: Field
        u_before: object
        v_before: object
        eta: Field
        H_bathy: Field
        land_mask: Field

    state = _OceanState(**vars(state))
    means = NemoSurfaceMeans(*(zero_yx for _ in range(7)))
    stress = SI3StressOutputs(
        zero_xy, zero_xy, zero_xy, zero_xy, zero_xy, zero_xy, -one_yx.T
    )
    fluxes = SI3UpdateFluxOutputs(*(
        zero_xy if name not in ("fr_i", "snwice_mass", "snwice_mass_b", "snwice_fmass")
        else one_xy
        for name in SI3UpdateFluxOutputs._fields
    ))
    runoff = NemoRunoffSurface(zero_yx, zero_yx, zero_yx)
    ice_state = _IceState(zero_xy, zero_xy, _Aggregated(one_xy, one_xy), one_xy)
    carry = bulk.nemo_orca2_si3_cold_start(
        ocean=state,
        ice=_IceStep(ice_state, zero_xy),
        means=means,
        stress=stress,
        fluxes=fluxes,
        runoff=runoff,
        restoring_volume_flux=zero_yx,
        restoring_heat_flux=zero_yx,
        restoring_coefficient=one_yx,
        rn_fwb0=jnp.asarray(0.0),
    )
    surface = SI3O17SurfaceForcing(*(
        zero_yx if name not in ("ssu_m", "ssv_m") else zero_xy
        for name in SI3O17SurfaceForcing._fields
    ))
    stress_inputs = SI3StressInputs(
        *(zero_xy for _ in SI3StressInputs._fields[:-1]), tmask_k1=one_xy
    )
    o17 = SI3O17LiveForcing(
        1, zero_xy, zero_xy, zero_xy, zero_xy,
        _AEVPForcing(zero_xy, zero_xy), zero_xy, zero_xy, surface, stress_inputs,
    )
    atmosphere = bulk.NemoStandaloneAtmosphereFields(*(zero_yx for _ in range(9)))
    pair = bulk.NemoStandaloneFieldPair(zero_yx, zero_yx, 1, 0, 2)
    forcing = bulk.NemoORCA2SI3Forcing(
        bulk.NemoStandaloneS0FInputs(atmosphere, atmosphere, 1, 0, 2, pair, pair),
        o17,
        zero_yx,
        zero_yx,
    )
    ice_cfg = SimpleNamespace(nn_hls=0, nn_fsbc=2)
    ice_card = SimpleNamespace(
        state=ice_cfg,
        fold=SimpleNamespace(nfd_full_when_absent=True),
        ln_drgice_imp=True,
    )
    zcoord = SimpleNamespace(nemo_e3t_0=jnp.ones(yx + (1,)))
    ocean_card = SimpleNamespace(recipe=SimpleNamespace(z_coord=zcoord))
    index = jnp.arange(6, dtype=jnp.int64).reshape(yx)
    lbc = bulk.NemoTStressLbcPlan(
        index, index, index, one_yx, one_yx, one_yx
    )
    card = bulk.NemoORCA2SI3Card(
        ocean_card, ice_card, lbc, one_yx, one_yx, one_yx, one_yx,
        one_yx, one_yx, one_yx, one_yx, jnp.asarray(1.0), jnp.asarray(1.0),
        jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(1.0),
        jnp.asarray(0.0), jnp.asarray(1.0), 4, 2, jnp.asarray(0.0),
        jnp.asarray(1.0), jnp.asarray(1.0), 1,
    )

    calls = []

    def mark(name):
        calls.append(name)

    monkeypatch.setattr("legoesm.ocean.eos.nemo_r3t_stretch", lambda *_a, **_k: one_yx)

    def ssm(prior, **_kwargs):
        mark("ssm")
        return prior._replace(sst=prior.sst + 1.0)

    monkeypatch.setattr(bulk, "nemo_sbc_ssm_step", ssm)
    monkeypatch.setattr(
        bulk, "nemo_orca2_k0_s0f", lambda _x: (atmosphere, zero_yx, zero_yx)
    )

    def open_water(*_args, **_kwargs):
        mark("bulk")
        return bulk.NemoBlkOceResult(*(
            one_yx if name in ("rhoa", "qsr", "qns", "emp", "utau", "vtau", "taum")
            else zero_yx
            for name in bulk.NemoBlkOceResult._fields
        ))

    monkeypatch.setattr(bulk, "nemo_blk_oce_ncar", open_water)

    def core(entry, live, _card):
        mark("ice")
        out = fluxes._replace(qsr=fluxes.qsr + 1.0)
        next_ice = entry._replace(
            ice=entry.ice._replace(
                aggregated=entry.ice.aggregated._replace(hm_i=entry.ice.aggregated.hm_i + 1.0)
            )
        )
        return _Core(next_ice, _Final(out), live.stress)

    monkeypatch.setattr("legoesm.ice.si3_o17.si3_o17_through_update_flx", core)

    def tau(inputs, **_kwargs):
        mark("tau")
        return SI3StressOutputs(
            inputs.utau + 1.0, inputs.vtau + 1.0, inputs.taum + 1.0,
            inputs.tmod_io + 1.0, inputs.utau_oce + 1.0,
            inputs.vtau_oce + 1.0, -one_yx.T,
        )

    monkeypatch.setattr("legoesm.ice.si3_finalize.si3_ice_update_tau", tau)

    def pre_fwb(**kwargs):
        mark("runoff_restoring")
        return NemoRunoffRestoringResult(
            runoff, one_yx, zero_yx, zero_yx, kwargs["post_ice_qns"],
            kwargs["post_ice_emp"], kwargs["post_ice_qns"], kwargs["post_ice_emp"],
        )

    monkeypatch.setattr(bulk, "nemo_orca2_runoff_restoring", pre_fwb)

    def fwb(old, *, kt, emp, qns, **_kwargs):
        mark("fwb")
        delta = jnp.asarray(kt, dtype=jnp.float64)
        return NemoFwbCase2Result(
            NemoFwbCase2Carry(old.a_fwb + delta, old.a_fwb_b, old.emp_corr),
            emp + delta,
            qns + delta,
        )

    monkeypatch.setattr(bulk, "nemo_fwb_case2_step", fwb)

    def lbc(u, v, _plan):
        mark("lbc")
        return u + 1.0, v + 1.0, u + 1.0, v + 1.0

    monkeypatch.setattr(bulk, "nemo_t_stress_lbc_neighbors", lbc)
    monkeypatch.setattr(bulk, "nemo_t_stress_to_uv", lambda u, v, *_a: (u, v))

    def interfaces(tail, **_kwargs):
        mark("interfaces")
        return OceanSurfaceForcing(
            taum=tail.taum,
            tau_i_native=tail.utau,
            tau_j_native=tail.vtau,
            ice_concentration=one_yx,
        ), FreshwaterForcing(zero_yx, tail.emp, tail.runoff.mass_flux, zero_yx, zero_yx)

    monkeypatch.setattr(bulk, "nemo_s2_to_ocean_interfaces", interfaces)

    def ocean_step(_card, ocean, *, surface_forcing, **_kwargs):
        mark("ocean")
        drag = surface_forcing.ice_ocean_top_drag_rCdu
        drag_term = zero_yx if drag is None else drag
        eta = (
            ocean.eta.data
            + surface_forcing.ice_mean_thickness_hm_i
            + drag_term
            + surface_forcing.tau_i_native
        )
        return _OceanStep(ocean._replace(eta=ocean.eta.replace(data=eta)))

    monkeypatch.setattr(bulk, "nemo_orca2_ocean_consumption_step", ocean_step)
    return carry, forcing, card, calls


def _bits(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return b"".join(np.asarray(leaf).tobytes() for leaf in leaves)


def test_carry_pytree_round_trip(monkeypatch):
    carry, _, _, _ = _fixture(monkeypatch)
    leaves, definition = jax.tree_util.tree_flatten(carry)
    rebuilt = jax.tree_util.tree_unflatten(definition, leaves)
    assert rebuilt.kt == 1
    assert _bits(rebuilt) == _bits(carry)


def test_source_order_and_exactly_once_routes(monkeypatch):
    carry, forcing, card, calls = _fixture(monkeypatch)
    carry, _ = bulk.nemo_orca2_si3_coupled_step(carry, forcing, card)
    carry, _ = bulk.nemo_orca2_si3_coupled_step(carry, forcing, card)
    assert calls == [
        "ssm", "bulk", "ice", "tau", "runoff_restoring", "fwb", "lbc",
        "interfaces", "ocean", "ssm", "bulk", "tau", "runoff_restoring",
        "fwb", "lbc", "interfaces", "ocean",
    ]


@pytest.mark.parametrize(
    "plant",
    (
        "_plant_skip_even_tau",
        "_plant_run_si3_even",
        "_plant_drop_hm_i",
        "_plant_drop_rcdu_ice",
        "_plant_fwb_even",
        "_plant_double_stress_lbc",
    ),
)
def test_each_c2_route_plant_changes_the_pair(monkeypatch, plant):
    carry, forcing, card, _ = _fixture(monkeypatch)
    baseline, _ = bulk.nemo_orca2_si3_coupled_step(carry, forcing, card)
    baseline, _ = bulk.nemo_orca2_si3_coupled_step(baseline, forcing, card)
    planted, _ = bulk.nemo_orca2_si3_coupled_step(
        carry, forcing, card, **({plant: True} if plant not in (
            "_plant_skip_even_tau", "_plant_run_si3_even", "_plant_fwb_even"
        ) else {})
    )
    planted, _ = bulk.nemo_orca2_si3_coupled_step(
        planted, forcing, card, **{plant: True}
    )
    assert _bits(planted) != _bits(baseline)


def test_compiled_pair_has_finite_reverse_mode_gradient(monkeypatch):
    carry, forcing, card, _ = _fixture(monkeypatch)

    def loss(x):
        seeded = carry._replace(
            ocean=carry.ocean._replace(
                eta=carry.ocean.eta.replace(data=carry.ocean.eta.data + x)
            )
        )
        even, _ = bulk.nemo_orca2_si3_coupled_pair(seeded, forcing, card)
        return jnp.sum(even.ocean.eta.data)

    compiled = jax.jit(jax.value_and_grad(loss))
    value, gradient = compiled(jnp.asarray(0.25, dtype=jnp.float64))
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))
    assert float(gradient) != 0.0


def _c2_observations():
    from scripts.validate.ocean_fidelity.testcases.nemo_coupled_c_ladder_acquisition import (
        scorer,
    )

    values = {}
    for kind in (*scorer.C2_REPORT_ONLY_KINDS, *scorer.C2_VERDICT_KINDS):
        for kt in (1, 2):
            if kind in (15, 16) and kt == 2:
                continue
            for rank in (0, 1):
                values[(kind, kt, rank)] = {
                    "field": np.zeros((6, 7), dtype=np.float64)
                }
    masks = {
        (kt, rank): {
            "tmask": np.pad(np.ones((2, 3)), 2),
            "umask": np.pad(np.ones((2, 3)), 2),
            "vmask": np.pad(np.ones((2, 3)), 2),
        }
        for kt in (1, 2)
        for rank in (0, 1)
    }
    return scorer, values, masks


def test_c2_d225_allows_only_dry_signed_zero_and_reports_kinds_1_2():
    scorer, expected, masks = _c2_observations()
    candidate = {
        key: {name: value.copy() for name, value in fields.items()}
        for key, fields in expected.items()
    }
    masks[(1, 0)]["tmask"][2, 2] = 0.0
    candidate[(3, 1, 0)]["field"][2, 2] = -0.0
    candidate[(1, 1, 0)]["field"][2, 3] = np.nextafter(0.0, 1.0)
    report = scorer.score_c2_observations(candidate, expected, masks)
    assert report["status"] == "PASS_C2_0_ULP_WET"
    assert report["counts"][3]["dry_zero_sign_different"] == 1
    assert report["report_only_counts"][1]["wet_different"] == 1


def test_c2_d225_wet_plant_names_first_kind_field_cell_and_step():
    scorer, expected, masks = _c2_observations()
    candidate = {
        key: {name: value.copy() for name, value in fields.items()}
        for key, fields in expected.items()
    }
    candidate[(4, 2, 1)]["field"][2, 3] = np.nextafter(0.0, 1.0)
    report = scorer.score_c2_observations(candidate, expected, masks)
    assert report["status"].startswith(
        "FAIL_C2 kind=4 field=field cell=(2, 3) kt=2 rank=1"
    )
