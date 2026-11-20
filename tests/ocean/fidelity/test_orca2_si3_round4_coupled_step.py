from __future__ import annotations

from pathlib import Path
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
        ln_teos10=False,
    )
    zcoord = SimpleNamespace(
        nemo_e3t_0=jnp.ones(yx + (1,)), linear_free_surface=False
    )
    physics = SimpleNamespace(
        vertical_mixing=SimpleNamespace(tke=SimpleNamespace(n2_eos_form="eos80")),
        convection=SimpleNamespace(scheme="none"),
        lateral_mixing=SimpleNamespace(scheme="none"),
        shortwave_penetration=SimpleNamespace(),
    )
    ocean_card = SimpleNamespace(
        recipe=SimpleNamespace(
            z_coord=zcoord,
            model_config=SimpleNamespace(eos="nemo_eos80", physics=physics),
        )
    )
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
        kt=1,
        card=card,
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
        si3_qns_parts = {"qlwn": 2.0, "sensible": 3.0, "latent": 4.0, "qns": 99.0}
        return bulk.NemoBlkOceResult(*(
            jnp.full_like(one_yx, si3_qns_parts[name]) if name in si3_qns_parts
            else one_yx if name in ("rhoa", "qsr", "emp", "utau", "vtau", "taum")
            else zero_yx
            for name in bulk.NemoBlkOceResult._fields
        ))

    monkeypatch.setattr(bulk, "nemo_blk_oce_ncar", open_water)

    def core(entry, live, _card):
        mark("ice")
        assert np.array_equal(np.asarray(live.surface.qns_oce), np.full(xy, 9.0))
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
    monkeypatch.setattr(
        bulk, "_nemo_orca2_k0_next_fraqsr_compiled", lambda *_a, **_k: one_yx
    )

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


def test_coupled_card_rejects_nameos_disagreement_at_build_boundary(monkeypatch):
    _, _, card, _ = _fixture(monkeypatch)
    bad_ice = SimpleNamespace(**{**vars(card.ice), "ln_teos10": True})
    mismatched = card._replace(ice=bad_ice)
    with pytest.raises(ValueError, match="ocean/ice EOS settings disagree"):
        bulk._validate_nemo_orca2_si3_eos(mismatched)


def test_source_order_and_exactly_once_routes(monkeypatch):
    carry, forcing, card, calls = _fixture(monkeypatch)
    carry, _ = bulk.nemo_orca2_si3_coupled_step(
        carry, forcing, card, model=object(), sponge=None
    )
    carry, _ = bulk.nemo_orca2_si3_coupled_step(
        carry, forcing, card, model=object(), sponge=None
    )
    assert calls == [
        "ssm", "bulk", "ice", "tau", "runoff_restoring", "fwb", "lbc",
        "interfaces", "ocean", "ssm", "tau", "runoff_restoring",
        "fwb", "lbc", "interfaces", "ocean",
    ]


@pytest.fixture(scope="module")
def real_coupled_fixture():
    """The production O17 12x10 physics coupled to a real 6x8 ocean."""
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ice.si3_finalize import si3_ice_update_tau
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.shortwave_penetration import ShortwavePenetrationConfig
    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig, VerticalMixingConfig
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate
    from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_ad_probe import (
        DEFAULT_MASTER,
        _case,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    entry, o17, ice_card = _case(DEFAULT_MASTER, shape=(12, 10))
    ice_dt = 120.0
    ice_card = ice_card._replace(
        state=ice_card.state._replace(nn_fsbc=2, rDt_ice=ice_dt),
        aevp=ice_card.aevp._replace(dt_s=ice_dt),
        redistribution=ice_card.redistribution._replace(rdt_ice=ice_dt),
        correction=ice_card.correction._replace(rdt_ice=ice_dt),
        surface=ice_card.surface._replace(rDt_ice=ice_dt),
        ln_drgice_imp=True,
        ln_teos10=False,
    )

    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z_ref = create_ocean_z_star(n_levels=5, H_max=600.0)
    bathymetry = jnp.full((6, 8), 372.0, dtype=jnp.float64)
    z_coord = create_partial_cell_coordinate(z_ref, bathymetry)
    z_coord = z_coord._replace(
        nemo_e3t_0=jnp.broadcast_to(z_ref.dz_ref, (6, 8, 5))
    )
    state = rest_state_latlon_cgrid_ocean(
        grid, z_ref, T_water_init_C=2.0, T_deep=1.0, S_uniform=34.0,
        H_bathy_override=bathymetry, nemo_prognostic_barotropic_velocity=True,
    )
    state = state._replace(
        u=state.u.replace(data=state.u.data + 0.01 * state.u_mask.data[..., None]),
        v=state.v.replace(data=state.v.data - 0.007 * state.v_mask.data[..., None]),
        tke=Field(
            jnp.full(state.T.data.shape[:-1] + (4,), 0.01),
            name="tke", dims=("lat", "lon", "level"), units="m^2/s^2",
        ),
    )
    tke = TKEConfig(
        prognostic=True, tke_mxl_choice=3, nemo_mxlice=2,
        nemo_mxl0_surface_tmask=True,
        tke_preclosure_coeff_source="carried_previous_step",
        n2_eos_form="eos80",
    )
    base = OceanPhysicsConfig()
    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="tke", tke=tke),
        lateral_mixing=type(base.lateral_mixing)(scheme="none"),
        surface_forcing=type(base.surface_forcing)(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=ShortwavePenetrationConfig(
            scheme="nemo_qsr_rgb", nemo_time_step_s=60.0
        ),
    )
    config = LatLonCGridOceanConfig.from_flat(
        A_h=0.0, B_h=0.0, K_h=0.0, K_bih=0.0, hyperdiff_coeff=0.0,
        A_v=1.0e-4, K_v=1.0e-5, n_barotropic_substeps=5,
        physics=physics, implicit_vertical_mixing=True,
        barotropic_solver="explicit_substep",
    )
    config = recipe._nemo_linear_bottom_drag(config)._replace(
        eos="nemo_eos80", implicit_ice_ocean_top_drag=True
    )
    model = LatLonCGridOceanModel(grid, z_coord, config)
    ocean_card = SimpleNamespace(
        dt_s=60.0,
        recipe=SimpleNamespace(z_coord=z_coord, model_config=config),
    )

    zero = jnp.zeros((6, 8), dtype=jnp.float64)
    one = jnp.ones((6, 8), dtype=jnp.float64)
    atmosphere = bulk.NemoStandaloneAtmosphereFields(
        jnp.full_like(zero, 5.0), jnp.full_like(zero, 2.0),
        jnp.full_like(zero, 270.0), jnp.full_like(zero, 0.002),
        jnp.full_like(zero, 101000.0), jnp.full_like(zero, 80.0),
        jnp.full_like(zero, 250.0), jnp.full_like(zero, 1.0e-7),
        jnp.full_like(zero, 4.0e-8),
    )
    pair = bulk.NemoStandaloneFieldPair(zero, zero, 60, 59, 61)
    s0f = bulk.NemoStandaloneS0FInputs(atmosphere, atmosphere, 60, 59, 61, pair, pair)
    forcing = bulk.NemoORCA2SI3Forcing(s0f, o17, jnp.full_like(zero, 0.05), zero)

    index = jnp.arange(48, dtype=jnp.int64).reshape(6, 8)
    center = jnp.roll(index, 1, axis=1)
    east = jnp.roll(center, -1, axis=1)
    north = jnp.roll(center, -1, axis=0)
    lbc = bulk.NemoTStressLbcPlan(center, east, north, one, one, one)
    card = bulk.NemoORCA2SI3Card(
        ocean_card, ice_card, lbc, one, one, zero, one,
        one, one, one, one, jnp.asarray(1.0), jnp.asarray(1.0),
        jnp.asarray(-166.67), jnp.asarray(4.0), jnp.asarray(86400.0),
        jnp.asarray(365.0), jnp.asarray(1.0e-20), jnp.asarray(3991.86795711963),
        2920, 2, jnp.asarray(0.066), jnp.asarray(1.0), jnp.asarray(1.0), 5,
    )
    stress_inputs = o17.stress._replace(tmask_k1=jnp.ones_like(o17.stress.u_ice))
    o17 = o17._replace(stress=stress_inputs)
    forcing = forcing._replace(o17=o17)
    stress = si3_ice_update_tau(
        stress_inputs, config=ice_card.state, ln_drgice_imp=True, nn_fsbc=2, kt=1
    )
    owned = (8, 6)
    initial_fluxes = SI3UpdateFluxOutputs(*(
        jnp.zeros(owned, dtype=jnp.float64) for _ in SI3UpdateFluxOutputs._fields
    ))
    means = NemoSurfaceMeans(zero, zero, zero, zero, zero, zero, one)
    runoff = NemoRunoffSurface(zero, zero, zero)
    carry = bulk.nemo_orca2_si3_cold_start(
        ocean=state, ice=entry, means=means, stress=stress,
        fluxes=initial_fluxes, runoff=runoff,
        restoring_volume_flux=zero, restoring_heat_flux=zero,
        restoring_coefficient=one, rn_fwb0=jnp.asarray(0.0), kt=1, card=card,
    )
    carry = carry._replace(
        a_fwb=jnp.asarray(0.0), a_fwb_b=jnp.asarray(1.0),
        emp_corr=jnp.asarray(0.1),
    )
    fixture = (carry, forcing, card, model)
    return (*fixture, _real_pair(fixture))


def _real_pair(fixture):
    carry, forcing, card, model = fixture[:4]
    return bulk.nemo_orca2_si3_coupled_pair(
        carry, forcing, card, even_forcing=forcing, model=model,
        sponge=None, even_sponge=None,
    )[0]


_C2_PLANTS = (
    "skip-even-tau", "run-si3-even", "drop-hm-i", "drop-rcdu-ice",
    "skip-fwb-application", "wrong-sign-center-lbc",
)


def _install_c2_route_plant(monkeypatch, plant):
    from legoesm.ice import si3_finalize
    from legoesm.ocean.forcing import nemo_surface

    if plant == "skip-even-tau":
        original = si3_finalize.si3_ice_update_tau

        def planted(inputs, **kwargs):
            if kwargs["kt"] % kwargs["nn_fsbc"] == 0:
                # Skip only the even-step private-field hold by executing the
                # refresh arm; the real odd step remains completely unchanged.
                kwargs["kt"] -= 1
            return original(inputs, **kwargs)

        monkeypatch.setattr(si3_finalize, "si3_ice_update_tau", planted)
    elif plant in ("drop-hm-i", "drop-rcdu-ice"):
        original = bulk.nemo_orca2_ocean_consumption_step

        def planted(card, state, *, surface_forcing, **kwargs):
            name = (
                "ice_mean_thickness_hm_i"
                if plant == "drop-hm-i"
                else "ice_ocean_top_drag_rCdu"
            )
            changed = surface_forcing._replace(
                **{name: jnp.zeros_like(getattr(surface_forcing, name))}
            )
            return original(card, state, surface_forcing=changed, **kwargs)

        monkeypatch.setattr(bulk, "nemo_orca2_ocean_consumption_step", planted)
    elif plant == "skip-fwb-application":
        original = nemo_surface.nemo_fwb_case2_step

        def planted(carry, **kwargs):
            result = original(carry, **kwargs)
            if (kwargs["kt"] - 1) % kwargs["forcing_stride"] == 0:
                # Omit only sbcfwb.F90:388-390.  The independent annual state
                # sample above it executes with the real kt and annual_steps.
                return result._replace(emp=kwargs["emp"], qns=kwargs["qns"])
            return result

        monkeypatch.setattr(bulk, "nemo_fwb_case2_step", planted)
    elif plant == "wrong-sign-center-lbc":
        def planted(u, v, _plan):
            # Omit sbcmod.F90:534's lbc_lnk call: interpolation below sees
            # untreated T stresses and their untreated east/north neighbors.
            east = jnp.concatenate((u[:, 1:], u[:, -1:]), axis=1)
            north = jnp.concatenate((v[1:, :], v[-1:, :]), axis=0)
            return u, v, east, north

        monkeypatch.setattr(bulk, "nemo_t_stress_lbc_neighbors", planted)
    elif plant == "run-si3-even":
        # Omit the production cadence guard.  The real coupled step now takes
        # its own SI3 path on even kt; no external SI3 call is injected.
        monkeypatch.setattr(
            bulk, "_nemo_orca2_si3_surface_update_due", lambda _kt, _stride: True
        )
    else:
        raise AssertionError(f"unknown C2 route plant {plant}")


@pytest.mark.parametrize("plant", _C2_PLANTS)
def test_each_c2_route_plant_changes_real_pair(real_coupled_fixture, monkeypatch, plant):
    baseline = real_coupled_fixture[4]
    _install_c2_route_plant(monkeypatch, plant)
    planted = _real_pair(real_coupled_fixture)
    assert _bits(planted) != _bits(baseline)


@pytest.mark.parametrize("plant", _C2_PLANTS)
def test_each_c2_route_plant_fails_real_rec_c2_scorer(monkeypatch, plant):
    from scripts.validate.ocean_fidelity.testcases.nemo_coupled_c_ladder_acquisition import (
        scorer,
    )

    evidence = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round31/"
        "oracle_coupled_c_ladder"
    )
    if not (evidence / "REC_C2").is_dir():
        pytest.skip("read-only REC_C2 evidence is unavailable")
    if plant == "skip-fwb-application":
        original_cold_start = scorer._c2_cold_start

        def controlled_cold_start(run):
            carry, forcing, sponge, card, model = original_cold_start(run)
            # Synthetic observability control: REC_C2 has rn_fwb0=0, so the
            # correction application is structurally unobservable here.  The
            # first admitted real rung that exercises it is the future annual
            # (2920-step) restart/continuation rung beyond C4.
            carry = carry._replace(emp_corr=jnp.asarray(0.1, dtype=jnp.float64))
            return carry, forcing, sponge, card, model

        monkeypatch.setattr(scorer, "_c2_cold_start", controlled_cold_start)
        control, _, masks = scorer.run_c2_compiled_observations(evidence)
        _install_c2_route_plant(monkeypatch, plant)
        candidate, _, _ = scorer.run_c2_compiled_observations(evidence)
        report = scorer.score_c2_observations(candidate, control, masks)
    else:
        _install_c2_route_plant(monkeypatch, plant)
        report = scorer.run_c2_compiled(evidence)
    changed_wet = sum(row["wet_different"] for row in report["field_counts"])
    changed_dry = sum(row["dry_other_different"] for row in report["field_counts"])
    print(
        f"C2_PLANT {plant} status={report['status']} "
        f"wet_different={changed_wet} dry_other_different={changed_dry}"
    )
    assert report["status"].startswith("FAIL_C2"), report["status"]
    assert changed_wet or changed_dry


def test_compiled_real_pair_has_finite_reverse_mode_gradient(real_coupled_fixture):
    carry, forcing, card, model = real_coupled_fixture[:4]

    def loss(x):
        seeded = carry._replace(
            ocean=carry.ocean._replace(
                eta=carry.ocean.eta.replace(data=carry.ocean.eta.data + x)
            )
        )
        even, _ = bulk.nemo_orca2_si3_coupled_pair(
            seeded, forcing, card, even_forcing=forcing, model=model,
            sponge=None, even_sponge=None,
        )
        return jnp.sum(even.ocean.eta.data * even.ocean.eta.data)

    value, gradient = jax.jit(jax.value_and_grad(loss))(jnp.asarray(0.25))
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))
    assert float(gradient) != 0.0


def _c_observations(steps=(1, 2)):
    from scripts.validate.ocean_fidelity.testcases.nemo_coupled_c_ladder_acquisition import (
        scorer,
    )

    values = {}
    for kind in (*scorer.C2_REPORT_ONLY_KINDS, *scorer.C2_VERDICT_KINDS):
        for kt in steps:
            if kind in (15, 16) and kt % 2 == 0:
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
        for kt in steps
        for rank in (0, 1)
    }
    return scorer, values, masks


def test_c2_d225_allows_only_dry_signed_zero_and_scores_kinds_1_2():
    scorer, expected, masks = _c_observations()
    candidate = {
        key: {name: value.copy() for name, value in fields.items()}
        for key, fields in expected.items()
    }
    masks[(1, 0)]["tmask"][2, 2] = 0.0
    candidate[(3, 1, 0)]["field"][2, 2] = -0.0
    candidate[(1, 1, 0)]["field"][2, 3] = np.nextafter(0.0, 1.0)
    report = scorer.score_c2_observations(candidate, expected, masks)
    assert report["status"].startswith("FAIL_C2 step=1 kind=1")
    assert report["counts"][3]["dry_zero_sign_different"] == 1
    assert report["counts"][1]["wet_different"] == 1
    assert report["report_only_counts"] == {}
    fields = {
        (row["kind"], row["kt"], row["field"]): row
        for row in report["field_counts"]
    }
    assert fields[(3, 1, "field")]["dry_zero_sign_different"] == 1
    assert fields[(3, 1, "field")]["dry_max_abs_difference"] == 0.0
    assert fields[(1, 1, "field")]["wet_different"] == 1
    assert fields[(1, 1, "field")]["wet_max_abs_difference"] == np.nextafter(0.0, 1.0)


def test_c2_d225_wet_plant_names_first_kind_field_cell_and_step():
    scorer, expected, masks = _c_observations()
    candidate = {
        key: {name: value.copy() for name, value in fields.items()}
        for key, fields in expected.items()
    }
    candidate[(4, 2, 1)]["field"][2, 3] = np.nextafter(0.0, 1.0)
    report = scorer.score_c2_observations(candidate, expected, masks)
    assert report["status"].startswith(
        "FAIL_C2 step=2 kind=4 field=field cell=(2, 3) rank=1"
    )


def test_u1_kind1_kind2_use_staggered_velocity_masks():
    scorer, expected, masks = _c_observations()
    for (kind, _kt, _rank), fields in expected.items():
        template = fields["field"]
        if kind == 1:
            fields.clear()
            fields.update(
                ssu_prior=template.copy(), ssv_prior=template.copy(),
                u_before=template.copy(), v_before=template.copy(),
            )
        elif kind == 2:
            fields.clear()
            fields.update(ssu=template.copy(), ssv=template.copy())
    candidate = {
        key: {name: value.copy() for name, value in fields.items()}
        for key, fields in expected.items()
    }
    for mask in masks.values():
        mask["umask"][2, 3] = 0.0
        mask["vmask"][2, 3] = 0.0
    for field in ("ssu_prior", "ssv_prior", "u_before", "v_before"):
        candidate[(1, 1, 0)][field][2, 3] = -0.0
    for field in ("ssu", "ssv"):
        candidate[(2, 1, 0)][field][2, 3] = -0.0
    report = scorer.score_c2_observations(candidate, expected, masks)
    assert report["status"] == "PASS_C2_0_ULP_WET"
    assert report["counts"][1]["dry_zero_sign_different"] == 4
    assert report["counts"][2]["dry_zero_sign_different"] == 2


def test_c3_late_wet_one_ulp_plant_turns_pass_into_fail():
    scorer, expected, masks = _c_observations((99, 100))
    candidate = {
        key: {name: value.copy() for name, value in fields.items()}
        for key, fields in expected.items()
    }
    baseline = scorer._score_c_observations(
        candidate, expected, masks, steps=(99, 100), mode="C3"
    )
    assert baseline["status"] == "PASS_C3_0_ULP_WET"
    candidate[(1, 99, 0)]["field"][2, 3] = np.nextafter(0.0, 1.0)
    planted = scorer._score_c_observations(
        candidate, expected, masks, steps=(99, 100), mode="C3"
    )
    assert planted["status"].startswith(
        "FAIL_C3 step=99 kind=1 field=field cell=(2, 3) rank=0"
    )


def test_c2_full_halo_shard_preserves_internal_rank_halos():
    from scripts.validate.ocean_fidelity.testcases.nemo_coupled_c_ladder_acquisition import (
        scorer,
    )

    global_field = np.arange(184 * 152, dtype=np.float64).reshape(184, 152)
    for rank, start in ((0, 0), (1, 90)):
        got = scorer._c2_shard(global_field, (94, 152), rank)
        np.testing.assert_array_equal(got, global_field[start:start + 94])
