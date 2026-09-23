"""ORCA2 round 10 — the gate that checks the PRODUCTION ROUTING of qsr_RGBc.

The RGB kernel is gated elsewhere.  What this gate asserts is that the operand
expressions it feeds the kernel are the ones the production sites build, so
these tests pin exactly that: the live/reference ladder pair, the wet mask that
falls out of the partial-cell thickness, and the two controls' non-vacuity.
They run on a tiny synthetic column and need neither the deck nor the record.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round10_qsr_routing_gate as gate,
)
from legoesm.ocean.physics.shortwave_penetration import (  # noqa: E402
    ShortwavePenetrationConfig,
    apply_shortwave_penetration,
)

NLAT, NLON, NLEV = 3, 4, 6


def _partial_cell_coordinate(h_ref, z_coord):
    """A real OceanPartialCellCoordinate whose reference column IS ``h_ref``."""
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate, create_z_star_from_thicknesses)

    return create_partial_cell_coordinate(
        create_z_star_from_thicknesses(np.asarray(z_coord.dz_ref)),
        np.asarray(h_ref).sum(axis=-1),
    )


def _operands():
    dz_ref = np.array([10.0, 12.0, 15.0, 20.0, 30.0, 50.0])
    z_half_ref = -np.concatenate([[0.0], np.cumsum(dz_ref)])
    z_coord = types.SimpleNamespace(dz_ref=dz_ref, z_half_ref=z_half_ref)
    config = types.SimpleNamespace(
        rho_0=1026.0,
        physics=types.SimpleNamespace(
            constants=types.SimpleNamespace(c_sw=3991.86795711963)),
    )
    sw_cfg = ShortwavePenetrationConfig(
        scheme="nemo_qsr_rgb", rgb_ir_fraction=0.58,
        rgb_ir_extinction_m=0.35, rgb_chl_profile="morel_berthon",
        nemo_time_step_s=10800.0)
    rng = np.random.default_rng(10)
    tmask = np.ones((NLAT, NLON, NLEV))
    tmask[0, 0, 3:] = 0.0           # a shallow column
    tmask[1, 2, :] = 0.0            # a land column
    h_ref = np.broadcast_to(dz_ref, (NLAT, NLON, NLEV)) * tmask
    return (sw_cfg, config, z_coord,
            dict(qsr=rng.uniform(0.0, 300.0, (NLAT, NLON)),
                 chl=rng.uniform(0.02, 3.0, (NLAT, NLON)),
                 h_ref=h_ref,
                 r3t=rng.uniform(-1.0e-3, 1.0e-3, (NLAT, NLON)),
                 tmask=tmask))


def test_the_gate_uses_the_PRODUCTION_thickness_and_stretch_helpers():
    """Not a retyping of the gate: the reference comes from the model's own
    ``compute_layer_thickness`` / ``compute_ocean_jacobian``, which is what the
    shared pipeline calls.  If the gate ever drifts to a different live
    ladder, this fails."""
    from legoesm.ocean.vertical import (
        compute_layer_thickness, compute_ocean_jacobian)

    sw_cfg, config, z_coord, kw = _operands()
    # A real partial-cell coordinate whose reference column is the test's
    # h_ref, so the production helpers can be asked the same question.
    zc = _partial_cell_coordinate(kw["h_ref"], z_coord)
    H = kw["h_ref"].sum(axis=-1)
    eta = kw["r3t"] * H
    J = np.asarray(compute_ocean_jacobian(jnp.asarray(eta),
                                          jnp.asarray(H), zc))
    dz_live = np.asarray(compute_layer_thickness(jnp.asarray(eta),
                                                 jnp.asarray(H), zc))
    gdepw_ref = -jnp.asarray(z_coord.z_half_ref)
    want = np.asarray(apply_shortwave_penetration(
        sw_cfg, jnp.asarray(kw["qsr"]),
        chl=jnp.asarray(kw["chl"]),
        dz_live=jnp.asarray(dz_live),
        wet_cell=jnp.asarray(dz_live > 0.0, dtype=jnp.asarray(dz_live).dtype),
        gdepw_bottom_live=gdepw_ref[1:] * jnp.asarray(J)[..., None],
        gdepw_ref=gdepw_ref,
        e3t_ref=jnp.asarray(z_coord.dz_ref),
        rho_0=config.rho_0, c_sw=config.physics.constants.c_sw))
    got = np.asarray(gate.production_rgb_tendency(
        sw_cfg, config, z_coord, live=True, **kw))
    np.testing.assert_array_equal(got, want)


def test_the_shared_pipeline_still_REFUSES_the_generic_rgb_scheme():
    """Round 10 opened the pipeline to the NEMO identity selector ONLY.  The
    generic ``rgb_chl`` deposit is owned by the external surface-forcing
    stage; if the pipeline ever accepts it too, qsr is counted twice."""
    from legoesm.ocean.physics.shortwave_penetration import (
        shortwave_penetration_tendency)

    _, config, z_coord, kw = _operands()
    generic = ShortwavePenetrationConfig(scheme="rgb_chl")
    with pytest.raises(ValueError, match="two-band Jerlov kernel"):
        shortwave_penetration_tendency(
            jnp.asarray(kw["qsr"]), jnp.asarray(z_coord.dz_ref),
            jnp.asarray(z_coord.z_half_ref), jnp.ones_like(jnp.asarray(kw["qsr"])),
            generic, config.rho_0, config.physics.constants.c_sw)


def test_the_wet_mask_falls_out_of_the_partial_cell_thickness():
    """A dry cell must receive exactly zero, not a deposit on a phantom layer."""
    sw_cfg, config, z_coord, kw = _operands()
    got = np.asarray(gate.production_rgb_tendency(
        sw_cfg, config, z_coord, live=True, **kw))
    dry = kw["tmask"] == 0.0
    assert dry.any()
    assert np.all(got[dry] == 0.0)
    assert np.abs(got[~dry]).max() > 0.0


def test_the_static_ladder_control_is_not_vacuous():
    sw_cfg, config, z_coord, kw = _operands()
    live = np.asarray(gate.production_rgb_tendency(
        sw_cfg, config, z_coord, live=True, **kw))
    static = np.asarray(gate.production_rgb_tendency(
        sw_cfg, config, z_coord, live=False, **kw))
    assert not np.array_equal(live, static)


def test_an_unresolved_chlorophyll_profile_is_refused_not_substituted():
    sw_cfg, config, z_coord, kw = _operands()
    with pytest.raises(ValueError, match="nn_chlprfl"):
        gate.production_rgb_tendency(
            sw_cfg._replace(rgb_chl_profile="surface"), config, z_coord,
            live=True, **kw)


def test_the_resolved_table_names_every_setting_the_kernel_reads():
    """Each row is a claim about the record; none may be dropped silently."""
    assert set(gate.RESOLVED) == {
        "scheme", "rgb_chl_profile", "rgb_ir_fraction",
        "rgb_ir_extinction_m", "nemo_time_step_s"}
    for value, provenance in gate.RESOLVED.values():
        assert provenance and "ocean.output:" in provenance
