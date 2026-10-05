"""Regression gates for the coupled QCO ww + Kmm-thickness ZAD path."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from types import SimpleNamespace

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    _nemo_qco_zad_operands,
    nemo_qco_kmm_velocity_cycle,
    nemo_qco_wzv_operands,
)
from legoesm.ocean.experiments.dino import DINOConfig, dino_config_for_recipe
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import (
    create_levy_stretched_z_star,
    create_partial_cell_coordinate,
)


def test_qco_zad_pair_matches_source_ordered_oracle():
    nlat, nlon, nlev = 3, 4, 3
    base = create_levy_stretched_z_star(nlev, 30.0, 5.0, 2.0, 1.0)
    h0 = np.broadcast_to(np.asarray(base.dz_ref), (nlat, nlon, nlev)).copy()
    coord = create_partial_cell_coordinate(
        base, jnp.full((nlat, nlon), 30.0))._replace(
            nemo_e3t_0=jnp.asarray(h0),
            nemo_hu_0=jnp.asarray(h0.sum(axis=-1)),
            nemo_hv_0=jnp.asarray(h0.sum(axis=-1)),
            nemo_e1e2t=jnp.asarray(2.0 + np.arange(nlat * nlon).reshape(nlat, nlon) / 7),
            nemo_e1e2u=jnp.asarray(3.0 + np.arange(nlat * nlon).reshape(nlat, nlon) / 11),
            nemo_e1e2v=jnp.asarray(4.0 + np.arange(nlat * nlon).reshape(nlat, nlon) / 13),
            nemo_e2u=jnp.asarray(1.5 + np.arange(nlat * nlon).reshape(nlat, nlon) / 17),
            nemo_e1v=jnp.asarray(1.7 + np.arange(nlat * nlon).reshape(nlat, nlon) / 19),
        )
    rng = np.random.default_rng(1455)
    eta_now = rng.normal(scale=0.1, size=(nlat, nlon))
    eta_before = rng.normal(scale=0.1, size=(nlat, nlon))
    u_native = rng.normal(scale=1e-2, size=(nlat, nlon, nlev))
    v_native = rng.normal(scale=1e-2, size=(nlat, nlon, nlev))
    u = np.concatenate([u_native[:, -1:], u_native], axis=1)
    v = np.concatenate([np.zeros_like(v_native[:1]), v_native], axis=0)
    um = np.ones_like(u)
    vm = np.ones_like(v)
    vm[0] = 0.0
    vm[-1] = 0.0
    tm = np.ones((nlat, nlon, nlev))
    dt = 5400.0
    e2u = np.asarray(coord.nemo_e2u)
    e1v = np.asarray(coord.nemo_e1v)
    grid = SimpleNamespace(
        dy_u=jnp.asarray(np.concatenate([e2u[:, -1:], e2u], axis=1)),
        dx_v=jnp.asarray(np.concatenate([np.zeros_like(e1v[:1]), e1v], axis=0)),
        area=jnp.asarray(coord.nemo_e1e2t),
    )

    ww, hu, hv = _nemo_qco_zad_operands(
        jnp.asarray(eta_now), jnp.asarray(eta_before), jnp.asarray(u),
        jnp.asarray(v), grid, coord, jnp.asarray(um), jnp.asarray(vm),
        jnp.asarray(tm), dt, after_ssh_form="leapfrog_continuity")

    at = np.asarray(coord.nemo_e1e2t)
    au = np.asarray(coord.nemo_e1e2u)
    av = np.asarray(coord.nemo_e1e2v)
    weighted = at * eta_now
    r3u = 0.5 * (weighted + np.roll(weighted, -1, axis=1)) / 30.0 / au
    r3v = 0.5 * (weighted + np.roll(weighted, -1, axis=0)) / 30.0 / av
    hu_raw = h0 * (1.0 + r3u[..., None])
    hv_raw = h0 * (1.0 + r3v[..., None])
    # NEMO carries reference e3 on the closed outer V row; the transport mask
    # zeros that row separately (same operand separation as dom_qco_r3c).
    hv_raw[-1] = h0[-1]
    zu = e2u[..., None] * hu_raw * u_native
    zv = e1v[..., None] * hv_raw * v_native
    zv[-1] = 0.0
    south = np.concatenate([np.zeros_like(zv[:1]), zv[:-1]], axis=0)
    div = (zu - np.roll(zu, 1, axis=1) + zv - south) / at[..., None]
    eta_after = eta_before - dt * np.sum(div, axis=-1)
    stretch = h0 * ((eta_after - eta_before) / 30.0 / dt)[..., None]
    expected_w = np.zeros((nlat, nlon, nlev + 1))
    for k in range(nlev - 1, -1, -1):
        expected_w[..., k] = expected_w[..., k + 1] - div[..., k] - stretch[..., k]

    np.testing.assert_array_equal(np.asarray(hu)[:, 1:], hu_raw)
    np.testing.assert_array_equal(np.asarray(hv)[1:], hv_raw)
    # Red-capable row-8.5 plant: replacing the live Kmm QCO thickness with
    # the reference/generic face thickness must be detectably different on
    # this nonzero-SSH state.
    assert np.any(np.asarray(hu)[:, 1:] != h0)
    assert np.any(np.asarray(hv)[1:] != h0)
    np.testing.assert_allclose(np.asarray(ww), expected_w, rtol=0, atol=2e-16)
    np.testing.assert_array_equal(np.asarray(ww)[..., -1], 0.0)
    assert np.any(np.asarray(ww)[..., 1:-1] != 0.0)

    # WZV call 2 uses the SAME second-hdiv composition but the actual
    # barotropic Kaa SSH/r3t.  A Kaa override must change W, follow the
    # literal bottom-up recurrence, and remain JIT/grad capable.
    eta_override = eta_after + np.linspace(
        -2e-3, 3e-3, nlat * nlon).reshape(nlat, nlon)
    ww2, hu2, hv2 = nemo_qco_wzv_operands(
        jnp.asarray(eta_now), jnp.asarray(eta_before), jnp.asarray(u),
        jnp.asarray(v), grid, coord, jnp.asarray(um), jnp.asarray(vm),
        jnp.asarray(tm), dt, eta_after_override=jnp.asarray(eta_override))
    stretch2 = h0 * ((eta_override - eta_before) / 30.0 / dt)[..., None]
    expected_w2 = np.zeros((nlat, nlon, nlev + 1))
    for k in range(nlev - 1, -1, -1):
        expected_w2[..., k] = expected_w2[..., k + 1] - div[..., k] - stretch2[..., k]
    np.testing.assert_allclose(np.asarray(ww2), expected_w2, rtol=0, atol=2e-16)
    np.testing.assert_array_equal(hu2, hu)
    np.testing.assert_array_equal(hv2, hv)
    assert not np.array_equal(np.asarray(ww2), np.asarray(ww))

    # Call 2 first rewrites the entry Kmm velocity with NEMO's secondary
    # transport mean (dynspg_ts.F90:1170-1174), then recomputes div_hor.
    un_adv_native = rng.normal(scale=0.4, size=(nlat, nlon))
    vn_adv_native = rng.normal(scale=0.4, size=(nlat, nlon))
    un_adv = np.concatenate(
        [un_adv_native[:, -1:], un_adv_native], axis=1)
    vn_adv = np.concatenate(
        [np.zeros_like(vn_adv_native[:1]), vn_adv_native], axis=0)
    ww3, _, _ = nemo_qco_wzv_operands(
        jnp.asarray(eta_now), jnp.asarray(eta_before), jnp.asarray(u),
        jnp.asarray(v), grid, coord, jnp.asarray(um), jnp.asarray(vm),
        jnp.asarray(tm), dt, eta_after_override=jnp.asarray(eta_override),
        transport_after_override=(jnp.asarray(un_adv), jnp.asarray(vn_adv)))
    r1u = 1.0 / (30.0 * (1.0 + r3u))
    r1v = 1.0 / (30.0 * (1.0 + r3v))
    puu_b = np.sum(hu_raw * u_native, axis=-1) * r1u
    pvv_b = np.sum(hv_raw * v_native, axis=-1) * r1v
    u_call2 = u_native + un_adv_native[..., None] * r1u[..., None] \
        - puu_b[..., None]
    v_call2 = v_native + vn_adv_native[..., None] * r1v[..., None] \
        - pvv_b[..., None]
    v_call2 = v_call2 * vm[1:]
    corrected_u, corrected_v, restored_u, restored_v = (
        nemo_qco_kmm_velocity_cycle(
            jnp.asarray(eta_now), jnp.asarray(u), jnp.asarray(v),
            jnp.asarray(un_adv), jnp.asarray(vn_adv), coord,
            jnp.asarray(um), jnp.asarray(vm)))
    np.testing.assert_allclose(
        np.asarray(corrected_u)[:, 1:], u_call2, rtol=0, atol=2e-16)
    np.testing.assert_allclose(
        np.asarray(corrected_v)[1:], v_call2, rtol=0, atol=2e-16)
    restored_u_expected = (
        u_call2 - un_adv_native[..., None] * r1u[..., None]
        + puu_b[..., None])
    restored_v_expected = (
        v_call2 - vn_adv_native[..., None] * r1v[..., None]
        + pvv_b[..., None]) * vm[1:]
    np.testing.assert_allclose(
        np.asarray(restored_u)[:, 1:], restored_u_expected,
        rtol=0, atol=2e-16)
    np.testing.assert_allclose(
        np.asarray(restored_v)[1:], restored_v_expected,
        rtol=0, atol=2e-16)
    # Tracer entry consumes the execute half, before stpmlf undoes it for the
    # Asselin filter.  A stale/generic Kmm handoff is a red-capable violation.
    assert np.any(np.asarray(corrected_u)[:, 1:] != u_native)
    assert np.any(np.asarray(corrected_v)[1:] != v_native)
    # Planted violation: skipping the execute+undo cycle is algebraically
    # tempting and measurably wrong in floating-point arithmetic.
    assert np.any(np.asarray(restored_u)[:, 1:] != u_native)
    assert np.any(np.asarray(restored_v)[1:] != v_native)
    zu3 = e2u[..., None] * hu_raw * u_call2
    zv3 = e1v[..., None] * hv_raw * v_call2
    zv3[-1] = 0.0
    south3 = np.concatenate([np.zeros_like(zv3[:1]), zv3[:-1]], axis=0)
    div3 = (zu3 - np.roll(zu3, 1, axis=1) + zv3 - south3) / at[..., None]
    expected_w3 = np.zeros((nlat, nlon, nlev + 1))
    for k in range(nlev - 1, -1, -1):
        expected_w3[..., k] = (
            expected_w3[..., k + 1] - div3[..., k] - stretch2[..., k])
    np.testing.assert_allclose(np.asarray(ww3), expected_w3, rtol=0, atol=3e-16)
    assert not np.array_equal(np.asarray(ww3), np.asarray(ww2))

    # traadv.F90:220-226 hands wzv(np_transport) the already materialized
    # pFu/pFv pair.  The override must consume those exact transports rather
    # than rebuilding them from velocity, and it must retain the same literal
    # bottom-up recurrence.
    volume_u = np.concatenate([zu[:, -1:], zu], axis=1)
    volume_v = np.concatenate([np.zeros_like(zv[:1]), zv], axis=0)
    ww4, _, _ = nemo_qco_wzv_operands(
        jnp.asarray(eta_now), jnp.asarray(eta_before), jnp.zeros_like(u),
        jnp.zeros_like(v), grid, coord, jnp.asarray(um), jnp.asarray(vm),
        jnp.asarray(tm), dt, eta_after_override=jnp.asarray(eta_override),
        volume_transport_override=(
            jnp.asarray(volume_u), jnp.asarray(volume_v)))
    np.testing.assert_allclose(np.asarray(ww4), expected_w2, rtol=0, atol=2e-16)

    with pytest.raises(ValueError, match="mutually exclusive"):
        nemo_qco_wzv_operands(
            jnp.asarray(eta_now), jnp.asarray(eta_before), jnp.asarray(u),
            jnp.asarray(v), grid, coord, jnp.asarray(um), jnp.asarray(vm),
            jnp.asarray(tm), dt, eta_after_override=jnp.asarray(eta_override),
            transport_after_override=(jnp.asarray(un_adv), jnp.asarray(vn_adv)),
            volume_transport_override=(
                jnp.asarray(volume_u), jnp.asarray(volume_v)))

    with pytest.raises(ValueError, match="requires eta_after_override"):
        nemo_qco_wzv_operands(
            jnp.asarray(eta_now), jnp.asarray(eta_before), jnp.asarray(u),
            jnp.asarray(v), grid, coord, jnp.asarray(um), jnp.asarray(vm),
            jnp.asarray(tm), dt,
            transport_after_override=(jnp.asarray(un_adv), jnp.asarray(vn_adv)))

    def objective(eta_a):
        value, _, _ = nemo_qco_wzv_operands(
            jnp.asarray(eta_now), jnp.asarray(eta_before), jnp.asarray(u),
            jnp.asarray(v), grid, coord, jnp.asarray(um), jnp.asarray(vm),
            jnp.asarray(tm), dt, eta_after_override=eta_a,
            transport_after_override=(
                jnp.asarray(un_adv), jnp.asarray(vn_adv)))
        return jnp.sum(value * value)

    compiled = jax.jit(objective)(jnp.asarray(eta_override))
    gradient = jax.jit(jax.grad(objective))(jnp.asarray(eta_override))
    assert np.isfinite(np.asarray(compiled))
    assert np.isfinite(np.asarray(gradient)).all()
    assert np.any(np.asarray(gradient) != 0.0)


def test_qco_selector_defaults_are_scoped_to_two_dino_cards():
    assert DINOConfig().zad_qco_evaluation == "generic"
    assert DINOConfig().wzv_call2_evaluation == "generic"
    faithful = {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
    for recipe in (
        "legoesm_default", "nemo_paper", "nemo_dino_kamm",
        "nemo_dino_kamm_mlf", "veros", "mitgcm", "oceananigans",
    ):
        expected = "nemo_literal" if recipe in faithful else "generic"
        assert dino_config_for_recipe(recipe).zad_qco_evaluation == expected
        assert dino_config_for_recipe(recipe).wzv_call2_evaluation == expected


def test_qco_call2_selector_is_red_for_unknown_and_half_configurations():
    with pytest.raises(ValueError, match="wzv_call2_evaluation must be"):
        LatLonCGridOceanModel._validate_config(
            LatLonCGridOceanConfig.from_flat(wzv_call2_evaluation="typo"))
    with pytest.raises(ValueError, match="requires zad_qco_evaluation"):
        LatLonCGridOceanModel._validate_config(
            LatLonCGridOceanConfig.from_flat(
                wzv_call2_evaluation="nemo_literal",
                zad_qco_evaluation="generic"))


def _wzv_operand_fixture():
    """The same small QCO mesh the oracle-ordering test above builds."""
    nlat, nlon, nlev = 3, 4, 3
    base = create_levy_stretched_z_star(nlev, 30.0, 5.0, 2.0, 1.0)
    h0 = np.broadcast_to(np.asarray(base.dz_ref), (nlat, nlon, nlev)).copy()
    idx = np.arange(nlat * nlon).reshape(nlat, nlon)
    coord = create_partial_cell_coordinate(
        base, jnp.full((nlat, nlon), 30.0))._replace(
            nemo_e3t_0=jnp.asarray(h0),
            nemo_hu_0=jnp.asarray(h0.sum(axis=-1)),
            nemo_hv_0=jnp.asarray(h0.sum(axis=-1)),
            nemo_e1e2t=jnp.asarray(2.0 + idx / 7),
            nemo_e1e2u=jnp.asarray(3.0 + idx / 11),
            nemo_e1e2v=jnp.asarray(4.0 + idx / 13),
            nemo_e2u=jnp.asarray(1.5 + idx / 17),
            nemo_e1v=jnp.asarray(1.7 + idx / 19),
        )
    rng = np.random.default_rng(1455)
    eta_now = rng.normal(scale=0.1, size=(nlat, nlon))
    eta_before = rng.normal(scale=0.1, size=(nlat, nlon))
    u_native = rng.normal(scale=1e-2, size=(nlat, nlon, nlev))
    v_native = rng.normal(scale=1e-2, size=(nlat, nlon, nlev))
    u = np.concatenate([u_native[:, -1:], u_native], axis=1)
    v = np.concatenate([np.zeros_like(v_native[:1]), v_native], axis=0)
    um = np.ones_like(u)
    vm = np.ones_like(v)
    vm[0] = 0.0
    vm[-1] = 0.0
    tm = np.ones((nlat, nlon, nlev))
    e2u = np.asarray(coord.nemo_e2u)
    e1v = np.asarray(coord.nemo_e1v)
    grid = SimpleNamespace(
        dy_u=jnp.asarray(np.concatenate([e2u[:, -1:], e2u], axis=1)),
        dx_v=jnp.asarray(np.concatenate([np.zeros_like(e1v[:1]), e1v], axis=0)),
        area=jnp.asarray(coord.nemo_e1e2t),
    )
    args = (jnp.asarray(eta_now), jnp.asarray(eta_before), jnp.asarray(u),
            jnp.asarray(v), grid, coord, jnp.asarray(um), jnp.asarray(vm),
            jnp.asarray(tm))
    return args, 5400.0, eta_now, eta_before


def test_wzv_after_ssh_form_is_the_time_stepping_program():
    """The first wzv call's "after" SSH follows NEMO's OWN time-stepping
    program, and an unknown form is refused rather than defaulted.

    NEMO's RK3 program leaves a LINEAR EXTRAPOLATION in the after slot at the
    end of every step (``ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)``,
    stprk3.F90:225) and ``stp_2D`` reads it straight into ``r3t(:,:,Kaa)``
    (stp2d.F90:149) before ``CALL wzv`` (stp2d.F90:153).  The modified
    leapfrog instead fills that slot from the barotropic continuity in
    ``ssh_nxt``.  The two give DIFFERENT vertical velocities, so the form is
    a transcription of the program the card runs, not a preference.

    Non-vacuity: the two forms are asserted to DISAGREE, so a change that
    collapsed them would turn this red; and the RK3 arm is pinned against
    the closed form ``2*eta_now - eta_before`` passed explicitly.
    """
    args, dt, eta_now, eta_before = _wzv_operand_fixture()

    ww_rk3, _, _ = nemo_qco_wzv_operands(
        *args, dt, after_ssh_form="rk3_extrapolated")
    ww_mlf, _, _ = nemo_qco_wzv_operands(
        *args, dt, after_ssh_form="leapfrog_continuity")
    assert not np.array_equal(np.asarray(ww_rk3), np.asarray(ww_mlf)), (
        "the two time-stepping programs must give different vertical "
        "velocities, or this test cannot fail")

    ww_explicit, _, _ = nemo_qco_wzv_operands(
        *args, dt,
        eta_after_override=jnp.asarray(2.0 * eta_now - eta_before))
    np.testing.assert_array_equal(np.asarray(ww_rk3),
                                  np.asarray(ww_explicit))

    with pytest.raises(ValueError, match="nemo_first_wzv_after_ssh"):
        nemo_qco_wzv_operands(*args, dt, after_ssh_form="whatever")


def test_vortex_vector_card_selects_nemos_own_first_wzv():
    """The vector-EEN VORTEX card runs dyn_zad, so it must consume NEMO's
    own wzv vertical velocity rather than legoESM's generic z-star
    diagnosis.  Round 5 measured the generic form as the WHOLE of that
    card's first-stage momentum error.  The flux card never calls dyn_zad
    and is left alone, which is asserted here so the pair stays one
    variable apart."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    vec = build_nemo_testcase_card("VORTEX_VEC-zco").recipe.model_config
    assert vec.zad_qco_evaluation == "nemo_literal"
    assert vec.vertical_momentum_scheme == "nemo_advective"
    assert vec.momentum_time_integrator == "rk3_ws"
    flux = build_nemo_testcase_card("VORTEX-zco").recipe.model_config
    assert flux.zad_qco_evaluation == "generic"


def test_an_unstated_after_ssh_form_raises_rather_than_guessing():
    """Unset is a hard error, not a default.  A card that resolves NEMO's own
    first wzv call states which program's after-SSH slot it reads; inferring
    it from a sibling selector is the hidden coupling decision 75 bans."""
    args, dt, _, _ = _wzv_operand_fixture()
    with pytest.raises(ValueError, match="nemo_first_wzv_after_ssh"):
        nemo_qco_wzv_operands(*args, dt, after_ssh_form="")


def test_the_vortex_and_gyre_cards_state_the_form_and_dino_states_the_other():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    for case in ("VORTEX_VEC-zco", "GYRE-zco"):
        cfg = build_nemo_testcase_card(case).recipe.model_config
        assert cfg.nemo_first_wzv_after_ssh == "rk3_extrapolated_carried"
    assert (dino_config_for_recipe("nemo_dino_kamm_mlf")
            .nemo_first_wzv_after_ssh == "leapfrog_continuity")
