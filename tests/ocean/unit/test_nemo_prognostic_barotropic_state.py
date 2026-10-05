"""Decision-8 guards for NEMO's prognostic ``uu_b/vv_b`` state pair."""

from __future__ import annotations

import ast
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    barotropic_substeps_latlon_cgrid,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    _nemo_ws_stage_barotropic_velocity,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.restart import load_run_restart, save_run_restart
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


def _case(*, carried: bool):
    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=3, H_max=300.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=10.0, H_max=300.0,
        land_lat_threshold=70.0,
        nemo_prognostic_barotropic_velocity=carried,
    )
    return grid, z, state


def test_non_nemo_state_has_no_depth_mean_array_leaves():
    """The approved state expansion is confined to NEMO-identity recipes."""
    _, _, plain = _case(carried=False)
    _, _, nemo = _case(carried=True)
    assert plain.uu_b is None and plain.vv_b is None
    assert len(jax.tree_util.tree_leaves(nemo)) == (
        len(jax.tree_util.tree_leaves(plain)) + 2)
    assert nemo.uu_b.data.shape == nemo.u.data.shape[:-1]
    assert nemo.vv_b.data.shape == nemo.v.data.shape[:-1]
    assert np.count_nonzero(np.asarray(nemo.uu_b.data)) == 0
    assert np.count_nonzero(np.asarray(nemo.vv_b.data)) == 0


def test_external_mode_reads_and_rewrites_carried_pair():
    """A changed Kbb pair changes the production seed despite identical 3-D u/v."""
    grid, z, state = _case(carried=True)
    cfg = LatLonCGridOceanConfig(fix_eta_drift=False)
    cfg = cfg._replace(barotropic=cfg.barotropic._replace(
        # The NEMO identity is a CONFIG choice: the window reads the carried
        # pair because the card says so, not because the arrays exist.
        nemo_prognostic_barotropic_state=True,
        barotropic_solver="explicit_substep",
        bebt=0.0,
        maxvel_barotropic=0.0,
        barotropic_diffusion_alpha=0.0,
        barotropic_local_subcycle_clamp=False,
    ))
    u_seed = 1.0e-4 * state.u_mask.data
    seeded = state._replace(uu_b=state.uu_b.replace(data=u_seed))
    out_zero, _ = barotropic_substeps_latlon_cgrid(
        state, 1.0, 1, grid, z, cfg, add_barotropic_coriolis=False)
    out_seed, _ = barotropic_substeps_latlon_cgrid(
        seeded, 1.0, 1, grid, z, cfg, add_barotropic_coriolis=False)
    assert np.max(np.abs(np.asarray(
        out_seed.uu_b.data - out_zero.uu_b.data))) > 0.0
    assert out_seed.uu_b is not seeded.uu_b
    with pytest.raises(ValueError, match="both uu_b and vv_b"):
        barotropic_substeps_latlon_cgrid(
            seeded._replace(vv_b=None), 1.0, 1, grid, z, cfg,
            add_barotropic_coriolis=False)


def test_stage_time_levels_keep_kbb_out_of_same_step_nnn():
    """Stage-1 S-21 reads last-step Kbb; stages 2/3 read this-step Nnn.

    This is the direct negative guard for stprk3.F90:186,195-207.  Distinct
    sentinels make replacing stage 1's Kbb operand with the external-mode Naa
    target fail by value, rather than merely checking a source string.
    """
    kbb = (jnp.array([[1.0]]), jnp.array([[2.0]]))
    nnn = (jnp.array([[101.0]]), jnp.array([[202.0]]))
    for got, want in zip(
            _nemo_ws_stage_barotropic_velocity(1, kbb, nnn), kbb):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    for stage in (2, 3):
        for got, want in zip(
                _nemo_ws_stage_barotropic_velocity(stage, kbb, nnn), nnn):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    with pytest.raises(ValueError, match="stage must be 1, 2, or 3"):
        _nemo_ws_stage_barotropic_velocity(0, kbb, nnn)


def test_identity_path_has_one_external_mode_writer_for_carried_pair():
    """No live identity-path writer may compete with dynspg_ts's Kaa write."""
    root = Path(__file__).parents[3] / "packages" / "ocean" / "legoesm" / "ocean"
    found = []
    for area in (root / "dynamics", root / "fidelity"):
        for path in area.rglob("*.py"):
            tree = ast.parse(path.read_text())
            stack = []

            class PairWrites(ast.NodeVisitor):
                def visit_FunctionDef(self, node):
                    stack.append(node.name)
                    self.generic_visit(node)
                    stack.pop()

                visit_AsyncFunctionDef = visit_FunctionDef

                def visit_Call(self, node):
                    names = {kw.arg for kw in node.keywords}
                    if names & {"uu_b", "vv_b"}:
                        found.append((
                            str(path.relative_to(root)),
                            stack[-1] if stack else "<module>",
                        ))
                    self.generic_visit(node)

            PairWrites().visit(tree)

    # The production dynamics assignments are the standard- and wide-halo
    # forms of the same external-mode Kaa write.  The _step_impl assignment is
    # reachable only through the private stage-twin output override, whose
    # production default is None.  Fidelity assignments are construction,
    # restart parsing, or restart bridging; none executes as a second live
    # model writer.
    assert sorted(found) == sorted([
        ("dynamics/barotropic_latlon_cgrid.py",
         "barotropic_substeps_latlon_cgrid"),
        ("dynamics/barotropic_latlon_cgrid.py",
         "barotropic_substeps_wide_halo_latlon_cgrid"),
        ("dynamics/ocean_model_latlon_cgrid.py", "_step_impl"),
        ("fidelity/nemo_io.py", "read_nemo_restart"),
        ("fidelity/nemo_recipe.py", "build_nemo_eady_recipe"),
        # The VORTEX card is the first testcase card whose oracle starts with
        # a non-zero velocity, so it transcribes NEMO's own istate.F90:149-154
        # depth average into the card's construction-time state.  Card
        # construction, like the bridges above, is not a live model writer.
        ("fidelity/nemo_testcase_recipe.py", "build_vortex_zco_card"),
        # Decision 88's seamount pair runs the same construction-time depth
        # average over its own partial-cell e3u/e3v and hu_0/hv_0.  Same
        # classification as the flat card above: construction, not a live
        # model writer.
        ("fidelity/nemo_testcase_recipe.py", "build_vortex_smt_zps_card"),
        ("fidelity/nemo_state_bridge.py", "bridge_nemo_to_legoesm"),
        ("fidelity/nemo_state_bridge.py", "bridge_nemo_to_legoesm_topo"),
    ])


def test_generic_nemo_recipes_all_carry_pair_without_changing_base_eady():
    """Every NEMO recipe opts in; the underlying non-NEMO setup stays leaf-free."""
    from legoesm.ocean.experiments.eady_uniform import build_eady_uniform_setup
    from legoesm.ocean.fidelity.nemo_recipe import (
        build_nemo_eady_recipe,
        build_nemo_gyre_recipe,
        build_nemo_rest_recipe,
    )

    base = build_eady_uniform_setup(n_lat=8, n_lon=8, nlev=3)
    eady = build_nemo_eady_recipe(n_lat=8, n_lon=8, nlev=3)
    assert base.initial_state.uu_b is None and base.initial_state.vv_b is None
    for name, value in base.initial_state._asdict().items():
        if name in {"uu_b", "vv_b"}:
            continue
        other = getattr(eady.initial_state, name)
        if value is None:
            assert other is None
        else:
            np.testing.assert_array_equal(
                np.asarray(other.data), np.asarray(value.data))
    for recipe in (
        eady,
        build_nemo_rest_recipe(n_lat=6, n_lon=8, nlev=3),
        build_nemo_gyre_recipe(),
    ):
        assert recipe.initial_state.uu_b is not None
        assert recipe.initial_state.vv_b is not None


def test_run_restart_round_trips_depth_mean_bits(tmp_path):
    """The production restart inventory persists the pair as prognostic state."""
    _, _, state = _case(carried=True)
    ub = jnp.arange(state.uu_b.data.size, dtype=jnp.float64).reshape(
        state.uu_b.data.shape) * jnp.float64(2.0**-40)
    vb = jnp.arange(state.vv_b.data.size, dtype=jnp.float64).reshape(
        state.vv_b.data.shape) * jnp.float64(-2.0**-41)
    marked = state._replace(
        uu_b=state.uu_b.replace(data=ub), vv_b=state.vv_b.replace(data=vb))
    path = tmp_path / "nemo_identity.npz"
    save_run_restart(path, marked, step=5, time_days=5.0,
                     grid_type="latlon")
    got, _, meta = load_run_restart(
        path, state, grid_type="latlon")
    assert meta["step"] == 5
    np.testing.assert_array_equal(np.asarray(got.uu_b.data), np.asarray(ub))
    np.testing.assert_array_equal(np.asarray(got.vv_b.data), np.asarray(vb))


def test_nemo_identity_kt5_restart_matches_unbroken_kt6_to_10(tmp_path):
    """NEMO-style prognostics, including uu_b/vv_b, restart without a cold start."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    old_policy = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        card = build_nemo_testcase_card("LOCK_EXCHANGE-zco")
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
        unbroken = card.recipe.initial_state
        checkpoint = tmp_path / "kt5.npz"
        after = {}
        for kt in range(1, 11):
            unbroken = model.step(unbroken, dt=card.dt_s)
            if kt == 5:
                save_run_restart(
                    checkpoint, unbroken, step=5,
                    time_days=5.0 * card.dt_s / 86400.0,
                    grid_type="latlon")
            elif kt >= 6:
                after[kt] = unbroken

        resumed, _, meta = load_run_restart(
            checkpoint, card.recipe.initial_state, grid_type="latlon")
        assert meta["step"] == 5
        for kt in range(6, 11):
            resumed = model.step(resumed, dt=card.dt_s)
            want_leaves = jax.tree_util.tree_leaves(after[kt])
            got_leaves = jax.tree_util.tree_leaves(resumed)
            assert len(got_leaves) == len(want_leaves)
            for got, want in zip(got_leaves, want_leaves):
                np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    finally:
        set_policy(old_policy)


def test_every_stage_barotropic_operand_routes_through_the_time_level_selector():
    """The three RK3 stage call sites must go through the fail-closed selector.

    The selector's own unit test above pins stprk3.F90:186,195,197,200,207.  It
    cannot see a call site that bypasses the selector and hands stage 1 the
    same-step external-mode target directly, which is exactly the defect the
    round-24 review asked to be guarded.  This walks the AST of the routine
    that RUNS -- ``LatLonCGridOceanModel._nemo_ws_rk3_step`` and its nested
    helpers -- and requires every ``barotropic_velocity=`` keyword on the
    identity path to be a ``_nemo_ws_stage_barotropic_velocity`` call whose
    first argument is the literal stage number, with 1, 2 and 3 each present
    exactly once.
    """
    import inspect
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as module

    source = Path(inspect.getsourcefile(module)).read_text()
    tree = ast.parse(source)
    stages = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "barotropic_velocity":
                continue
            # Unwrap the `None if <legacy arm> else <selector>` guard.  The
            # BODY branch is walked too: a bypass placed there -- e.g.
            # `(target_u, target_v) if <hook> else _nemo_ws_stage_...(...)` --
            # is live whenever that hook is set, and unwrapping only `.orelse`
            # would never see it.
            value = keyword.value
            if isinstance(value, ast.IfExp):
                assert (isinstance(value.body, ast.Constant)
                        and value.body.value is None), (
                    "the guarded arm must disable the operand, not supply a "
                    f"second one: {ast.dump(value.body)}")
                value = value.orelse
            assert isinstance(value, ast.Call), ast.dump(keyword.value)
            assert isinstance(value.func, ast.Name)
            assert value.func.id == "_nemo_ws_stage_barotropic_velocity", (
                f"stage operand bypasses the selector: {ast.dump(value.func)}")
            first = value.args[0]
            assert isinstance(first, ast.Constant), ast.dump(first)
            # Checking only the stage number is not enough: the call
            # `_nemo_ws_stage_barotropic_velocity(1, (target_u, target_v),
            # (target_u, target_v))` routes through the selector, carries the
            # literal stage, and still hands stage 1 the SAME-STEP target the
            # selector exists to keep out.  Require the Kbb operand to read
            # the carried pair and to differ from the Nnn operand.
            assert len(value.args) == 3, ast.dump(value)
            kbb_src, nnn_src = ast.dump(value.args[1]), ast.dump(value.args[2])
            assert kbb_src != nnn_src, (
                "stage operand passes the same expression as both the "
                f"last-step and this-step pair: {kbb_src}")
            assert "uu_b" in kbb_src and "vv_b" in kbb_src, (
                f"the last-step operand does not read the carried pair: {kbb_src}")
            stages.append(first.value)
    assert sorted(stages) == [1, 2, 3], stages
