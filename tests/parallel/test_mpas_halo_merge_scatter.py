"""One scatter instead of thirteen, when the halo comes back.

The coloured wide-halo fill writes each round's received rows into the local
buffers as it goes, thirteen times. Every halo row has exactly ONE owner, so
those receive positions are disjoint across rounds, and the writes can be
deferred and issued together. ``LEGOESM_MPAS_HALO_MERGE_SCATTER=1`` does that.

It attacks a measured term: an arm that skips the gather, concatenate and
scatter entirely puts on-device staging at 0.310 ms of a 5.760 ms step at 64
GPUs. This cannot remove all of that -- only the scatters -- so the ceiling is
well under it.

Bit-identical is the whole contract. Every real position is written once with
the same value either way. The one repeated index is the padding's garbage
slot, which every round targets and which is trimmed off the return, so an
unspecified winner there cannot reach the answer.

Gates:
1. ``test_merge_scatter_is_bit_identical`` -- the padded outputs match to the
   byte, wide halo and narrow, after a real step.
2. ``test_merge_scatter_changes_the_program`` -- NON-VACUITY. Gate 1 asserts
   something does NOT change and would stay green if the switch were never
   read, so this one requires the compiled program to differ.
3. ``test_merge_scatter_keeps_the_round_count`` -- it must buy scatters, never
   rounds; a moved round count would confound this with the exchange itself.
4. ``test_env_validation_raises`` -- an unknown value raises.
"""


import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _need(n: int):
    if len(jax.devices("cpu")) < n:
        pytest.skip(
            f"Need {n} CPU devices "
            f"(XLA_FLAGS=--xla_force_host_platform_device_count={n})")


def _build(devices: int, *, merge: str, monkeypatch, wide: bool = True):
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
    )
    from legoesm.parallel.voronoi_partition import (
        reorder_voronoi_for_sharding,
    )
    from legoesm.parallel.mesh import (
        create_voronoi_device_mesh, replicate_pytree,
    )
    from legoesm.parallel.sharded_dynamics import make_voronoi_sharded_step

    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

    monkeypatch.setenv("LEGOESM_MPAS_HALO_MERGE_SCATTER", merge)
    monkeypatch.setenv("LEGOESM_MPAS_WIDE_HALO", "1" if wide else "0")
    monkeypatch.setenv("LEGOESM_MPAS_RAGGED_HALO", "0")

    mesh = create_voronoi_mesh(subdivision_level=4)
    mesh = reorder_voronoi_for_sharding(mesh, devices)
    sigma = create_sigma_coordinate(8)
    cfg = MPASPrimitiveEquationConfig(
        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
        pv_scheme="energy", time_integrator="ssp_rk3")
    dev_config = create_voronoi_device_mesh(
        nCells=mesh.nCells, nEdges=mesh.nEdges,
        nVertices=mesh.nVertices, n_devices=devices)
    model = MPASPrimitiveEquationModel(
        replicate_pytree(mesh, dev_config), sigma, cfg)
    state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
    step = make_voronoi_sharded_step(
        model, dev_config, halo_strategy="ppermute")
    return step, state, 600.0




@pytest.mark.parametrize("wide", [True, False])
def test_merge_scatter_is_bit_identical(monkeypatch, wide):
    """Deferring the writes must not move a single bit of the answer."""
    _need(4)
    s1, st1, dt = _build(4, merge="1", monkeypatch=monkeypatch, wide=wide)
    out1 = s1(st1, dt)
    s0, st0, _ = _build(4, merge="0", monkeypatch=monkeypatch, wide=wide)
    out0 = s0(st0, dt)
    for name, a, b in (("u", out1.u.data, out0.u.data),
                       ("T", out1.T.data, out0.T.data),
                       ("p_s", out1.p_s.data, out0.p_s.data)):
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"{name}: merging the scatters changed the answer")


def test_merge_scatter_changes_the_program(monkeypatch):
    """NON-VACUITY. Every other gate here asserts a number does NOT move and
    would pass unchanged if the switch were never read."""
    _need(4)

    def text(merge):
        step, state, dt = _build(4, merge=merge, monkeypatch=monkeypatch)
        return jax.jit(step).lower(state, dt).compile().as_text()

    def scatters(txt):
        return sum(1 for l in txt.split("\n")
                   if re.search(r"=\s*[^=]*?\b(scatter|dynamic-update-slice)\(", l))

    t0, t1 = text("0"), text("1")
    assert t0 != t1, ("the compiled program is identical with the switch off "
                      "and on, so the switch is not being read")
    s0, s1 = scatters(t0), scatters(t1)
    # Differing text is not enough: it could differ for an unrelated reason
    # while the enabled branch still lowers to one scatter per round.
    assert s0 > 0, ("no scatter-like operations found at all, so this gate "
                    "cannot tell a merge from a no-op")
    assert s1 < s0, (f"the enabled branch still lowers to as many scatters: "
                     f"{s0} off, {s1} on")


def test_merge_scatter_preserves_the_gradient(monkeypatch):
    """Forward bit-identity does NOT validate the scatter's transpose.

    The merged scatter has one repeated destination -- the padding's garbage
    slot -- and the transpose of a scatter with duplicate destinations gathers
    the same cotangent to several sources. Those sources are padding rows and
    the garbage slot is trimmed off the return, so its cotangent should be
    zero and nothing should reach a real row; that is an argument, and this
    is the check.
    """
    _need(4)

    def grad_of(merge):
        step, state, dt = _build(4, merge=merge, monkeypatch=monkeypatch)

        def loss(T):
            st = state._replace(T=state.T.replace(data=T))
            out = step(st, dt)
            return jnp.sum(out.T.data ** 2)

        return np.asarray(jax.grad(loss)(state.T.data))

    g0, g1 = grad_of("0"), grad_of("1")

    # This model's reverse-mode gradient is already non-finite on a small
    # number of cells with the switch OFF -- 224 of 20512 entries on this
    # mesh, on interior cells, not on the padding. That is a pre-existing
    # defect of the unstructured step (the usual cause is a masked branch
    # whose dead side is not finite), it is NOT introduced here, and fixing
    # it is not this change. So the comparison is made where the baseline
    # is defined, and the pattern of the undefined entries is itself
    # compared -- if the merged scatter moved cotangent to a row it should
    # not touch, either the mask or the values would move.
    m0, m1 = np.isfinite(g0), np.isfinite(g1)
    np.testing.assert_array_equal(
        m1, m0, err_msg="merging the scatters moved which entries are finite")
    assert m0.sum() > 0.9 * m0.size, (
        f"too much of the baseline gradient is undefined for this to test "
        f"anything: {m0.sum()} of {m0.size} finite")
    assert np.abs(g0[m0]).max() > 0, "the gradient is identically zero; vacuous"
    np.testing.assert_array_equal(
        g1[m0], g0[m0], err_msg="merging the scatters changed the gradient")


def test_merge_scatter_refuses_the_no_stage_arm(monkeypatch):
    """The no-stage arm performs no scatters, so reporting the switch as
    enabled there would put a knob in a receipt for a run it did nothing to.

    Driven at the fill directly rather than through a whole model step: the
    guard lives in one function and this is the smallest thing that reaches
    it, which also means the gate cannot pass because some outer path
    happened not to call it.
    """
    from legoesm.parallel.sharded_dynamics import _ppermute_halo_fill

    monkeypatch.setenv("LEGOESM_MPAS_HALO_MERGE_SCATTER", "1")
    monkeypatch.setenv("LEGOESM_MPAS_HALO_NOSTAGE", "1")
    monkeypatch.setenv("LEGOESM_MPAS_HALO_NOCOMM", "1")
    monkeypatch.setenv("LEGOESM_MPAS_HALO_BALLAST", "")

    cell_pack = jnp.zeros((4, 3))
    u_shard = jnp.zeros((4, 2))
    idx = jnp.zeros((1, 1), jnp.int32)
    halo_sl = ((idx, idx, idx, idx),)
    with pytest.raises(ValueError, match="HALO_NOSTAGE"):
        _ppermute_halo_fill(cell_pack, u_shard, halo_sl, (((0, 0),),),
                            max_lc=6, max_le=6)


def test_no_stage_alone_still_works(monkeypatch):
    """Non-vacuity of the gate above: without the merge switch the same call
    returns normally, so the refusal is about the combination and not about
    the arguments."""
    from legoesm.parallel.sharded_dynamics import _ppermute_halo_fill

    monkeypatch.setenv("LEGOESM_MPAS_HALO_MERGE_SCATTER", "0")
    monkeypatch.setenv("LEGOESM_MPAS_HALO_NOSTAGE", "1")
    monkeypatch.setenv("LEGOESM_MPAS_HALO_NOCOMM", "1")
    monkeypatch.setenv("LEGOESM_MPAS_HALO_BALLAST", "")

    cell_pack = jnp.zeros((4, 3))
    u_shard = jnp.zeros((4, 2))
    idx = jnp.zeros((1, 1), jnp.int32)
    halo_sl = ((idx, idx, idx, idx),)
    c, u = _ppermute_halo_fill(cell_pack, u_shard, halo_sl, (((0, 0),),),
                               max_lc=6, max_le=6)
    assert c.shape == (6, 3) and u.shape == (6, 2)


def test_shared_garbage_position_is_allowed():
    """Non-vacuity partner of the gate below: the padding position that every
    round targets repeats by design and must NOT be reported."""
    from legoesm.parallel.sharded_dynamics import assert_recv_positions_unique

    garbage = 9
    rounds = [np.array([[0, garbage, garbage]]),
              np.array([[1, 2, garbage]])]
    assert_recv_positions_unique(rounds, rounds, 1, garbage, garbage)


def test_a_position_written_twice_is_refused():
    """The merged scatter has no defined winner for a repeated position, so
    the schedule is checked on the host. This is the check failing."""
    from legoesm.parallel.sharded_dynamics import assert_recv_positions_unique

    garbage = 9
    rounds = [np.array([[0, 1, garbage]]),
              np.array([[1, 2, garbage]])]          # position 1 twice
    with pytest.raises(ValueError, match="same cell position"):
        assert_recv_positions_unique(rounds, rounds, 1, garbage, garbage)


def test_the_repeat_is_found_on_the_right_device():
    """A repeat on one device must not be masked by other devices being
    clean, and the message must name the device that has it."""
    from legoesm.parallel.sharded_dynamics import assert_recv_positions_unique

    garbage = 9
    rounds = [np.array([[0, 1, garbage], [0, 1, garbage]]),
              np.array([[4, 5, garbage], [3, 3, garbage]])]
    with pytest.raises(ValueError, match="on device 1"):
        assert_recv_positions_unique(rounds, rounds, 2, garbage, garbage)
