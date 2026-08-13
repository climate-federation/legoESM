"""The halo-payload ballast knob used to measure the exchange's
bandwidth term.

``LEGOESM_MPAS_HALO_BALLAST=N`` repeats each packed halo message N times
on the wire and discards the copies on receipt, so the collective moves
N x the bytes with the SAME round count, the SAME schedule and the SAME
arithmetic. The step's change is then the bandwidth term of the
exchange, measured by moving one variable rather than fitted out of
three separate A/B receipts against a compute time taken from an older
trace.

Two things must hold or the measurement is worthless: the answer must
not move (it is a wire-payload knob, not a physics knob), and the wire
payload must actually grow (a ballast that the compiler folds away
measures nothing — the campaign's own rule about a control that
perturbs a zero).

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""

import re

import jax
import numpy as np
import pytest


def _need(n: int):
    if len(jax.devices("cpu")) < n:
        pytest.skip(
            f"Need {n} CPU devices "
            f"(XLA_FLAGS=--xla_force_host_platform_device_count={n})")


def _build(devices: int, *, ballast: str, monkeypatch, wide: bool = False):
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

    monkeypatch.setenv("LEGOESM_MPAS_HALO_BALLAST", ballast)
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


@pytest.mark.parametrize("wide", [False, True])
def test_ballast_is_bit_identical(monkeypatch, wide):
    """A wire-payload knob must not move a single bit of the answer."""
    _need(4)
    s2, st2, dt = _build(4, ballast="2", monkeypatch=monkeypatch, wide=wide)
    out2 = s2(st2, dt)
    s1, st1, _ = _build(4, ballast="1", monkeypatch=monkeypatch, wide=wide)
    out1 = s1(st1, dt)
    for name, a, b in (("u", out2.u.data, out1.u.data),
                       ("T", out2.T.data, out1.T.data),
                       ("p_s", out2.p_s.data, out1.p_s.data)):
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"{name}: ballast changed the answer")


def test_ballast_actually_grows_the_wire_payload(monkeypatch):
    """A ballast the compiler folds away measures nothing. Compare the
    collective-permute operand widths in the compiled program."""
    _need(4)

    def widths(ballast):
        step, state, dt = _build(
            4, ballast=ballast, monkeypatch=monkeypatch)
        txt = jax.jit(step).lower(state, dt).compile().as_text()
        # Widths off the collective-permute LINES themselves; the exact
        # operand spelling differs between the CPU and GPU backends, so
        # match the line first and the shape second.
        out = []
        for line in txt.split("\n"):
            if "collective-permute" in line and "-done" not in line:
                out += [int(w) for w in re.findall(r"f32\[(\d+)\]", line)]
        return sorted(out)

    w1, w2 = widths("1"), widths("2")
    assert w1 and w2, "no collective-permute operands found to compare"
    assert sum(w2) >= 2 * sum(w1) - len(w1), (
        f"ballast=2 did not double the bytes on the wire: {w1} -> {w2}")


def test_ballast_round_count_unchanged(monkeypatch):
    """Ballast buys bytes, never rounds — if the round count moved, the
    A/B would confound bandwidth with latency."""
    _need(4)

    def n_cp(ballast):
        step, state, dt = _build(
            4, ballast=ballast, monkeypatch=monkeypatch)
        txt = jax.jit(step).lower(state, dt).compile().as_text()
        return len(re.findall(r"collective-permute(?!-done)", txt))

    assert n_cp("2") == n_cp("1") > 0


def test_env_validation_raises():
    from legoesm.parallel.sharded_dynamics import _resolve_halo_ballast

    assert _resolve_halo_ballast("") == 1
    assert _resolve_halo_ballast("1") == 1
    assert _resolve_halo_ballast("4") == 4
    for bad in ("0", "yes", "+2", " 2 ", "02", "2_0", "9", "-1"):
        with pytest.raises(ValueError, match="HALO_BALLAST"):
            _resolve_halo_ballast(bad)
