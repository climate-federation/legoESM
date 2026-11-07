"""Direct non-vacuity tests for the stage-3 remainder probe's transcriptions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_stage3_remainder_probe.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_testcase_stage3_remainder_probe", SCRIPT)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _row_mesh(ni=8, nz=6, dz=20.0, dx=1000.0):
    tmask = np.ones((ni, nz))
    tmask[:, -1] = 0.0                      # NEMO bottom dummy level
    umask = tmask.copy()
    umask[-1] = 0.0
    e3t_0 = np.full((ni, nz), dz)
    gdept_1d = (np.arange(nz) + 0.5) * dz
    e3w_1d = np.concatenate([[2.0 * gdept_1d[0]], np.diff(gdept_1d)])
    ht_0 = np.sum(e3t_0 * tmask, axis=-1)
    hu_0 = np.sum(e3t_0 * umask, axis=-1)
    return {
        "tmask": tmask, "umask": umask, "e3t_0": e3t_0, "e3u_0": e3t_0.copy(),
        "e3w_1d": e3w_1d, "gdept_1d": gdept_1d, "e1u": np.full(ni, dx),
        "r1_ht_0": 1.0 / ht_0, "hu_0": hu_0,
        "r1_hu_0": np.where(hu_0 > 0, 1.0 / np.maximum(hu_0, 1e-300), 0.0),
        "e1e2t": np.full(ni, dx * dx), "e1e2u": np.full(ni, dx * dx), "e2u": np.full(ni, dx),
    }


def test_up3_row_is_zero_for_a_uniform_velocity_and_moves_on_a_planted_curvature():
    mesh = _row_mesh()
    ni, nz = mesh["tmask"].shape
    u = np.full((ni, nz), 0.1) * mesh["umask"]
    Fu = mesh["e2u"][:, None] * mesh["e3u_0"] * u
    Fw = np.zeros((ni, nz))
    e3u = mesh["e3u_0"]
    adv_h, adv_v = probe.up3_row(u, Fu, Fw, e3u, e3u, mesh)
    interior = mesh["umask"].astype(bool)
    interior[0] = False
    interior[-2:] = False
    assert np.max(np.abs(adv_h[interior])) == 0.0 and np.max(np.abs(adv_v)) == 0.0
    u_bump = u.copy()
    u_bump[4, :] += 0.01
    adv_h2, _ = probe.up3_row(u_bump, Fu, Fw, e3u, e3u, mesh)
    assert np.max(np.abs(adv_h2[interior])) > 0.0
    # the two upwind selectors differ only where the transport and velocity pair signs disagree
    adv_t, _ = probe.up3_row(u_bump, Fu, Fw, e3u, e3u, mesh, selector="transport")
    assert np.array_equal(adv_t, adv_h2)
    adv_t2, _ = probe.up3_row(u_bump, -Fu, Fw, e3u, e3u, mesh, selector="transport")
    assert not np.array_equal(adv_t2, probe.up3_row(u_bump, -Fu, Fw, e3u, e3u, mesh)[0])


def test_bc_row_removes_the_e3u0_mean_exactly_and_returns_it():
    mesh = _row_mesh()
    ni, nz = mesh["tmask"].shape
    act = mesh["umask"].astype(bool)
    field = np.linspace(-1.0, 1.0, nz)[None, :] + np.arange(ni)[:, None]
    bc, mean = probe.bc_row(field, act, mesh)
    assert np.allclose(np.sum(bc * mesh["e3u_0"] * act, axis=-1), 0.0, atol=1e-12)
    expected = np.sum(field * mesh["e3u_0"] * act, axis=-1) / np.maximum(np.sum(mesh["e3u_0"] * act, axis=-1), 1e-300)
    assert np.allclose(mean[:-1], expected[:-1], atol=1e-12)
    assert np.all(bc[~act] == 0.0)


def test_wzv_row_telescopes_the_column_transport_to_the_surface():
    mesh = _row_mesh()
    ni, nz = mesh["tmask"].shape
    Fu = np.zeros((ni, nz))
    Fu[3, :nz - 1] = 1.0e3                       # one face, uniform transport
    dt = 10.0
    Fw = probe.wzv_row(Fu, np.zeros(ni), np.zeros(ni), dt, mesh)
    assert np.allclose(Fw[:, -1], 0.0)
    # surface transport = minus the column divergence (no ssh change supplied)
    assert np.isclose(Fw[4, 0], -np.sum(Fu[4] - Fu[3]))
    assert np.isclose(Fw[3, 0], -np.sum(Fu[3] - Fu[2]))
    assert Fw[5, 0] == 0.0


def test_zdf_apply_row_is_identity_without_viscosity_and_conserves_the_column():
    mesh = _row_mesh()
    ni, nz = mesh["tmask"].shape
    u = (np.arange(nz)[None, :] * 0.01 + 0.1) * mesh["umask"]
    ssh = np.zeros(ni)
    assert np.array_equal(probe.zdf_apply_row(u, ssh, ssh, mesh, 10.0, 0.0), u)
    mu = probe.zdf_apply_row(u, ssh, ssh, mesh, 10.0, 1.0e-4)
    assert not np.array_equal(mu, u)
    # a flux-form operator with no-flux boundaries keeps the e3u_0-weighted column sum
    assert np.allclose(np.sum(mu * mesh["e3u_0"], axis=-1), np.sum(u * mesh["e3u_0"], axis=-1), rtol=1e-12)


def test_fit_helpers_recover_a_planted_line_and_a_planted_scaling():
    gdept = (np.arange(6) + 0.5) * 20.0
    active = np.ones(6, dtype=bool)
    active[-1] = False
    fit = probe.fit_linear_in_depth(2.0e-10 + 3.0e-12 * gdept, gdept, active)
    assert np.isclose(fit["b"], 3.0e-12) and fit["r2"] > 1.0 - 1e-12
    pattern = np.random.default_rng(0).normal(size=(4, 6))
    act = np.ones((4, 6), dtype=bool)
    f = probe.fit_pattern(2.5 * pattern, pattern, act)
    assert np.isclose(f["slope"], 2.5) and np.isclose(f["corr"], 1.0) and f["max_abs_residual"] < 1e-12
