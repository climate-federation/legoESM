"""Direct tests of the round-10 SWG online-learning probe scripts' pure parts.

Covers the exact range projector of residual_decomposition.py (sparse
operator equals swe_rotated_potential_forcing; in-range fields return with
zero remainder; cross*grad(p) fields are exactly orthogonal) and the set
semantics of closure_scaling.lambdas_where_closure_exceeds_abs_drag, and the
T-point work density and Pearson helpers of subgrid_snapshots.py.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

# The probe modules refuse to guess a run directory (SWGONL_RUN / SWGONL_DIAG must
# be set when they run as scripts).  These tests exercise only their pure parts, so
# they point the modules at the first pilot's run and a scratch output directory
# unless the environment already names them, and skip where that run is absent.
_PILOT_RUN = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round9_pilot"
                  "/pilot/swgonl_pilot_template_ring_excluded/seed20261013")
if "SWGONL_RUN" not in os.environ:
    if not (_PILOT_RUN / "run_manifest.json").is_file():
        pytest.skip("the first pilot's run directory is not on this machine", allow_module_level=True)
    os.environ["SWGONL_RUN"] = str(_PILOT_RUN)
if "SWGONL_DIAG" not in os.environ:
    import tempfile
    os.environ["SWGONL_DIAG"] = tempfile.mkdtemp(prefix="swgonl-probe-tests-")

_ROOT = Path(__file__).resolve().parents[2]
_DIR = _ROOT / "scripts/validate/swgonl_round10"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _basin(ny=6, nx=7, pad=1):
    """Rectangular wet basin inside a padded T grid, native masks."""
    tmask = np.zeros((ny + 2 * pad, nx + 2 * pad), dtype=bool)
    tmask[pad:pad + ny, pad:pad + nx] = True
    east = np.zeros_like(tmask); east[:, :-1] = tmask[:, 1:]
    north = np.zeros_like(tmask); north[:-1, :] = tmask[1:, :]
    umask = tmask & east
    vmask = tmask & north
    vtx = np.zeros_like(tmask)
    vtx[pad:pad + ny - 1, pad:pad + nx - 1] = True
    return tmask, umask, vmask, vtx


def test_sparse_operator_matches_swe_rotated_potential_forcing():
    rd = _load("residual_decomposition")
    from legoesm.ocean.dynamics.swe_rk3 import swe_rotated_potential_forcing
    rng = np.random.default_rng(0)
    tmask, umask, vmask, vtx = _basin()
    e2u = 1.0 + rng.random(tmask.shape)
    e1v = 1.0 + rng.random(tmask.shape)
    e3u = 2.0 + rng.random(tmask.shape)
    e3v = 2.0 + rng.random(tmask.shape)
    phi = np.where(vtx, rng.standard_normal(tmask.shape), 0.0)
    A, _ = rd.rotated_gradient_matrix(e2u * e3u, e1v * e3v, umask, vmask, vtx)
    fu, fv = (np.asarray(x) for x in swe_rotated_potential_forcing(
        phi, e2u, e1v, e3u, e3v, umask, vmask, vtx))
    expected = np.concatenate([fu[umask], fv[vmask]])
    got = A @ phi[vtx]
    assert np.abs(got - expected).max() <= 1e-15 * np.abs(expected).max()
    # Counting: faces = (wet T - 1) + interior F, the exact Hodge split.
    assert A.shape == (int(umask.sum() + vmask.sum()), int(vtx.sum()))
    assert A.shape[0] == int(tmask.sum()) - 1 + A.shape[1]


def test_projection_returns_in_range_field_and_kills_cross_grad_p():
    rd = _load("residual_decomposition")
    rng = np.random.default_rng(1)
    tmask, umask, vmask, vtx = _basin()
    cross_u = 1.0 + rng.random(tmask.shape)
    cross_v = 1.0 + rng.random(tmask.shape)
    A, _ = rd.rotated_gradient_matrix(cross_u, cross_v, umask, vmask, vtx)
    proj = rd.RangeProjector(A)
    in_range = A @ rng.standard_normal(A.shape[1])
    rep, rem, _, res = proj(in_range)
    assert np.linalg.norm(rem) <= 1e-13 * np.linalg.norm(in_range)
    assert res <= 1e-12
    p = np.where(tmask, rng.standard_normal(tmask.shape), 0.0)
    g_u = cross_u * (np.roll(p, -1, axis=1) - p) * umask
    g_v = cross_v * (np.roll(p, -1, axis=0) - p) * vmask
    g = np.concatenate([g_u[umask], g_v[vmask]])
    rep_g, rem_g, _, _ = proj(g)
    assert np.linalg.norm(rep_g) <= 1e-13 * np.linalg.norm(g)
    # A mixed field splits into exactly these two parts (Pythagoras).
    mixed = in_range + g
    rep_m, rem_m, _, _ = proj(mixed)
    assert np.linalg.norm(rep_m - in_range) <= 1e-12 * np.linalg.norm(mixed)
    assert np.linalg.norm(rem_m - g) <= 1e-12 * np.linalg.norm(mixed)


def test_closure_exceeds_abs_drag_is_a_set_not_a_first_hit():
    cs = _load("closure_scaling")
    rows = [{"lambda": 0.25, "closure_work": 2.0, "drag_work": -1.0},
            {"lambda": 0.5, "closure_work": 2.0, "drag_work": -1.5},
            {"lambda": 1.0, "closure_work": 2.0, "drag_work": -3.0},
            {"lambda": 1.5, "closure_work": 5.0, "drag_work": -4.0}]
    assert cs.lambdas_where_closure_exceeds_abs_drag(rows) == [0.25, 0.5, 1.5]
    assert cs.lambdas_where_closure_exceeds_abs_drag(rows[2:3]) == []


def test_work_density_at_t_sums_to_the_face_work_and_pearson_is_the_correlation():
    ss = _load("subgrid_snapshots")
    rng = np.random.default_rng(1)
    tmask, umask, vmask, _ = _basin()
    u, v, au, av = (rng.standard_normal(tmask.shape) for _ in range(4))
    density = ss.work_density_t(u, v, au, av, umask, vmask)
    face_sum = np.sum(u * au * umask) + np.sum(v * av * vmask)
    assert abs(density.sum() - face_sum) <= 1e-14 * abs(face_sum)
    assert np.all(density[~tmask] == 0.0)
    # A face's work lands half on each of its two adjoining T cells: a single
    # nonzero U face splits exactly.
    one_u = np.zeros_like(u)
    j, i = np.argwhere(umask)[3]
    one_u[j, i] = 2.0
    split = ss.work_density_t(one_u, v, np.ones_like(au), np.zeros_like(av), umask, vmask)
    assert split[j, i] == 1.0 and split[j, i + 1] == 1.0 and split.sum() == 2.0
    x = rng.standard_normal(50)
    assert abs(ss.pearson(x, 3.0 * x + 2.0) - 1.0) <= 1e-14
    assert abs(ss.pearson(x, -x) + 1.0) <= 1e-14
    assert abs(ss.pearson(x, rng.standard_normal(50)) - np.corrcoef(x, x)[0, 1]) < 1.0


def test_hourly_eq7_loss_is_the_training_loss_resolved_by_hour():
    """loss_floor.hourly_eq7_loss: its mean is deterministic_trajectory_loss,
    and each entry is local to its hour (zeroing one hour's error zeroes
    that entry only)."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_DG50_NAMELIST, build_swg_card
    from legoesm.training.swg_online import build_closure_geometry, deterministic_trajectory_loss

    lf = _load("loss_floor")
    old = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        card = build_swg_card(SWG_DG50_NAMELIST._replace(rn_domsiz_m=1.0e6))
    finally:
        set_policy(old)
    geometry = build_closure_geometry(card)
    kw = {"objective": "template", "beta_m1_s1": 2.0e-11, "basin_length_m": 1.0e6,
          "wall_ring": "weighted"}
    shape = (3, 4) + tuple(geometry.tmask.shape)   # windows, hours, y, x
    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    predicted = tuple(0.1 * jax.random.normal(k, shape, jnp.float64) for k in keys[:3])
    target = tuple(0.1 * jax.random.normal(k, shape, jnp.float64) for k in keys[3:])
    hourly = lf.hourly_eq7_loss(predicted, target, geometry, **kw)
    assert hourly.shape == shape[:2]
    total = deterministic_trajectory_loss(predicted, target, geometry, **kw)
    assert abs(float(hourly.mean()) / float(total) - 1.0) <= 1e-12
    assert float(hourly.min()) > 0.0
    matched = tuple(p.at[:, 2].set(t[:, 2]) for p, t in zip(predicted, target))
    hourly_matched = lf.hourly_eq7_loss(matched, target, geometry, **kw)
    assert float(jnp.max(jnp.abs(hourly_matched[:, 2]))) == 0.0
    others = jnp.delete(jnp.arange(4), 2)
    assert float(jnp.max(jnp.abs(hourly_matched[:, others] - hourly[:, others]))) == 0.0


def test_free_run_pretrained_ke_is_the_wet_face_mean_definition():
    """free_run_pretrained.ke_and_peak: truth_ke_reference's basin KE
    (0.5 (mean_U u^2 + mean_V v^2) over wet faces) and the wet-face peak
    speed; dry faces do not count."""
    fp = _load("free_run_pretrained")
    _, umask, vmask, _ = _basin()
    rng = np.random.default_rng(3)
    u = rng.standard_normal(umask.shape)
    v = rng.standard_normal(vmask.shape)
    u[~umask] = 50.0   # dry faces: excluded from both numbers
    v[~vmask] = 50.0
    ke, peak = fp.ke_and_peak(u, v, umask, vmask)
    assert ke == pytest.approx(0.5 * (np.mean(u[umask] ** 2) + np.mean(v[vmask] ** 2)), rel=1e-15)
    assert peak == max(np.abs(u[umask]).max(), np.abs(v[vmask]).max()) < 50.0
    assert fp.DAYS == 3 * 365 and fp.REPORT_DAYS == 30
