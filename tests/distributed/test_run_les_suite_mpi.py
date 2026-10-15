"""LES-suite driver under the y-slab MPI decomposition == the serial driver.

``run_les_suite.py --mpi`` builds the global initial state on every rank, slices
it into y-slabs, steps with the distributed spectral core and gathers each output
frame to rank 0. Its artifact must equal the serial artifact:

* static-Smagorinsky cases to round-off (the distributed FFT reorders sums only):
  the free-convective CBL (flux surface) and GABLS1 (prescribed-cooling MOST
  surface, Rayleigh sponge with its planar means — the sponge used local slab
  means before, which a 2-slab run exposes);
* the dynamic LASD closure (every production case) to its documented MPI
  tolerance: the dynamic coefficient amplifies the distributed FFT's ~1e-10
  reordering (``test_lasd_mpi.py``), so it is not bit-matched;
* the column budgets the run prints close and agree with the serial run;
* a checkpointed MPI run (stopped after its first step, resumed from the per-rank
  checkpoints) reproduces the uninterrupted MPI run exactly, and checkpoints from
  another rank count are refused.

Run::

    mpiexec -n 2 python -m pytest tests/distributed/test_run_les_suite_mpi.py
    mpiexec -n 4 python -m pytest tests/distributed/test_run_les_suite_mpi.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

MPI = pytest.importorskip("mpi4py.MPI")
COMM = MPI.COMM_WORLD
RANK, NPROC = COMM.Get_rank(), COMM.Get_size()

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "run"))
import run_les_suite  # noqa: E402
from legoesm.atmosphere.les_suite.bridge import load_artifact  # noqa: E402

pytestmark = pytest.mark.skipif(NPROC < 2, reason="needs >= 2 MPI ranks")

# (case, sgs, extra args, tolerance relative to each array's max): short smoke
# windows (>= 2 steps, so --wall-limit-s 0 stops before the end), 3 frames.
_CASES = (
    ("cbl_nieuwstadt", "smagorinsky", ["--hours", f"{8 / 3600}", "--dt", "1.0"], 1e-9),
    ("sbl_gabls1", "smagorinsky", ["--hours", f"{1.0 / 3600}"], 1e-9),
    ("cbl_nieuwstadt", "lasd", ["--hours", f"{8 / 3600}", "--dt", "1.0"], 1e-3),
)
_BUDGET = re.compile(r"budget (\w+): d\(column\)=(\S+) sources=(\S+) sponge=(\S+) "
                     r"residual=(\S+) column=(\S+)")


def _shared_dir(tmp_path_factory, name):
    d = tmp_path_factory.mktemp(name) if RANK == 0 else None
    return Path(COMM.bcast(str(d) if d is not None else None, root=0))


def _run(case, sgs, extra, out, *more):
    argv = ["--case", case, "--sgs", sgs, "--smoke", "--frames", "3",
            "--output", str(out), *extra, *more]
    return run_les_suite.main(argv)


def _file(out, case, sgs):
    (f,) = sorted(Path(out).glob(f"{case}__{sgs}*.npz"))
    return f


def _artifact(out, case, sgs):
    with np.load(_file(out, case, sgs), allow_pickle=True) as z:
        return {k: z[k] for k in z.files if k != "__meta__"}


def _budgets(text):
    return {m.group(1): [float(x) for x in m.groups()[1:]]
            for m in _BUDGET.finditer(text)}


def _assert_same(a, b, rtol, label):
    assert set(a) == set(b), (label, sorted(set(a) ^ set(b)))
    num = [k for k in a if np.asarray(a[k]).dtype.kind == "f"]
    assert num, "artifact has no numeric arrays"
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.shape == y.shape, (label, k, x.shape, y.shape)
        if k not in num:
            assert np.array_equal(x, y), (label, k)
            continue
        x, y = x.astype(np.float64), y.astype(np.float64)
        # every physical field must be finite on both sides (no vacuous NaN match)
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)), (label, k)
        scale = max(float(np.max(np.abs(x), initial=0.0)), 1e-30)
        np.testing.assert_allclose(y, x, rtol=0.0, atol=rtol * scale,
                                   err_msg=f"{label}: {k}")


@pytest.mark.parametrize("case,sgs,extra,tol", _CASES)
def test_mpi_artifact_matches_serial(tmp_path_factory, capsys, case, sgs, extra, tol):
    ser = _shared_dir(tmp_path_factory, "ser")
    par = _shared_dir(tmp_path_factory, "par")
    out_ser, err = "", None
    if RANK == 0:   # the serial reference; its failure must fail EVERY rank
        try:
            rc = _run(case, sgs, extra, ser)
            err = None if rc == 0 else f"serial run returned {rc}"
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            err = f"serial run raised {type(exc).__name__}: {exc}"
        out_ser = capsys.readouterr().out
    err = COMM.bcast(err, root=0)
    assert err is None, err
    assert _run(case, sgs, extra, par, "--mpi") == 0
    if RANK != 0:
        return
    out_par = capsys.readouterr().out
    label = f"{case}/{sgs} np={NPROC}"
    _assert_same(_artifact(ser, case, sgs), _artifact(par, case, sgs), tol, label)
    # provenance: the rank count is recorded, the configuration record agrees; the
    # MOST solver residual is a computed diagnostic, compared numerically
    sa = load_artifact(_file(ser, case, sgs)).les_setup
    sb = load_artifact(_file(par, case, sgs)).les_setup
    assert (sa["mpi_ranks"], sb["mpi_ranks"]) == (1, NPROC)
    diag = ("mpi_ranks", "most_residual_max")
    assert {k: v for k, v in sa.items() if k not in diag} == {
        k: v for k, v in sb.items() if k not in diag}
    if sa["most_residual_max"] is not None:
        np.testing.assert_allclose(sb["most_residual_max"], sa["most_residual_max"],
                                   rtol=1e-3, atol=1e-12)
    # RULE 0: the column budgets the MPI run reports close to round-off of the
    # column content, and equal the serial run's (global, not slab, means)
    bs, bp = _budgets(out_ser), _budgets(out_par)
    assert set(bp) == {"theta", "u", "v"} == set(bs), (bs, bp)
    for name, (dc, src, sp, resid, col) in bp.items():
        assert abs(resid) <= 1e-9 * max(abs(col), abs(dc), 1.0), (label, name, resid)
        # scaled by the budget TERMS (not the column content, ~1e3 x larger), plus
        # a round-off floor: a column with ~zero net content (u in a CBL with no
        # mean wind) still has O(1e-13) reordering noise in its terms
        ref = bs[name][:3]
        np.testing.assert_allclose(
            [dc, src, sp], ref, rtol=0.0,
            atol=tol * max(max(abs(x) for x in ref), 1e-30)
            + 1e-12 * max(abs(col), 1.0),
            err_msg=f"{label}: budget {name}")


def test_sponge_step_matches_serial():
    """The driver's sponge step (relaxation toward the PLANAR mean + its column
    budget entries) on a random state with structure at the sponge levels: a
    slab-local mean would differ from the serial one at once."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl

    ny, nx, nz = 16, 16, 8
    cfg = sl.SpectralLESConfig(nx=nx, ny=ny, nz=nz, Lx=400.0, Ly=400.0, Lz=200.0,
                               dealias=False, smagorinsky_dynamic=False,
                               sgs_model="smagorinsky", buoyancy=True,
                               theta_ref0=265.0, time_scheme="rk3", nu_floor=0.01)
    rng = np.random.default_rng(7)
    u = jnp.asarray(8.0 + rng.standard_normal((ny, nx, nz)))
    v = jnp.asarray(rng.standard_normal((ny, nx, nz)))
    w = np.zeros((ny, nx, nz + 1))
    w[..., 1:nz] = 0.1 * rng.standard_normal((ny, nx, nz - 1))
    th = jnp.asarray(265.0 + rng.standard_normal((ny, nx, nz)))
    g_ser = sl.make_grid(cfg)
    u, v, w = sl.project(u, v, jnp.asarray(w), dt=0.1, g=g_ser)
    z = jnp.zeros_like(u)
    st = sl.SpectralLESState(u=u, v=v, w=w, rhs_u_prev=z, rhs_v_prev=z,
                             rhs_w_prev=jnp.zeros_like(w), theta=th,
                             rhs_theta_prev=z)
    args = ((8.0, 0.0), 1.39e-4, -0.005, jnp.float64)
    step_ser = run_les_suite._make_step(g_ser, *args, sponge=(100.0, 50.0))
    s_ser, _, b_ser = step_ser(st, jnp.asarray(0.1))
    g_dist, st_loc = run_les_suite._distribute(g_ser, st, COMM, jnp.float64)
    step_dist = run_les_suite._make_step(g_dist, *args, sponge=(100.0, 50.0))
    s_dist, _, b_dist = step_dist(st_loc, jnp.asarray(0.1))
    for f in ("u", "v", "w", "theta"):
        full = np.concatenate(COMM.allgather(np.asarray(getattr(s_dist, f))), axis=0)
        ref = np.asarray(getattr(s_ser, f))
        np.testing.assert_allclose(full, ref, rtol=0.0,
                                   atol=1e-9 * float(np.max(np.abs(ref))), err_msg=f)
    # budget entries are DIFFERENCES of column integrals (~265 K x 200 m): their
    # reordered-sum round-off scales with the column content, not the difference
    col = float(np.max(np.abs(ref))) * cfg.Lz
    np.testing.assert_allclose(np.asarray(b_dist), np.asarray(b_ser), rtol=1e-9,
                               atol=1e-12 * col, err_msg="budget")


def test_mpi_checkpoint_resume_is_exact(tmp_path_factory):
    case, sgs, extra, _tol = _CASES[1]
    full = _shared_dir(tmp_path_factory, "full")
    part = _shared_dir(tmp_path_factory, "part")
    ck = part / "run.ckpt"
    assert _run(case, sgs, extra, full, "--mpi") == 0
    rc = _run(case, sgs, extra, part, "--mpi", "--checkpoint", str(ck),
              "--wall-limit-s", "0")
    assert rc == run_les_suite.EXIT_CHECKPOINTED
    assert (part / f"run.ckpt.r{RANK}of{NPROC}").exists()
    assert _run(case, sgs, extra, part, "--mpi", "--checkpoint", str(ck)) == 0
    assert not (part / f"run.ckpt.r{RANK}of{NPROC}").exists()
    COMM.Barrier()
    if RANK == 0:
        _assert_same(_artifact(full, case, sgs), _artifact(part, case, sgs), 0.0,
                     "resume")


def test_mpi_refuses_checkpoints_from_another_rank_count(tmp_path_factory):
    """A checkpoint written over a different rank count (or serially) would be
    ignored and the run would silently restart from t=0 — it is refused."""
    case, sgs, extra, _tol = _CASES[1]
    d = _shared_dir(tmp_path_factory, "stray")
    ck = d / "run.ckpt"
    if RANK == 0:
        (d / f"run.ckpt.r0of{NPROC + 1}").write_bytes(b"x")
    COMM.Barrier()
    with pytest.raises(SystemExit, match="different rank count"):
        _run(case, sgs, extra, d, "--mpi", "--checkpoint", str(ck))
