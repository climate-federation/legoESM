"""x-y PENCIL distributed spectral LES == serial (``parallel/distributed_fft``
pencil functions + ``make_spectral_les_pencil_layout``).

Every test runs each (Py, Px) rank mesh that factors the communicator size, with
a grid chosen so every padding path is live: nkx=13 split over Py∈{2,4} (kx
padding), ny=16 split over Px=3 in the projection (ky padding) and nz=7 / nz±1
split over Px∈{2,3,4} (z padding). Tolerances are the slab tests' (same
operators, same reduction-order class): operators 1e-10, dealiased step 1e-9,
dynamic-LASD step 1e-4 (``test_lasd_mpi.py``), gradients 1e-5.

Run (each size exercises different meshes)::

    for n in 1 2 3 4 6 8; do
        mpiexec -n $n python -m pytest tests/distributed/test_spectral_les_pencil_mpi.py
    done
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402
from legoesm.atmosphere.dynamics.les.spectral_les_plane import (  # noqa: E402
    SpectralLESConfig, SpectralLESState, make_grid,
    make_spectral_les_pencil_layout, step)
from legoesm.atmosphere.physics.turbulence.lasd_core import (  # noqa: E402
    imfilter_box3, lasd_cs2, spectral_test_filter)
from legoesm.parallel.distributed_fft import (  # noqa: E402
    chunk_slice, layout_irfft2, layout_rfft2, pencil_spec_to_zwhole,
    pencil_zwhole_to_spec, split_concat_transpose)
from legoesm.parallel.reductions import global_sum_mpi  # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NPROC = COMM.Get_size()

NY, NX, NZ = 16, 24, 7
_MESHES = {1: [(1, 1)], 2: [(2, 1), (1, 2)], 3: [(1, 3)],
           4: [(2, 2), (4, 1), (1, 4)], 6: [(2, 3)], 8: [(4, 2), (2, 4)]}
MESHES = _MESHES.get(NPROC, [])
pytestmark = pytest.mark.skipif(not MESHES, reason=f"no mesh for {NPROC} ranks")


def _layout(mesh):
    return make_spectral_les_pencil_layout(COMM, mesh[0], mesh[1], NY, NX)


def _block(f, mesh):
    """This rank's physical (ny/Py, nx/Px, ...) block of a global field."""
    py, px = mesh
    ry, rx = divmod(RANK, px)
    nyl, nxl = f.shape[0] // py, f.shape[1] // px
    return f[ry * nyl:(ry + 1) * nyl, rx * nxl:(rx + 1) * nxl]


def _gather(blk, mesh):
    """Assemble the global field from every rank's block (test-only I/O)."""
    py, px = mesh
    parts = COMM.allgather(np.asarray(blk))
    rows = [np.concatenate(parts[r * px:(r + 1) * px], axis=1) for r in range(py)]
    return np.concatenate(rows, axis=0)


def _rand(seed, shape):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


def _spec_ref(fh_serial, mesh, nzf):
    """Serial rfft2 spectrum cut to this rank's pencil spectral block: full ky,
    kx chunk ry of Py, z chunk rx of Px (both zero-padded)."""
    py, px = mesh
    ry, rx = divmod(RANK, px)
    nkx = NX // 2 + 1
    klo, khi = chunk_slice(nkx, py, ry)
    zlo, zhi = chunk_slice(nzf, px, rx)
    pad = np.zeros((NY, py * (khi - klo), px * (zhi - zlo)), complex)
    pad[:, :nkx, :nzf] = np.asarray(fh_serial)
    return pad[:, klo:khi, zlo:zhi]


# --------------------------------------------------------------------------- #
# Transforms                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dz", [-1, 0, 1])
def test_pencil_rfft2_matches_serial_and_roundtrips(dz):
    nzf = NZ + dz
    f = _rand(1, (NY, NX, nzf))
    for mesh in MESHES:
        L = _layout(mesh)
        fh = layout_rfft2(_block(f, mesh), L)
        ref = _spec_ref(jnp.fft.rfft2(f, axes=(0, 1)), mesh, nzf)
        np.testing.assert_allclose(np.asarray(fh), ref, rtol=1e-10, atol=1e-10)
        back = layout_irfft2(fh, L, nzf)
        np.testing.assert_allclose(_gather(back, mesh), np.asarray(f),
                                   rtol=1e-10, atol=1e-10)


def test_pencil_zwhole_roundtrip_and_content():
    f = _rand(2, (NY, NX, NZ))
    fh_ser = np.asarray(jnp.fft.rfft2(f, axes=(0, 1)))
    for mesh in MESHES:
        L = _layout(mesh)
        if L.n_ranks_x == 1:
            continue
        fh = layout_rfft2(_block(f, mesh), L)
        zw = pencil_spec_to_zwhole(fh, nz_out=NZ, n_ranks_x=L.n_ranks_x,
                                   comm_row=L.comm_row)
        klo, khi = chunk_slice(NX // 2 + 1, L.n_ranks_y, L.ry)
        ylo, yhi = chunk_slice(NY, L.n_ranks_x, L.rx)
        ref = np.zeros((L.n_ranks_x * (yhi - ylo), L.n_ranks_y * (khi - klo), NZ),
                       complex)
        ref[:NY, :NX // 2 + 1] = fh_ser
        np.testing.assert_allclose(np.asarray(zw), ref[ylo:yhi, klo:khi],
                                   rtol=1e-10, atol=1e-10)
        back = pencil_zwhole_to_spec(zw, ny_global=NY, n_ranks_x=L.n_ranks_x,
                                     comm_row=L.comm_row)
        np.testing.assert_array_equal(np.asarray(back), np.asarray(fh))


def test_pencil_pad_truncate_matches_serial():
    f = _rand(3, (NY, NX, NZ))
    fine_ser = sl._pad_to_fine(f)
    coarse_ser = sl._truncate_from_fine(fine_ser, NY, NX)
    cfg = SpectralLESConfig(nx=NX, ny=NY, nz=NZ, Lx=300.0, Ly=200.0, Lz=100.0)
    for mesh in MESHES:
        g = make_grid(cfg, layout=_layout(mesh))
        fine = sl._pad_to_fine(_block(f, mesh), g)
        np.testing.assert_allclose(_gather(fine, mesh), np.asarray(fine_ser),
                                   rtol=1e-10, atol=1e-10)
        coarse = sl._truncate_from_fine(fine, NY, NX, g)
        np.testing.assert_allclose(_gather(coarse, mesh), np.asarray(coarse_ser),
                                   rtol=1e-10, atol=1e-10)


# --------------------------------------------------------------------------- #
# LASD horizontally-coupled operators                                          #
# --------------------------------------------------------------------------- #
def test_pencil_lasd_operators_match_serial():
    f = _rand(4, (NY, NX, NZ))
    filt_ser, box_ser = spectral_test_filter(f, 4, 5), imfilter_box3(f)
    yy = np.linspace(0, 2 * np.pi, NY, endpoint=False)
    xx = np.linspace(0, 2 * np.pi, NX, endpoint=False)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    base = (np.sin(X) + 0.7 * np.cos(2 * Y) + 0.5 * np.sin(3 * Y + X)
            + 0.3 * np.cos(4 * Y))
    args = [jnp.asarray(base[:, :, None] + 0.1 * np.random.default_rng(s)
                        .standard_normal((NY, NX, NZ))) for s in range(10)]
    args[-1] = jnp.sqrt(args[-1] ** 2 + 1e-2)
    delta = jnp.linspace(1.0, 2.0, NZ)
    cs2_ser = lasd_cs2(*args, delta, cs_max=1.0)
    for mesh in MESHES:
        L = _layout(mesh)
        np.testing.assert_allclose(
            _gather(spectral_test_filter(_block(f, mesh), 4, 5, L), mesh),
            np.asarray(filt_ser), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            _gather(imfilter_box3(_block(f, mesh), L), mesh),
            np.asarray(box_ser), rtol=1e-10, atol=1e-10)
        cs2 = lasd_cs2(*[_block(a, mesh) for a in args], delta, cs_max=1.0,
                       layout=L)
        np.testing.assert_allclose(_gather(cs2, mesh), np.asarray(cs2_ser),
                                   rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# Full step() trajectories + conservation                                      #
# --------------------------------------------------------------------------- #
_DT, _UG, _FC, _FORCE = 2.0e-3, (1.0, 0.0), 1.0e-4, (1.0e-3, 0.0)


def _init(seed=101, theta=False):
    rng = np.random.default_rng(seed)
    u = jnp.asarray(1.0 + 0.1 * rng.standard_normal((NY, NX, NZ)))
    v = jnp.asarray(0.1 * rng.standard_normal((NY, NX, NZ)))
    w = np.zeros((NY, NX, NZ + 1))
    w[..., 1:NZ] = 0.05 * rng.standard_normal((NY, NX, NZ - 1))
    z = jnp.zeros((NY, NX, NZ))
    kw = {}
    if theta:
        kw = dict(theta=jnp.asarray(290.0 + rng.standard_normal((NY, NX, NZ))),
                  rhs_theta_prev=z)
    return SpectralLESState(u=u, v=v, w=jnp.asarray(w), rhs_u_prev=z,
                            rhs_v_prev=z, rhs_w_prev=jnp.zeros((NY, NX, NZ + 1)),
                            **kw)


def _block_state(st, mesh):
    return st._replace(**{k: _block(getattr(st, k), mesh)
                          for k in st._fields if getattr(st, k) is not None})


def _budget(st, gather=None):
    """Domain integrals: momentum (Σu, Σv, Σw), energy ½Σ|u|², scalar Σθ."""
    get = (lambda a: np.asarray(a)) if gather is None else gather
    u, v, w = get(st.u), get(st.v), get(st.w)
    out = [u.sum(), v.sum(), w.sum(), 0.5 * ((u ** 2).sum() + (v ** 2).sum()
                                             + (w ** 2).sum())]
    if st.theta is not None:
        out.append(get(st.theta).sum())
    return np.array(out)


def _budget_scale(st):
    """Σ|integrand| per budget term: the round-off scale of each domain sum."""
    u, v, w = (np.abs(np.asarray(a)) for a in (st.u, st.v, st.w))
    out = [u.sum(), v.sum(), w.sum(), 0.5 * ((u ** 2).sum() + (v ** 2).sum()
                                             + (w ** 2).sum())]
    if st.theta is not None:
        out.append(np.abs(np.asarray(st.theta)).sum())
    return np.array(out)


def _trajectory(st, g, nsteps, gather=None, sfc_theta_flux=0.0):
    states, budgets = [], [_budget(st, gather)]
    for i in range(nsteps):
        st, _ = step(st, g, _DT, _UG, _FC, first=(i == 0), force=_FORCE,
                     sfc_theta_flux=sfc_theta_flux)
        states.append(st)
        budgets.append(_budget(st, gather))
    return states, np.diff(np.array(budgets), axis=0)   # per-step changes


@pytest.mark.parametrize("time_scheme", ["rk3", "ab2"])
def test_pencil_dealiased_step_matches_serial_and_conserves(time_scheme):
    """dealias=True + θ scalar + spectral filter: 3/2 pad/truncate, projection,
    filter, planar-mean wall model all on the pencil. State to 1e-9 (slab
    tolerance); per-step changes of the domain-integrated momentum, energy and
    θ equal to serial (RULE 0: the decomposition creates/destroys nothing)."""
    cfg = SpectralLESConfig(nx=NX, ny=NY, nz=NZ, Lx=300.0, Ly=200.0, Lz=100.0,
                            dealias=True, smagorinsky_dynamic=False,
                            sgs_model="smagorinsky", spectral_filter=True,
                            buoyancy=True, time_scheme=time_scheme)
    ser, dser = _trajectory(_init(theta=True), make_grid(cfg), 3,
                            sfc_theta_flux=0.05)
    for mesh in MESHES:
        gat = lambda a, m=mesh: _gather(a, m)  # noqa: E731
        dist, ddist = _trajectory(_block_state(_init(theta=True), mesh),
                                  make_grid(cfg, layout=_layout(mesh)), 3,
                                  gather=gat, sfc_theta_flux=0.05)
        for name in ("u", "v", "w", "theta"):
            np.testing.assert_allclose(
                gat(getattr(dist[-1], name)), np.asarray(getattr(ser[-1], name)),
                rtol=1e-9, atol=1e-9, err_msg=f"{name} mismatch {mesh}")
        # Per-step budget changes agree to round-off of the domain sums
        # (1e-12 x sum|integrand|, i.e. ~1e4 ulp of a 2.7e3-term fp64 sum).
        tol = 1e-12 * _budget_scale(ser[-1])
        assert np.all(np.abs(ddist - dser) <= tol), (mesh, ddist - dser, tol)


def test_pencil_lasd_step_matches_serial():
    """Full oracle closure (3/2 de-aliasing + dynamic LASD SGS) step()
    trajectory; LASD tolerance 1e-4 as the slab test (raw C_s² is
    hypersensitive to FFT round-off, the integrated state is not)."""
    cfg = SpectralLESConfig(nx=NX, ny=NY, nz=NZ, Lx=300.0, Ly=200.0, Lz=100.0,
                            dealias=True, smagorinsky_dynamic=True,
                            spectral_filter=True, time_scheme="rk3")
    ser, _ = _trajectory(_init(), make_grid(cfg), 3)
    for mesh in MESHES:
        dist, _ = _trajectory(_block_state(_init(), mesh),
                              make_grid(cfg, layout=_layout(mesh)), 3)
        for name in ("u", "v", "w"):
            np.testing.assert_allclose(
                _gather(getattr(dist[-1], name), mesh),
                np.asarray(getattr(ser[-1], name)),
                rtol=1e-4, atol=1e-4, err_msg=f"{name} mismatch {mesh}")


# --------------------------------------------------------------------------- #
# Gradients                                                                    #
# --------------------------------------------------------------------------- #
def _fd_checks(loss, x, v, eps=1e-5, tol=1e-6):
    """Reverse-mode gradient vs a GLOBAL central finite difference
    (``jax.test_util.check_grads`` tests a per-rank identity that a distributed
    map does not satisfy): ``Σ_r <∇_r L, v_r>`` vs ``(L(x+εv)-L(x-εv))/2ε``.
    ``loss`` maps this rank's block to the replicated scalar
    ``global_sum_mpi(..., final_loss=True)``; ``v`` is this rank's block of a
    global direction. First order only: second-order (reverse-over-reverse) AD
    through the raw mpi4jax all-to-all / sendrecv has no differentiation rule
    (pre-existing, slab and pencil alike)."""
    g = jax.grad(loss)(x)
    ad1 = COMM.allreduce(float(jnp.vdot(g, v)))
    fd1 = (float(loss(x + eps * v)) - float(loss(x - eps * v))) / (2 * eps)
    assert abs(ad1 - fd1) <= tol * max(abs(fd1), 1.0), (ad1, fd1)
    return g


def _row_post_block(fg, mesh):
    """Expected content after the pencil ROW transpose (z -> x): this rank's
    y rows, ALL x, z chunk rx of Px (zero-padded)."""
    py, px = mesh
    ry, rx = divmod(RANK, px)
    nyl, nz = fg.shape[0] // py, fg.shape[2]
    lo, hi = chunk_slice(nz, px, rx)
    pad = np.zeros((fg.shape[0], fg.shape[1], px * (hi - lo)))
    pad[..., :nz] = np.asarray(fg)
    return pad[ry * nyl:(ry + 1) * nyl, :, lo:hi]


def test_pencil_transpose_and_fft_gradients():
    xg, vg = _rand(5, (NY, NX, NZ)), _rand(7, (NY, NX, NZ))
    wg = _rand(6, (NY, NX // 2 + 1, NZ))
    wpost = np.random.default_rng(8).standard_normal((NY, NX, NZ))
    for mesh in MESHES:
        L = _layout(mesh)
        if L.n_ranks_x > 1:
            # value: the row transpose puts every element where expected
            y = split_concat_transpose(_block(xg, mesh), 2, 1, L.n_ranks_x,
                                       L.comm_row)
            np.testing.assert_array_equal(np.asarray(y), _row_post_block(xg, mesh))
            wt = jnp.asarray(_row_post_block(wpost, mesh))   # position-dependent

            def loss_t(x, L=L, wt=wt):
                y = split_concat_transpose(x, 2, 1, L.n_ranks_x, L.comm_row)
                return global_sum_mpi(jnp.sum(wt * jnp.sin(y) * y), final_loss=True)

            _fd_checks(loss_t, _block(xg, mesh), _block(vg, mesh))
        wl = jnp.asarray(np.abs(_spec_ref(wg, mesh, NZ)))

        def loss_fft(x, L=L, wl=wl):
            back = layout_irfft2(layout_rfft2(x, L) * (1.0 + wl), L, NZ)
            return global_sum_mpi(jnp.sum(back ** 3), final_loss=True)

        _fd_checks(loss_fft, _block(xg, mesh), _block(vg, mesh))


def test_pencil_box3_gradient_matches_serial():
    """AD through the pencil x- and y-halo sendrecvs of the LASD box average."""
    xg, vg, wg = _rand(9, (NY, NX, NZ)), _rand(10, (NY, NX, NZ)), _rand(11, (NY, NX, NZ))
    gs = jax.grad(lambda x: jnp.sum(wg * imfilter_box3(x) ** 2))(xg)
    for mesh in MESHES:
        L, wb = _layout(mesh), _block(wg, mesh)
        g = _fd_checks(lambda x, L=L, wb=wb: global_sum_mpi(
            jnp.sum(wb * imfilter_box3(x, L) ** 2), final_loss=True),
            _block(xg, mesh), _block(vg, mesh))
        np.testing.assert_allclose(_gather(g, mesh), np.asarray(gs),
                                   rtol=1e-10, atol=1e-12)


def test_pencil_step_gradients_match_serial():
    """One full dealiased step(): reverse-mode gradient of a scalar loss passes
    the global FD check and equals the serial gradient (slab tolerance 1e-5,
    the wall-model reduction-order class)."""
    cfg = SpectralLESConfig(nx=NX, ny=NY, nz=NZ, Lx=300.0, Ly=200.0, Lz=100.0,
                            dealias=True, smagorinsky_dynamic=False,
                            sgs_model="smagorinsky", spectral_filter=True,
                            time_scheme="ab2")
    st = _init()
    vg = 0.01 * _rand(12, (NY, NX, NZ))

    def step_loss(u, s0, g, reduce):
        s, _ = step(s0._replace(u=u), g, _DT, _UG, _FC, first=True, force=_FORCE)
        return reduce(jnp.sum(s.u ** 2) + jnp.sum(s.v ** 2) + jnp.sum(s.w ** 2))

    gs = jax.grad(lambda u: step_loss(u, st, make_grid(cfg), lambda a: a))(st.u)
    for mesh in MESHES:
        g, s0 = make_grid(cfg, layout=_layout(mesh)), _block_state(st, mesh)
        fin = lambda a: global_sum_mpi(a, final_loss=True)  # noqa: E731
        loss = jax.jit(lambda u, s0=s0, g=g: step_loss(u, s0, g, fin))
        gd = _fd_checks(loss, s0.u, _block(vg, mesh), tol=1e-5)
        np.testing.assert_allclose(_gather(gd, mesh), np.asarray(gs),
                                   rtol=1e-5, atol=1e-5, err_msg=f"{mesh}")


def test_pencil_layout_validation():
    with pytest.raises(ValueError):
        make_spectral_les_pencil_layout(COMM, NPROC + 1, 1, NY, NX)
    if NPROC > 1:
        with pytest.raises(ValueError):
            make_spectral_les_pencil_layout(COMM, 1, NPROC, NY, NX + 1)
