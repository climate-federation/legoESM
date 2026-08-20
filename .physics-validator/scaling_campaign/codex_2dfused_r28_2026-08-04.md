Reading additional input from stdin...
OpenAI Codex v0.145.0
--------
workdir: /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
model: gpt-5.6-terra
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019fcda6-b913-7063-93f8-255838917893
--------
user
Adversarial review of consult-item #3's implementation (git diff HEAD -- packages/core/legoesm/parallel/latlon_mpi.py packages/core/legoesm/grids/halo_latlon.py; new file tests/distributed/test_latlon_2d_fused_pad.py).
Change: new pad_with_pole_bc_lat_multi_2d — fused lat-axis wall pad for LatLon2DLayout, dtype-grouped single sendrecv pair per cut (mirrors the band pad_with_pole_bc_lat_multi_mpi), dispatched from pad_with_pole_bc_lat_multi when topology is LatLon2DLayout (previously fell through to per-field). Guard: halo <= n_lat_global//proc_lat (same as _pad_lat_wall_2d). Verified on 2 ranks: values bit-equal to per-field pad_with_pole_bc_lat_2d across mixed dtypes/shapes AND jax.grad finite. A/B queued at r512/512 ranks.
Attack: (1) deadlock safety — is the fused schedule identical on every rank when dtype sets differ per rank (they can't in practice? prove or flag), and does the pole/interior branch structure match _pad_lat_wall_2d exactly; (2) the 2-D lon axis: does any caller expect the multi pad to also pad lon (band contract says lat-only — confirm 2-D matches); (3) tag scheme reuse (rank/south_rank tags) colliding with the concurrent lon-ring exchanges on the same comm; (4) the halo guard vs the band variant's n_lat_local check — right one for 2-D?; (5) test quality: is the equivalence test non-vacuous (would it fail if fusion mis-ordered a dtype group?), and should it be a tracked pytest rather than the ad-hoc runner I used; (6) any missing __all__/export or docs. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + numbered issues.
codex
I’ll trace the new 2-D fused path against the existing per-field and band implementations, including communication ordering, tags, caller expectations, and whether the test would catch schedule/order errors.
exec
/usr/bin/bash -lc 'git diff --find-renames HEAD -- packages/core/legoesm/parallel/latlon_mpi.py packages/core/legoesm/grids/halo_latlon.py tests/distributed/test_latlon_2d_fused_pad.py && rg -n -C 3 "pad_with_pole_bc_lat_multi|pad_with_pole_bc_lat_2d|_pad_lat_wall_2d|sendrecv|halo.*n_lat|LatLon2DLayout|lon.*ring" packages/core/legoesm tests/distributed' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 510ms:
diff --git a/packages/core/legoesm/grids/halo_latlon.py b/packages/core/legoesm/grids/halo_latlon.py
index 08bece42d..009c1cae1 100644
--- a/packages/core/legoesm/grids/halo_latlon.py
+++ b/packages/core/legoesm/grids/halo_latlon.py
@@ -788,7 +788,9 @@ def pad_with_pole_bc_lat_multi(
     fused = os.environ.get("LEGOESM_LATLON_FUSED_HALO", "1") != "0"
     if get_halo_backend() == "mpi" and fused:
         from legoesm.parallel.latlon_mpi import (
+            LatLon2DLayout,
             LatLonBandLayout,
+            pad_with_pole_bc_lat_multi_2d,
             pad_with_pole_bc_lat_multi_mpi,
         )
         topology = get_mpi_topology()
@@ -797,13 +799,19 @@ def pad_with_pole_bc_lat_multi(
                 fields, topology, halo=halo,
                 south_values=south_values, north_values=north_values,
             )
-    # Local backend / non-latlon topology / 2-D pencil / fused-off:
-    # per-field pads (bit-identical semantics; under band MPI this is the
-    # legacy one-sendrecv-pair-per-field schedule).  The 2-D pencil routes
-    # HERE on purpose — the fused path above is keyed to LatLonBandLayout,
-    # so each field re-enters ``pad_with_pole_bc_lat`` and takes its
-    # lat-only ``pad_with_pole_bc_lat_2d`` branch (fusing the 2-D lat
-    # sendrecv is a later perf increment, not a correctness gap).
+        if isinstance(topology, LatLon2DLayout):
+            # 2-D pencil twin (codex consult #3, 2026-08-04): same
+            # dtype-group fusion on the lat axis; lon stays a separate
+            # dispatched exchange, as in the single-field 2-D path.
+            return pad_with_pole_bc_lat_multi_2d(
+                fields, topology, halo=halo,
+                south_values=south_values, north_values=north_values,
+            )
+    # Local backend / non-latlon topology / fused-off: per-field pads
+    # (bit-identical semantics; under band MPI this is the legacy
+    # one-sendrecv-pair-per-field schedule).  The 2-D pencil now has its
+    # own fused branch above (2026-08-04); it lands here only when the
+    # fused gate is off.
     return tuple(
         pad_with_pole_bc_lat(
             f, halo=halo,
diff --git a/packages/core/legoesm/parallel/latlon_mpi.py b/packages/core/legoesm/parallel/latlon_mpi.py
index 9d85f4c5b..88fc6c611 100644
--- a/packages/core/legoesm/parallel/latlon_mpi.py
+++ b/packages/core/legoesm/parallel/latlon_mpi.py
@@ -1707,6 +1707,120 @@ def pad_with_pole_bc_lat_mpi(
 # the single-field path.
 
 
+def pad_with_pole_bc_lat_multi_2d(
+    fields,
+    layout: "LatLon2DLayout",
+    halo: int = 1,
+    south_values=None,
+    north_values=None,
+):
+    """Fused lat-axis wall pad for a 2-D PENCIL layout (codex consult #3).
+
+    The 2-D twin of :func:`pad_with_pole_bc_lat_multi_mpi`: N independent
+    wall-BC scalars share ONE lat sendrecv pair per cut per dtype group
+    instead of one pair per field, exactly as the band lane does. Only
+    the lat axis is touched (the pencil's lon halo is a separate
+    dispatched exchange) — the same contract as the single-field
+    :func:`pad_with_pole_bc_lat_2d`, which this is value-identical to
+    (both fill pole rows with the constants and sendrecv interior cuts;
+    concatenate/slice carry native VJPs and the exchange is the shared
+    AD-safe :func:`get_sendrecv_vjp`).
+
+    Guard: the halo must fit the SMALLEST local lat block, the same
+    condition :func:`_pad_lat_wall_2d` enforces — a neighbour owning
+    fewer rows would send a mismatched slab and hang.
+    """
+    fields = tuple(fields)
+    n = len(fields)
+    if n == 0:
+        return ()
+    if halo <= 0:
+        return fields
+    if south_values is None:
+        south_values = (0.0,) * n
+    if north_values is None:
+        north_values = (0.0,) * n
+    south_values = tuple(south_values)
+    north_values = tuple(north_values)
+    if len(south_values) != n or len(north_values) != n:
+        raise ValueError(
+            "pad_with_pole_bc_lat_multi_2d: south_values/north_values must "
+            f"match len(fields)={n}; got {len(south_values)}/"
+            f"{len(north_values)}.")
+    n_lat_local = fields[0].shape[0]
+    for i, f in enumerate(fields):
+        if f.shape[0] != n_lat_local:
+            raise ValueError(
+                "pad_with_pole_bc_lat_multi_2d: all fields must share "
+                f"n_lat_local (axis 0); field 0 has {n_lat_local}, field "
+                f"{i} has {f.shape[0]}.")
+    min_lat_block = layout.n_lat_global // layout.proc_lat
+    if halo > min_lat_block:
+        raise ValueError(
+            f"pad_with_pole_bc_lat_multi_2d: halo={halo} exceeds the "
+            f"smallest local lat block ({min_lat_block}); a neighbour "
+            f"would send/recv a mismatched halo and the exchange would "
+            f"abort/hang.")
+
+    south_slabs: list = [None] * n
+    north_slabs: list = [None] * n
+    if layout.south_rank is None:
+        for i, f in enumerate(fields):
+            south_slabs[i] = jnp.full(
+                (halo,) + f.shape[1:],
+                jnp.asarray(south_values[i], dtype=f.dtype))
+    if layout.north_rank is None:
+        for i, f in enumerate(fields):
+            north_slabs[i] = jnp.full(
+                (halo,) + f.shape[1:],
+                jnp.asarray(north_values[i], dtype=f.dtype))
+
+    if layout.south_rank is not None or layout.north_rank is not None:
+        import mpi4jax
+        from mpi4py import MPI
+
+        comm = MPI.COMM_WORLD
+        sendrecv = get_sendrecv_vjp(mpi4jax)
+        # dtype groups in first-appearance order — trace-deterministic, so
+        # every rank issues the same fused schedule (pairing depends on it).
+        groups: dict = {}
+        for i, f in enumerate(fields):
+            groups.setdefault(jnp.dtype(f.dtype), []).append(i)
+        for idxs in groups.values():
+            sizes = [
+                halo * int(np.prod(fields[i].shape[1:], dtype=np.int64))
+                for i in idxs
+            ]
+            offsets = np.concatenate([[0], np.cumsum(sizes)])
+            if layout.south_rank is not None:
+                send_bot = jnp.concatenate(
+                    [fields[i][:halo].reshape(-1) for i in idxs])
+                recv_south = sendrecv(
+                    send_bot, jnp.zeros_like(send_bot),
+                    layout.south_rank, layout.south_rank,
+                    layout.rank, layout.south_rank, comm)
+                for k, i in enumerate(idxs):
+                    south_slabs[i] = recv_south[
+                        offsets[k]:offsets[k + 1]
+                    ].reshape((halo,) + fields[i].shape[1:])
+            if layout.north_rank is not None:
+                send_top = jnp.concatenate(
+                    [fields[i][-halo:].reshape(-1) for i in idxs])
+                recv_north = sendrecv(
+                    send_top, jnp.zeros_like(send_top),
+                    layout.north_rank, layout.north_rank,
+                    layout.rank, layout.north_rank, comm)
+                for k, i in enumerate(idxs):
+                    north_slabs[i] = recv_north[
+                        offsets[k]:offsets[k + 1]
+                    ].reshape((halo,) + fields[i].shape[1:])
+
+    return tuple(
+        jnp.concatenate([south_slabs[i], fields[i], north_slabs[i]], axis=0)
+        for i in range(n)
+    )
+
+
 def pad_with_pole_bc_lat_multi_mpi(
     fields,
     layout: LatLonBandLayout,
tests/distributed/test_voronoi_mpi.py-95-
tests/distributed/test_voronoi_mpi.py-96-    ``ssp_rk54_scan`` is the production library default (what direct MPAS
tests/distributed/test_voronoi_mpi.py-97-    construction selects), and it evaluates the TRiSK tendency — including the
tests/distributed/test_voronoi_mpi.py:98:    mpi4jax ``sendrecv`` halo exchange — INSIDE ``lax.scan``.  That ordered-
tests/distributed/test_voronoi_mpi.py-99-    effect-through-scan path is structurally different from the inline
tests/distributed/test_voronoi_mpi.py-100-    ``ssp_rk3``, so the serial-vs-MPI equivalence and mass-conservation tests
tests/distributed/test_voronoi_mpi.py-101-    run under BOTH.  (Both pass on the verified FFI stack jax 0.10.x +
--
tests/distributed/test_latlon_transpose_ad_mpi.py-12-reference needed), plus a finiteness check on a scalar-loss gradient.
tests/distributed/test_latlon_transpose_ad_mpi.py-13-
tests/distributed/test_latlon_transpose_ad_mpi.py-14-Run: ``mpirun -np {2,3,6} python -m pytest <thisfile>`` (wired into
tests/distributed/test_latlon_transpose_ad_mpi.py:15:mpi-distributed.yml).  Single rank skips (no lon ring to gather).
tests/distributed/test_latlon_transpose_ad_mpi.py-16-Size-adaptive: np2→1×2, np3→1×3, np6→2×3 (proc_lon≥2 always, so a real
tests/distributed/test_latlon_transpose_ad_mpi.py-17-ring); n_lon = 4·proc_lon keeps the allgather's equal-split requirement.
tests/distributed/test_latlon_transpose_ad_mpi.py-18-"""
--
tests/distributed/test_latlon_transpose_ad_mpi.py-40-
tests/distributed/test_latlon_transpose_ad_mpi.py-41-def _pick_grid(n_ranks: int) -> tuple[int, int]:
tests/distributed/test_latlon_transpose_ad_mpi.py-42-    """(proc_lat, proc_lon) with proc_lon holding the larger factor so a
tests/distributed/test_latlon_transpose_ad_mpi.py:43:    real lon ring (proc_lon>=2) is always exercised: np2→(1,2),
tests/distributed/test_latlon_transpose_ad_mpi.py-44-    np3→(1,3), np6→(2,3)."""
tests/distributed/test_latlon_transpose_ad_mpi.py-45-    best = (1, n_ranks)
tests/distributed/test_latlon_transpose_ad_mpi.py-46-    for pr in range(2, int(n_ranks ** 0.5) + 1):
--
tests/distributed/test_latlon_transpose_ad_mpi.py-55-    comm = MPI.COMM_WORLD
tests/distributed/test_latlon_transpose_ad_mpi.py-56-    rank, n = comm.Get_rank(), comm.Get_size()
tests/distributed/test_latlon_transpose_ad_mpi.py-57-    if n < 2:
tests/distributed/test_latlon_transpose_ad_mpi.py:58:        pytest.skip("needs mpirun with >=2 ranks (lon ring to gather)")
tests/distributed/test_latlon_transpose_ad_mpi.py-59-    pr, pc = _pick_grid(n)
tests/distributed/test_latlon_transpose_ad_mpi.py-60-    n_lat, n_lon = 2 * pr, 4 * pc          # equal lon split (n_lon % pc == 0)
tests/distributed/test_latlon_transpose_ad_mpi.py-61-    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
--
tests/distributed/test_latlon_transpose_ad_mpi.py-84-def test_transpose_grad_finite():
tests/distributed/test_latlon_transpose_ad_mpi.py-85-    """A scalar loss through gather → fold(roll, block-mixing) → grad is
tests/distributed/test_latlon_transpose_ad_mpi.py-86-    finite on every rank (no NaN/inf from the collective VJP; the roll
tests/distributed/test_latlon_transpose_ad_mpi.py:87:    makes the loss non-separable across lon blocks so the ring adjoint is
tests/distributed/test_latlon_transpose_ad_mpi.py-88-    genuinely exercised)."""
tests/distributed/test_latlon_transpose_ad_mpi.py-89-    comm = MPI.COMM_WORLD
tests/distributed/test_latlon_transpose_ad_mpi.py-90-    rank, n = comm.Get_rank(), comm.Get_size()
--
tests/distributed/conftest.py-51-    """Resynchronize ranks and drain stray messages around every test.
tests/distributed/conftest.py-52-
tests/distributed/conftest.py-53-    Each distributed test builds its own halo-exchange scenario (different
tests/distributed/conftest.py:54:    grids, layouts, and ``sendrecv`` buffer sizes).  Without an explicit
tests/distributed/conftest.py-55-    barrier between tests the ranks can drift out of lock-step, and a
tests/distributed/conftest.py-56-    message posted by one test that is not consumed before the next test
tests/distributed/conftest.py-57-    starts is later matched — by source/tag — against a *different-sized*
--
tests/distributed/test_lasd_mpi.py-85-
tests/distributed/test_lasd_mpi.py-86-@pytest.mark.skipif(NY % NPROC != 0, reason="NY must divide by n_ranks")
tests/distributed/test_lasd_mpi.py-87-def test_imfilter_box3_grad_matches_serial():
tests/distributed/test_lasd_mpi.py:88:    """Grad through the box3 y-halo == serial (the AD-safe sendrecv VJP)."""
tests/distributed/test_lasd_mpi.py-89-    f = _rand(3)
tests/distributed/test_lasd_mpi.py-90-
tests/distributed/test_lasd_mpi.py-91-    def loss(x, layout):
--
tests/distributed/test_plane_pencil_mpi.py-22-2. ``test_gather_round_trips_scatter`` — ``scatter`` then ``gather``
tests/distributed/test_plane_pencil_mpi.py-23-   reconstructs the original global field exactly on rank 0.
tests/distributed/test_plane_pencil_mpi.py-24-3. ``test_halo_supports_jax_grad`` — a scalar loss on the padded
tests/distributed/test_plane_pencil_mpi.py:25:   field returns finite gradients through ``_sendrecv_vjp`` on
tests/distributed/test_plane_pencil_mpi.py-26-   every rank.
tests/distributed/test_plane_pencil_mpi.py-27-4. ``test_halo_width_2_corners_correct`` — corner cells are filled
tests/distributed/test_plane_pencil_mpi.py-28-   correctly when ``halo > 1`` (two-axis composition picks up the
--
tests/distributed/test_plane_pencil_mpi.py-173-def test_single_rank_y_axis_self_wrap(halo):
tests/distributed/test_plane_pencil_mpi.py-174-    """Codex review 2026-05-24: `(N, 1)` mesh — every rank owns the
tests/distributed/test_plane_pencil_mpi.py-175-    full x range so the periodic E/W neighbour is the rank itself.
tests/distributed/test_plane_pencil_mpi.py:176:    The implementation must NOT issue self-sendrecv (which would
tests/distributed/test_plane_pencil_mpi.py-177-    deadlock or silently swap halos) and instead use local wrap on
tests/distributed/test_plane_pencil_mpi.py-178-    the singleton axis. Validates halo=1 and halo=2 vs the global
tests/distributed/test_plane_pencil_mpi.py-179-    wrap reference."""
--
tests/distributed/test_plane_pencil_mpi.py-274-
tests/distributed/test_plane_pencil_mpi.py-275-def test_halo_supports_jax_grad(mpi_layout):
tests/distributed/test_plane_pencil_mpi.py-276-    """jax.grad through the multi-rank exchange returns a finite
tests/distributed/test_plane_pencil_mpi.py:277:    array on every rank — proves _sendrecv_vjp backward swaps
tests/distributed/test_plane_pencil_mpi.py-278-    source/dest correctly under MPI."""
tests/distributed/test_plane_pencil_mpi.py-279-    layout = mpi_layout
tests/distributed/test_plane_pencil_mpi.py-280-    rng = np.random.default_rng(seed=99)
--
tests/distributed/test_latlon_mpi_checkpoint.py-148-    band = scatter_field_latlon(global_v, layout, is_v_face=True)
tests/distributed/test_latlon_mpi_checkpoint.py-149-
tests/distributed/test_latlon_mpi_checkpoint.py-150-    # Each rank simultaneously sends its last v-row north and receives
tests/distributed/test_latlon_mpi_checkpoint.py:151:    # the southern neighbour's last v-row.  Use ``comm.sendrecv`` so
tests/distributed/test_latlon_mpi_checkpoint.py-152-    # the test is implementation-independent — MPI_Send's buffering
tests/distributed/test_latlon_mpi_checkpoint.py-153-    # behaviour differs across openmpi / mpich / cray-mpich and a
tests/distributed/test_latlon_mpi_checkpoint.py-154-    # naive ``send`` followed by ``recv`` can deadlock when the
tests/distributed/test_latlon_mpi_checkpoint.py-155-    # message exceeds the eager-protocol threshold.
tests/distributed/test_latlon_mpi_checkpoint.py-156-    send_buf = np.asarray(band[-1]) if layout.north_rank is not None else None
tests/distributed/test_latlon_mpi_checkpoint.py:157:    recv_buf = comm.sendrecv(
tests/distributed/test_latlon_mpi_checkpoint.py-158-        sendobj=send_buf,
tests/distributed/test_latlon_mpi_checkpoint.py-159-        dest=(layout.north_rank if layout.north_rank is not None
tests/distributed/test_latlon_mpi_checkpoint.py-160-              else MPI.PROC_NULL),
--
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-155-        Exercises ``use_fv3_a2b_zeta_corner=True`` which previously
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-156-        wrapped ``interp_center_to_corner_a2b_ord4`` in ``jax.vmap``
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-157-        on the NH 4D path — and that vmap put ``pad_halo`` (halo=2)
tests/distributed/test_mpi_fv3_nh_step_fidelity.py:158:        inside the vmap, triggering mpi4jax's sendrecv batching
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-159-        assertion under MPI.  iter-1043 lifts the vmap (the helper
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-160-        is shape-polymorphic), so this test guards the fix.
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-161-        """
--
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-200-
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-201-        Exercises the iterated Laplacian path
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-202-        (``fv3_corner_laplacian_iteration``) that previously sat
tests/distributed/test_mpi_fv3_nh_step_fidelity.py:203:        inside ``jax.vmap`` over levels, crashing mpi4jax's sendrecv
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-204-        batch-axis rule.  iter-1044 lifts the vmap by making the
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-205-        helper shape-polymorphic.
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-206-
--
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-378-        # ``synchronize_cgrid_fluxes`` reading non-owned face flux
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-379-        # values directly (which under MPI replicated mode were
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-380-        # computed with zero halos).  iter-1049 added an MPI-aware
tests/distributed/test_mpi_fv3_nh_step_fidelity.py:381:        # sendrecv path; the factory test now passes with the
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-382-        # documented production pairing.
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-383-        grid = create_cubed_sphere(n, use_duogrid=True)
tests/distributed/test_mpi_fv3_nh_step_fidelity.py-384-        z_top = 30000.0
--
tests/distributed/test_latlon_2d_pad_wall_mpi.py-12-
tests/distributed/test_latlon_2d_pad_wall_mpi.py-13-Size-adaptive: the process grid is factored from the world size so the
tests/distributed/test_latlon_2d_pad_wall_mpi.py-14-SAME file covers np=2 (1×2 ring), np=3 (1×3 ring — the phase-constant
tests/distributed/test_latlon_2d_pad_wall_mpi.py:15:tag reverse-ring case, proc_lon≥3), and np=6 (2×3 — N/S reverse sendrecv
tests/distributed/test_latlon_2d_pad_wall_mpi.py:16:AND the proc_lon=3 reverse ring together).  ``n_lat=2·pr+1`` /
tests/distributed/test_latlon_2d_pad_wall_mpi.py-17-``n_lon=2·pc+1`` give UNEVEN splits, so the remainder path is exercised
tests/distributed/test_latlon_2d_pad_wall_mpi.py-18-too (smallest block = 2 ≥ halo).
tests/distributed/test_latlon_2d_pad_wall_mpi.py-19-"""
--
tests/distributed/test_latlon_2d_pad_wall_mpi.py-44-    """Factor the world size into ``(proc_lat, proc_lon)`` maximising
tests/distributed/test_latlon_2d_pad_wall_mpi.py-45-    halo+VJP coverage: a balanced 2-D split (proc_lat≥2 AND proc_lon≥2)
tests/distributed/test_latlon_2d_pad_wall_mpi.py-46-    when ``n_ranks`` is composite, else a 1×n ring for primes.  Always
tests/distributed/test_latlon_2d_pad_wall_mpi.py:47:    prefers the LARGER factor on proc_lon so the periodic-ring reverse
tests/distributed/test_latlon_2d_pad_wall_mpi.py-48-    VJP (phase-constant tags at proc_lon≥3) gets exercised whenever a
tests/distributed/test_latlon_2d_pad_wall_mpi.py-49-    factor allows it."""
tests/distributed/test_latlon_2d_pad_wall_mpi.py-50-    best = (1, n_ranks)
--
tests/distributed/test_latlon_2d_pad_wall_mpi.py-107-
tests/distributed/test_latlon_2d_pad_wall_mpi.py-108-
tests/distributed/test_latlon_2d_pad_wall_mpi.py-109-def test_mpi_wall_grad_parity():
tests/distributed/test_latlon_2d_pad_wall_mpi.py:110:    """Reverse-mode AD through BOTH sendrecv VJPs.  The loss is
tests/distributed/test_latlon_2d_pad_wall_mpi.py-111-    ``global_sum_mpi(sum(pad_local**2))`` (AD-safe allreduce SUM) —
tests/distributed/test_latlon_2d_pad_wall_mpi.py-112-    WITHOUT the global sum, jax.grad of a rank's OWN loss collapses to
tests/distributed/test_latlon_2d_pad_wall_mpi.py-113-    ``2*local`` (ghost cotangents flow AWAY to neighbours and never
--
tests/distributed/test_latlon_2d_pad_wall_mpi.py-116-    flow BACK through the halo VJP into this rank's owned cells, so the
tests/distributed/test_latlon_2d_pad_wall_mpi.py-117-    distributed grad on a rank's OWNED block == the GLOBAL grad of
tests/distributed/test_latlon_2d_pad_wall_mpi.py-118-    ``sum_r sum(P[window_r]**2)`` (P=_serial_ref_jax) restricted to that
tests/distributed/test_latlon_2d_pad_wall_mpi.py:119:    block.  ``proc_lon≥3`` (np=3, np=6) drives the periodic ring through
tests/distributed/test_latlon_2d_pad_wall_mpi.py-120-    REVERSE with the phase-constant tags it requires; ``proc_lat≥2``
tests/distributed/test_latlon_2d_pad_wall_mpi.py:121:    (np=6) drives the N/S reverse sendrecv."""
tests/distributed/test_latlon_2d_pad_wall_mpi.py-122-    comm = MPI.COMM_WORLD
tests/distributed/test_latlon_2d_pad_wall_mpi.py-123-    rank, n = comm.Get_rank(), comm.Get_size()
tests/distributed/test_latlon_2d_pad_wall_mpi.py-124-    if n < 2:
--
packages/core/legoesm/parallel/cube_face_scatter.py-27-   halo cell, so metrics need slicing, NOT halo exchange.
packages/core/legoesm/parallel/cube_face_scatter.py-28-2. **The scattered face-only halo already works.** ``_pad_halo_mpi_face_only``
packages/core/legoesm/parallel/cube_face_scatter.py-29-   keys off ``data.shape[0]`` and supports ``shape[0] == len(local_face_ids)``
packages/core/legoesm/parallel/cube_face_scatter.py:30:   for scalar / 4D / vector / interp_offsets / AD (custom_vjp sendrecv).
packages/core/legoesm/parallel/cube_face_scatter.py-31-3. **interp_offsets and duogrid stay FULL ``(6, ...)``.** The halo machinery
packages/core/legoesm/parallel/cube_face_scatter.py-32-   indexes them by the GLOBAL face id (``interp_offsets[global_face, edge]``),
packages/core/legoesm/parallel/cube_face_scatter.py-33-   so they must NOT be sliced — see ``_GLOBAL_FACE_INDEXED_FIELDS``.  This is
--
packages/core/legoesm/parallel/latlon_mpi.py-41-  For ``v`` we keep the duplicated boundary row across neighbours; the
packages/core/legoesm/parallel/latlon_mpi.py-42-  scatter and halo-exchange helpers handle this explicitly.
packages/core/legoesm/parallel/latlon_mpi.py-43-
packages/core/legoesm/parallel/latlon_mpi.py:44:- AD safety: halo sendrecv uses
packages/core/legoesm/parallel/latlon_mpi.py:45:  :func:`legoesm.parallel.halo_exchange.get_sendrecv_vjp` (the AD-safe
packages/core/legoesm/parallel/latlon_mpi.py:46:  wrapper around ``mpi4jax.sendrecv``).  Backward swaps source/dest as
packages/core/legoesm/parallel/latlon_mpi.py-47-  required by the reverse-mode rule.
packages/core/legoesm/parallel/latlon_mpi.py-48-
packages/core/legoesm/parallel/latlon_mpi.py-49-Status
--
packages/core/legoesm/parallel/latlon_mpi.py-81-    fold_pole_rows,
packages/core/legoesm/parallel/latlon_mpi.py-82-    fold_pole_rows_3d,
packages/core/legoesm/parallel/latlon_mpi.py-83-)
packages/core/legoesm/parallel/latlon_mpi.py:84:from legoesm.parallel.halo_exchange import get_sendrecv_vjp
packages/core/legoesm/parallel/latlon_mpi.py-85-
packages/core/legoesm/parallel/latlon_mpi.py-86-
packages/core/legoesm/parallel/latlon_mpi.py-87-# ============================================================================
--
packages/core/legoesm/parallel/latlon_mpi.py-160-        Number of contiguous bands.
packages/core/legoesm/parallel/latlon_mpi.py-161-    min_rows : int
packages/core/legoesm/parallel/latlon_mpi.py-162-        Minimum rows per band (halo floor — the MPI pads raise when
packages/core/legoesm/parallel/latlon_mpi.py:163:        ``halo > n_lat_local``; pass 2 for the halo=2 PPM/biharmonic paths).
packages/core/legoesm/parallel/latlon_mpi.py-164-
packages/core/legoesm/parallel/latlon_mpi.py-165-    Returns
packages/core/legoesm/parallel/latlon_mpi.py-166-    -------
--
packages/core/legoesm/parallel/latlon_mpi.py-318-    )
packages/core/legoesm/parallel/latlon_mpi.py-319-
packages/core/legoesm/parallel/latlon_mpi.py-320-
packages/core/legoesm/parallel/latlon_mpi.py:321:class LatLon2DLayout(NamedTuple):
packages/core/legoesm/parallel/latlon_mpi.py-322-    """2-D pencil (lat × lon) decomposition layout for MPI.
packages/core/legoesm/parallel/latlon_mpi.py-323-
packages/core/legoesm/parallel/latlon_mpi.py-324-    Increment 2 of the lat-lon 2-D decomposition
--
packages/core/legoesm/parallel/latlon_mpi.py-372-    n_lat: int,
packages/core/legoesm/parallel/latlon_mpi.py-373-    n_lon: int,
packages/core/legoesm/parallel/latlon_mpi.py-374-    fold: "FoldDescriptor | None" = None,
packages/core/legoesm/parallel/latlon_mpi.py:375:) -> LatLon2DLayout:
packages/core/legoesm/parallel/latlon_mpi.py-376-    """Build a 2-D pencil decomposition layout for ``rank``.
packages/core/legoesm/parallel/latlon_mpi.py-377-
packages/core/legoesm/parallel/latlon_mpi.py-378-    ``rank = proc_row * proc_lon + proc_col`` (row-major).  Latitude is
packages/core/legoesm/parallel/latlon_mpi.py-379-    split over ``proc_lat`` (line, pole-terminated), longitude over
packages/core/legoesm/parallel/latlon_mpi.py:380:    ``proc_lon`` (periodic ring).  ``proc_lat * proc_lon`` must equal the
packages/core/legoesm/parallel/latlon_mpi.py-381-    world size; ``proc_lon == 1`` gives the 1-D-band-equivalent layout.
packages/core/legoesm/parallel/latlon_mpi.py-382-    """
packages/core/legoesm/parallel/latlon_mpi.py-383-    if proc_lat < 1 or proc_lon < 1:
--
packages/core/legoesm/parallel/latlon_mpi.py-413-    west_rank = _rank_at(proc_row, (proc_col - 1) % proc_lon)
packages/core/legoesm/parallel/latlon_mpi.py-414-    east_rank = _rank_at(proc_row, (proc_col + 1) % proc_lon)
packages/core/legoesm/parallel/latlon_mpi.py-415-
packages/core/legoesm/parallel/latlon_mpi.py:416:    return LatLon2DLayout(
packages/core/legoesm/parallel/latlon_mpi.py-417-        rank=rank, n_ranks=n_ranks, proc_lat=proc_lat, proc_lon=proc_lon,
packages/core/legoesm/parallel/latlon_mpi.py-418-        proc_row=proc_row, proc_col=proc_col,
packages/core/legoesm/parallel/latlon_mpi.py-419-        n_lat_global=n_lat, n_lon_global=n_lon,
--
packages/core/legoesm/parallel/latlon_mpi.py-426-
packages/core/legoesm/parallel/latlon_mpi.py-427-
packages/core/legoesm/parallel/latlon_mpi.py-428-def scatter_field_latlon_2d(
packages/core/legoesm/parallel/latlon_mpi.py:429:    global_field: jax.Array, layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-430-) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-431-    """Slice the rank's 2-D block ``[lat_start:lat_end, lon_start:lon_end]``
packages/core/legoesm/parallel/latlon_mpi.py-432-    from a global (n_lat, n_lon[, ...]) field.  Deterministic slice (every
--
packages/core/legoesm/parallel/latlon_mpi.py-438-
packages/core/legoesm/parallel/latlon_mpi.py-439-
packages/core/legoesm/parallel/latlon_mpi.py-440-def gather_field_latlon_2d(
packages/core/legoesm/parallel/latlon_mpi.py:441:    local_field, layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-442-    *, is_u_face: bool = False, is_v_face: bool = False,
packages/core/legoesm/parallel/latlon_mpi.py-443-):
packages/core/legoesm/parallel/latlon_mpi.py-444-    """Reassemble the global field from every rank's 2-D block (I/O only).
--
packages/core/legoesm/parallel/latlon_mpi.py-499-    return out
packages/core/legoesm/parallel/latlon_mpi.py-500-
packages/core/legoesm/parallel/latlon_mpi.py-501-
packages/core/legoesm/parallel/latlon_mpi.py:502:def make_lon_row_comm(layout: LatLon2DLayout):
packages/core/legoesm/parallel/latlon_mpi.py:503:    """Create the longitude-ring sub-communicator for this rank's
packages/core/legoesm/parallel/latlon_mpi.py-504-    ``proc_row`` (the ``proc_lon`` ranks that share a latitude band).
packages/core/legoesm/parallel/latlon_mpi.py-505-
packages/core/legoesm/parallel/latlon_mpi.py-506-    COLLECTIVE over ``COMM_WORLD`` (every rank must call it).  ``key=
--
packages/core/legoesm/parallel/latlon_mpi.py-576-_lon_gather_full_p.defvjp(_lon_gather_full_p_fwd, _lon_gather_full_p_bwd)
packages/core/legoesm/parallel/latlon_mpi.py-577-
packages/core/legoesm/parallel/latlon_mpi.py-578-
packages/core/legoesm/parallel/latlon_mpi.py:579:def lon_gather_full(local_block: jax.Array, layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-580-                    row_comm) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-581-    """In-trace lat-pencil transpose: assemble the FULL longitude axis
packages/core/legoesm/parallel/latlon_mpi.py:582:    for this rank's latitude band from the ``proc_lon`` lon-ring blocks.
packages/core/legoesm/parallel/latlon_mpi.py-583-
packages/core/legoesm/parallel/latlon_mpi.py-584-    The crux primitive for any operation that needs all longitudes on a
packages/core/legoesm/parallel/latlon_mpi.py-585-    lon-split grid — the N/S pole-fold (180° lon shift / tripole perm)
--
packages/core/legoesm/parallel/latlon_mpi.py-613-
packages/core/legoesm/parallel/latlon_mpi.py-614-
packages/core/legoesm/parallel/latlon_mpi.py-615-def lon_scatter_full(full_field: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py:616:                     layout: LatLon2DLayout) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-617-    """Slice this rank's longitude block ``[lon_start:lon_end]`` from a
packages/core/legoesm/parallel/latlon_mpi.py-618-    full-longitude field — inverse of :func:`lon_gather_full`."""
packages/core/legoesm/parallel/latlon_mpi.py-619-    return full_field[:, layout.lon_start:layout.lon_end]
--
packages/core/legoesm/parallel/latlon_mpi.py-681-            pad_halo_latlon_vector_local if negate
packages/core/legoesm/parallel/latlon_mpi.py-682-            else pad_halo_latlon_local
packages/core/legoesm/parallel/latlon_mpi.py-683-        )
packages/core/legoesm/parallel/latlon_mpi.py:684:        padded = helper(field, halo=halo)        # (n_lat + 2h, n_lon + 2h)
packages/core/legoesm/parallel/latlon_mpi.py-685-        # Strip the lon halos to recover (halo, n_lon).
packages/core/legoesm/parallel/latlon_mpi.py-686-        south = padded[:halo, halo:halo + field.shape[1]]
packages/core/legoesm/parallel/latlon_mpi.py-687-        north = padded[-halo:, halo:halo + field.shape[1]]
--
packages/core/legoesm/parallel/latlon_mpi.py-690-            pad_halo_latlon_vector_3d_local if negate
packages/core/legoesm/parallel/latlon_mpi.py-691-            else pad_halo_latlon_3d_local
packages/core/legoesm/parallel/latlon_mpi.py-692-        )
packages/core/legoesm/parallel/latlon_mpi.py:693:        padded = helper(field, halo=halo)        # (n_lat + 2h, n_lon + 2h, nlev)
packages/core/legoesm/parallel/latlon_mpi.py-694-        south = padded[:halo, halo:halo + field.shape[1], :]
packages/core/legoesm/parallel/latlon_mpi.py-695-        north = padded[-halo:, halo:halo + field.shape[1], :]
packages/core/legoesm/parallel/latlon_mpi.py-696-    else:
--
packages/core/legoesm/parallel/latlon_mpi.py-838-    is_vector_v: bool = False,
packages/core/legoesm/parallel/latlon_mpi.py-839-    is_vector_u: bool = False,
packages/core/legoesm/parallel/latlon_mpi.py-840-) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py:841:    """Exchange ``halo`` ghost lat rows on each side via MPI sendrecv.
packages/core/legoesm/parallel/latlon_mpi.py-842-
packages/core/legoesm/parallel/latlon_mpi.py-843-    Boundary ranks (south_rank is None / north_rank is None) pole-fold
packages/core/legoesm/parallel/latlon_mpi.py-844-    using the same convention as
--
packages/core/legoesm/parallel/latlon_mpi.py-893-        return field
packages/core/legoesm/parallel/latlon_mpi.py-894-
packages/core/legoesm/parallel/latlon_mpi.py-895-    n_lat_local = field.shape[0]
packages/core/legoesm/parallel/latlon_mpi.py:896:    if halo > n_lat_local:
packages/core/legoesm/parallel/latlon_mpi.py:897:        # At halo=2 this can bite when n_lat_global < 2 * n_ranks * halo;
packages/core/legoesm/parallel/latlon_mpi.py-898-        # surface a clear error rather than producing garbage halos.
packages/core/legoesm/parallel/latlon_mpi.py-899-        raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py:900:            f"exchange_halo_latlon: halo={halo} exceeds n_lat_local="
packages/core/legoesm/parallel/latlon_mpi.py-901-            f"{n_lat_local} on rank {layout.rank}.  Reduce n_ranks or "
packages/core/legoesm/parallel/latlon_mpi.py-902-            "increase grid resolution."
packages/core/legoesm/parallel/latlon_mpi.py-903-        )
--
packages/core/legoesm/parallel/latlon_mpi.py-924-        ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-925-
packages/core/legoesm/parallel/latlon_mpi.py-926-    comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:927:    sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-928-
packages/core/legoesm/parallel/latlon_mpi.py-929-    # ---- South halo ----
packages/core/legoesm/parallel/latlon_mpi.py-930-    if layout.south_rank is not None:
--
packages/core/legoesm/parallel/latlon_mpi.py-934-        # halo).
packages/core/legoesm/parallel/latlon_mpi.py-935-        send_bot = field[:halo].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py-936-        recv_template = jnp.zeros_like(send_bot)
packages/core/legoesm/parallel/latlon_mpi.py:937:        recv_south = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-938-            send_bot, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-939-            layout.south_rank,         # source
packages/core/legoesm/parallel/latlon_mpi.py-940-            layout.south_rank,         # dest
--
packages/core/legoesm/parallel/latlon_mpi.py-950-    if layout.north_rank is not None:
packages/core/legoesm/parallel/latlon_mpi.py-951-        send_top = field[-halo:].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py-952-        recv_template = jnp.zeros_like(send_top)
packages/core/legoesm/parallel/latlon_mpi.py:953:        recv_north = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-954-            send_top, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-955-            layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-956-            layout.north_rank,
--
packages/core/legoesm/parallel/latlon_mpi.py-979-    Increment 1 of the lat-lon 2-D pencil decomposition
packages/core/legoesm/parallel/latlon_mpi.py-980-    (``docs/performance/scaling/latlon_2d_decomposition_design.md``).  Longitude is
packages/core/legoesm/parallel/latlon_mpi.py-981-    GLOBALLY PERIODIC, so — unlike the N/S :func:`exchange_halo_latlon`
packages/core/legoesm/parallel/latlon_mpi.py:982:    — there are no pole/wall ends: every rank in the longitude ring
packages/core/legoesm/parallel/latlon_mpi.py-983-    sends its west edge west and its east edge east, and the wrap is
packages/core/legoesm/parallel/latlon_mpi.py-984-    just the ring topology (the rank owning the last lon block has the
packages/core/legoesm/parallel/latlon_mpi.py-985-    rank owning the first as its east neighbour).  This is a STANDALONE
packages/core/legoesm/parallel/latlon_mpi.py-986-    primitive (takes plain neighbour ranks, not a layout) so the future
packages/core/legoesm/parallel/latlon_mpi.py:987:    ``LatLon2DLayout`` can call it with ``layout.west_rank`` etc.; it is
packages/core/legoesm/parallel/latlon_mpi.py-988-    NOT yet wired into any step.
packages/core/legoesm/parallel/latlon_mpi.py-989-
packages/core/legoesm/parallel/latlon_mpi.py:990:    When the longitude ring has a single member (``west_rank == east_rank
packages/core/legoesm/parallel/latlon_mpi.py-991-    == rank`` — i.e. ``proc_lon == 1``, the current 1-D latitude-band
packages/core/legoesm/parallel/latlon_mpi.py-992-    case) the wrap is performed LOCALLY from the rank's own columns,
packages/core/legoesm/parallel/latlon_mpi.py-993-    byte-identical to the existing ``jnp.roll`` periodic wrap and
--
packages/core/legoesm/parallel/latlon_mpi.py-998-    field : jax.Array, shape (n_lat_local, n_lon_local[, nlev])
packages/core/legoesm/parallel/latlon_mpi.py-999-        Rank-local interior field, no lon halos in input.
packages/core/legoesm/parallel/latlon_mpi.py-1000-    west_rank, east_rank : int
packages/core/legoesm/parallel/latlon_mpi.py:1001:        MPI ranks of the western / eastern longitude neighbours (ring).
packages/core/legoesm/parallel/latlon_mpi.py-1002-    rank : int
packages/core/legoesm/parallel/latlon_mpi.py-1003-        This process's MPI rank.
packages/core/legoesm/parallel/latlon_mpi.py-1004-    halo : int, default 1
--
packages/core/legoesm/parallel/latlon_mpi.py-1019-            "increase longitude resolution."
packages/core/legoesm/parallel/latlon_mpi.py-1020-        )
packages/core/legoesm/parallel/latlon_mpi.py-1021-
packages/core/legoesm/parallel/latlon_mpi.py:1022:    # Single-member lon ring (proc_lon == 1): local periodic wrap — the
packages/core/legoesm/parallel/latlon_mpi.py-1023-    # west ghost is the rank's own EAST edge, the east ghost its WEST
packages/core/legoesm/parallel/latlon_mpi.py-1024-    # edge.  Byte-identical to the legacy ``jnp.roll`` wrap; no MPI.
packages/core/legoesm/parallel/latlon_mpi.py-1025-    if west_rank == rank and east_rank == rank:
--
packages/core/legoesm/parallel/latlon_mpi.py-1037-        ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1038-
packages/core/legoesm/parallel/latlon_mpi.py-1039-    comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1040:    sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1041-    trailing_lat = field.shape[0]
packages/core/legoesm/parallel/latlon_mpi.py-1042-    other = field.shape[2:]  # (nlev,) or ()
packages/core/legoesm/parallel/latlon_mpi.py-1043-
packages/core/legoesm/parallel/latlon_mpi.py-1044-    # PHASE-CONSTANT tags with sendtag == recvtag (one per shift
packages/core/legoesm/parallel/latlon_mpi.py-1045-    # direction, +1 for the opposite) — the AD-safe convention from
packages/core/legoesm/parallel/latlon_mpi.py:1046:    # plane_mpi._TAG_EW.  The shared sendrecv VJP's backward swaps
packages/core/legoesm/parallel/latlon_mpi.py-1047-    # source<->dest but KEEPS the tags, so a rank-as-tag scheme
packages/core/legoesm/parallel/latlon_mpi.py-1048-    # (sendtag=rank, recvtag=source) mismatches on the reverse ring at
packages/core/legoesm/parallel/latlon_mpi.py-1049-    # proc_lon>=3 (codex review MAJOR 2026-06-13: gradients would hang).
--
packages/core/legoesm/parallel/latlon_mpi.py-1060-    # same-neighbour pattern (source==dest) that the N/S
packages/core/legoesm/parallel/latlon_mpi.py-1061-    # ``exchange_halo_latlon`` uses only unwinds on a LINE — the poles
packages/core/legoesm/parallel/latlon_mpi.py-1062-    # terminate the chain.  Longitude is a closed ring with no ends, and
packages/core/legoesm/parallel/latlon_mpi.py:1063:    # the AD-safe sendrecv token serialises this rank's two exchanges,
packages/core/legoesm/parallel/latlon_mpi.py-1064-    # so a same-neighbour west-then-east pattern makes phase-1 wait on
packages/core/legoesm/parallel/latlon_mpi.py-1065-    # the neighbour's phase-2 → circular deadlock (observed: job
packages/core/legoesm/parallel/latlon_mpi.py:1066:    # 8476475 timed out in mpi_sendrecv).  Instead each phase is a
packages/core/legoesm/parallel/latlon_mpi.py-1067-    # UNIFORM shift where every send is matched by a recv IN THE SAME
packages/core/legoesm/parallel/latlon_mpi.py-1068-    # phase (a permutation), so no cross-phase ring dependency exists.
packages/core/legoesm/parallel/latlon_mpi.py-1069-
packages/core/legoesm/parallel/latlon_mpi.py-1070-    # Phase 1 — EASTWARD shift: send our EAST edge to the east neighbour,
packages/core/legoesm/parallel/latlon_mpi.py-1071-    # receive the west neighbour's east edge into our WEST halo.
packages/core/legoesm/parallel/latlon_mpi.py-1072-    send_e = field[:, -halo:].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py:1073:    recv_w = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1074-        send_e, jnp.zeros_like(send_e),
packages/core/legoesm/parallel/latlon_mpi.py-1075-        west_rank,   # source (recv from west)
packages/core/legoesm/parallel/latlon_mpi.py-1076-        east_rank,   # dest   (send to east)
--
packages/core/legoesm/parallel/latlon_mpi.py-1083-    # Phase 2 — WESTWARD shift: send our WEST edge to the west neighbour,
packages/core/legoesm/parallel/latlon_mpi.py-1084-    # receive the east neighbour's west edge into our EAST halo.
packages/core/legoesm/parallel/latlon_mpi.py-1085-    send_w = field[:, :halo].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py:1086:    recv_e = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1087-        send_w, jnp.zeros_like(send_w),
packages/core/legoesm/parallel/latlon_mpi.py-1088-        east_rank,     # source (recv from east)
packages/core/legoesm/parallel/latlon_mpi.py-1089-        west_rank,     # dest   (send to west)
--
packages/core/legoesm/parallel/latlon_mpi.py-1096-    return jnp.concatenate([west_halo, field, east_halo], axis=1)
packages/core/legoesm/parallel/latlon_mpi.py-1097-
packages/core/legoesm/parallel/latlon_mpi.py-1098-
packages/core/legoesm/parallel/latlon_mpi.py:1099:def _pad_lat_wall_2d(
packages/core/legoesm/parallel/latlon_mpi.py-1100-    field: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py:1101:    layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-1102-    halo: int,
packages/core/legoesm/parallel/latlon_mpi.py-1103-    south_value: float,
packages/core/legoesm/parallel/latlon_mpi.py-1104-    north_value: float,
packages/core/legoesm/parallel/latlon_mpi.py-1105-) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-1106-    """Lat-axis (N/S) wall pad for a 2-D pencil row — the shared core of
packages/core/legoesm/parallel/latlon_mpi.py:1107:    :func:`pad_halo_latlon_2d` and :func:`pad_with_pole_bc_lat_2d`.
packages/core/legoesm/parallel/latlon_mpi.py-1108-
packages/core/legoesm/parallel/latlon_mpi.py:1109:    Interior lat cuts MPI-sendrecv their edge row with the south / north
packages/core/legoesm/parallel/latlon_mpi.py-1110-    neighbour (the AD-safe rank-as-tag LINE pattern — the lat axis is a
packages/core/legoesm/parallel/latlon_mpi.py-1111-    pole-terminated line, not the periodic ring); pole-touching rows
packages/core/legoesm/parallel/latlon_mpi.py-1112-    (``south_rank``/``north_rank is None``) fill the constant wall.
--
packages/core/legoesm/parallel/latlon_mpi.py-1122-    # lon side).  _even_split gives the first blocks one extra row, so the
packages/core/legoesm/parallel/latlon_mpi.py-1123-    # smallest block is floor(n_lat_global / proc_lat).
packages/core/legoesm/parallel/latlon_mpi.py-1124-    min_lat_block = layout.n_lat_global // layout.proc_lat
packages/core/legoesm/parallel/latlon_mpi.py:1125:    if halo > min_lat_block:
packages/core/legoesm/parallel/latlon_mpi.py-1126-        raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py:1127:            f"_pad_lat_wall_2d: halo={halo} exceeds the smallest local lat "
packages/core/legoesm/parallel/latlon_mpi.py-1128-            f"block ({min_lat_block}=n_lat_global {layout.n_lat_global}//"
packages/core/legoesm/parallel/latlon_mpi.py-1129-            f"proc_lat {layout.proc_lat}); a neighbour would send/recv a "
packages/core/legoesm/parallel/latlon_mpi.py-1130-            f"mismatched halo and the MPI exchange would abort/hang.")
--
packages/core/legoesm/parallel/latlon_mpi.py-1134-
packages/core/legoesm/parallel/latlon_mpi.py-1135-    # Both lat ends are physical poles (proc_lat==1): pure local wall pad,
packages/core/legoesm/parallel/latlon_mpi.py-1136-    # no MPI — keep the path runnable serially (the proc_lon==1, proc_lat==1
packages/core/legoesm/parallel/latlon_mpi.py:1137:    # single-process equivalence test, and the proc_lon-only ring case).
packages/core/legoesm/parallel/latlon_mpi.py-1138-    if south_is_pole and north_is_pole:
packages/core/legoesm/parallel/latlon_mpi.py-1139-        south = jnp.full((halo,) + trailing, south_value, field.dtype)
packages/core/legoesm/parallel/latlon_mpi.py-1140-        north = jnp.full((halo,) + trailing, north_value, field.dtype)
--
packages/core/legoesm/parallel/latlon_mpi.py-1144-    from mpi4py import MPI
packages/core/legoesm/parallel/latlon_mpi.py-1145-
packages/core/legoesm/parallel/latlon_mpi.py-1146-    comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1147:    sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1148-
packages/core/legoesm/parallel/latlon_mpi.py-1149-    if south_is_pole:
packages/core/legoesm/parallel/latlon_mpi.py-1150-        south = jnp.full((halo,) + trailing, south_value, field.dtype)
packages/core/legoesm/parallel/latlon_mpi.py-1151-    else:
packages/core/legoesm/parallel/latlon_mpi.py-1152-        send_s = field[:halo].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py:1153:        recv_s = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1154-            send_s, jnp.zeros_like(send_s),
packages/core/legoesm/parallel/latlon_mpi.py-1155-            layout.south_rank, layout.south_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1156-            layout.rank, layout.south_rank, comm)
--
packages/core/legoesm/parallel/latlon_mpi.py-1160-        north = jnp.full((halo,) + trailing, north_value, field.dtype)
packages/core/legoesm/parallel/latlon_mpi.py-1161-    else:
packages/core/legoesm/parallel/latlon_mpi.py-1162-        send_n = field[-halo:].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py:1163:        recv_n = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1164-            send_n, jnp.zeros_like(send_n),
packages/core/legoesm/parallel/latlon_mpi.py-1165-            layout.north_rank, layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1166-            layout.rank, layout.north_rank, comm)
--
packages/core/legoesm/parallel/latlon_mpi.py-1169-    return jnp.concatenate([south, field, north], axis=0)
packages/core/legoesm/parallel/latlon_mpi.py-1170-
packages/core/legoesm/parallel/latlon_mpi.py-1171-
packages/core/legoesm/parallel/latlon_mpi.py:1172:def pad_with_pole_bc_lat_2d(
packages/core/legoesm/parallel/latlon_mpi.py-1173-    interior: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py:1174:    layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-1175-    halo: int = 1,
packages/core/legoesm/parallel/latlon_mpi.py-1176-    south_value: float = 0.0,
packages/core/legoesm/parallel/latlon_mpi.py-1177-    north_value: float = 0.0,
--
packages/core/legoesm/parallel/latlon_mpi.py-1180-
packages/core/legoesm/parallel/latlon_mpi.py-1181-    The 2-D analogue of the band
packages/core/legoesm/parallel/latlon_mpi.py-1182-    :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat` MPI path: pad
packages/core/legoesm/parallel/latlon_mpi.py:1183:    only the lat axis (interior cut sendrecv + pole wall constant), leaving
packages/core/legoesm/parallel/latlon_mpi.py-1184-    longitude untouched.  The band path never split longitude, so its
packages/core/legoesm/parallel/latlon_mpi.py-1185-    wall-BC pad is lat-only; the 2-D pencil keeps that contract and the
packages/core/legoesm/parallel/latlon_mpi.py-1186-    operator adds lon ghosts through its own dispatched lon halo.  Routed
packages/core/legoesm/parallel/latlon_mpi.py-1187-    here from ``halo_latlon.pad_with_pole_bc_lat`` when the active topology
packages/core/legoesm/parallel/latlon_mpi.py:1188:    is a :class:`LatLon2DLayout` (wall poles only; the tripolar north fold /
packages/core/legoesm/parallel/latlon_mpi.py:1189:    vector-u seam is excluded upstream).  AD-safe via the shared sendrecv
packages/core/legoesm/parallel/latlon_mpi.py-1190-    VJP.
packages/core/legoesm/parallel/latlon_mpi.py-1191-    """
packages/core/legoesm/parallel/latlon_mpi.py-1192-    if halo <= 0:
packages/core/legoesm/parallel/latlon_mpi.py-1193-        return interior
packages/core/legoesm/parallel/latlon_mpi.py:1194:    return _pad_lat_wall_2d(interior, layout, halo, south_value, north_value)
packages/core/legoesm/parallel/latlon_mpi.py-1195-
packages/core/legoesm/parallel/latlon_mpi.py-1196-
packages/core/legoesm/parallel/latlon_mpi.py-1197-def pad_halo_latlon_2d(
packages/core/legoesm/parallel/latlon_mpi.py-1198-    field: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py:1199:    layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-1200-    halo: int = 1,
packages/core/legoesm/parallel/latlon_mpi.py-1201-    pole_bc: str = "wall",
packages/core/legoesm/parallel/latlon_mpi.py-1202-    south_value: float = 0.0,
--
packages/core/legoesm/parallel/latlon_mpi.py-1206-    integration of the 2-D building blocks.
packages/core/legoesm/parallel/latlon_mpi.py-1207-
packages/core/legoesm/parallel/latlon_mpi.py-1208-    N/S then E/W (corners ride the lat-padded edge columns into the E/W
packages/core/legoesm/parallel/latlon_mpi.py:1209:    exchange).  N/S: interior cuts MPI-sendrecv with the lat neighbour;
packages/core/legoesm/parallel/latlon_mpi.py-1210-    pole-touching rows fill the pole side per ``pole_bc``.  Every rank
packages/core/legoesm/parallel/latlon_mpi.py:1211:    participates in its interior-facing sendrecv (the line terminates at
packages/core/legoesm/parallel/latlon_mpi.py-1212-    the pole rows' local fill) so there is NO collective-line deadlock —
packages/core/legoesm/parallel/latlon_mpi.py-1213-    the failure of the earlier "guard-and-skip" interior-only attempt.
packages/core/legoesm/parallel/latlon_mpi.py-1214-    E/W: the periodic-ring :func:`exchange_halo_lon`.
--
packages/core/legoesm/parallel/latlon_mpi.py-1218-        (``south_value``/``north_value``).  This is the REGULAR lat-lon
packages/core/legoesm/parallel/latlon_mpi.py-1219-        case (and ocean, whose poles are closed walls).  Fully LOCAL at
packages/core/legoesm/parallel/latlon_mpi.py-1220-        the poles ⇒ no longitude transpose, and the whole pad is
packages/core/legoesm/parallel/latlon_mpi.py:1221:        DEADLOCK-FREE and AD-SAFE (sendrecv-VJP on both axes).
packages/core/legoesm/parallel/latlon_mpi.py-1222-      * ``"fold"`` / ``"tripole"`` — the atmospheric 180° pole-fold and
packages/core/legoesm/parallel/latlon_mpi.py-1223-        the ocean tripole north-fold need the FULL longitude axis at the
packages/core/legoesm/parallel/latlon_mpi.py-1224-        pole row (180° shift / permutation), so they require the
--
packages/core/legoesm/parallel/latlon_mpi.py-1252-    # the SMALLEST block is exactly floor(n_global / parts).
packages/core/legoesm/parallel/latlon_mpi.py-1253-    min_lat_block = layout.n_lat_global // layout.proc_lat
packages/core/legoesm/parallel/latlon_mpi.py-1254-    min_lon_block = layout.n_lon_global // layout.proc_lon
packages/core/legoesm/parallel/latlon_mpi.py:1255:    if halo > min_lat_block or halo > min_lon_block:
packages/core/legoesm/parallel/latlon_mpi.py-1256-        raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py-1257-            f"pad_halo_latlon_2d: halo={halo} exceeds the smallest local "
packages/core/legoesm/parallel/latlon_mpi.py-1258-            f"block (lat {min_lat_block}=n_lat_global "
--
packages/core/legoesm/parallel/latlon_mpi.py-1261-            f"{layout.proc_lon}); a neighbour would send/recv a mismatched "
packages/core/legoesm/parallel/latlon_mpi.py-1262-            f"halo and the MPI exchange would abort/hang.")
packages/core/legoesm/parallel/latlon_mpi.py-1263-
packages/core/legoesm/parallel/latlon_mpi.py:1264:    # N/S — lat-axis wall pad (interior cut sendrecv at the AD-safe
packages/core/legoesm/parallel/latlon_mpi.py-1265-    # rank-as-tag LINE pattern, pole side wall); shared verbatim with
packages/core/legoesm/parallel/latlon_mpi.py:1266:    # pad_with_pole_bc_lat_2d so the two lat exchanges stay bit-identical.
packages/core/legoesm/parallel/latlon_mpi.py:1267:    ns = _pad_lat_wall_2d(field, layout, halo, south_value, north_value)
packages/core/legoesm/parallel/latlon_mpi.py-1268-    # E/W ring on the lat-padded block → fills lon ghosts + corners.
packages/core/legoesm/parallel/latlon_mpi.py-1269-    return exchange_halo_lon(
packages/core/legoesm/parallel/latlon_mpi.py-1270-        ns, layout.west_rank, layout.east_rank, layout.rank, halo=halo)
--
packages/core/legoesm/parallel/latlon_mpi.py-1277-# Used by :mod:`legoesm.grids.halo_latlon` when the global halo
packages/core/legoesm/parallel/latlon_mpi.py-1278-# backend is ``"mpi"`` and the active topology is a
packages/core/legoesm/parallel/latlon_mpi.py-1279-# :class:`LatLonBandLayout`.  Composes the existing
packages/core/legoesm/parallel/latlon_mpi.py:1280:# :func:`exchange_halo_latlon` (lat MPI sendrecv + boundary pole-fold)
packages/core/legoesm/parallel/latlon_mpi.py-1281-# with a periodic-lon wrap that every rank performs locally.  Output
packages/core/legoesm/parallel/latlon_mpi.py-1282-# shape matches the serial :func:`legoesm.grids.halo_latlon.pad_halo_latlon`
packages/core/legoesm/parallel/latlon_mpi.py-1283-# family — operators stay backend-oblivious.
--
packages/core/legoesm/parallel/latlon_mpi.py-1305-        pole-fold convention used by the serial
packages/core/legoesm/parallel/latlon_mpi.py-1306-        :func:`legoesm.grids.halo_latlon.pad_halo_latlon` (mirror +
packages/core/legoesm/parallel/latlon_mpi.py-1307-        180° lon shift + sign flip for vectors).
packages/core/legoesm/parallel/latlon_mpi.py:1308:      - Interior partition cuts MPI-sendrecv with the neighbour rank.
packages/core/legoesm/parallel/latlon_mpi.py-1309-
packages/core/legoesm/parallel/latlon_mpi.py-1310-    The result has the same shape and semantics as the serial
packages/core/legoesm/parallel/latlon_mpi.py-1311-    helper.  Under MPI the lat axis is the rank's band plus the halo
--
packages/core/legoesm/parallel/latlon_mpi.py-1330-    # Step 2: lat halo — pole-fold the LON-PADDED data at boundary
packages/core/legoesm/parallel/latlon_mpi.py-1331-    # ranks (matches serial ``pad_halo_latlon``'s lon-pad-then-fold
packages/core/legoesm/parallel/latlon_mpi.py-1332-    # order, so the pole-fold lon-shift is ``(n_lon + 2*halo) // 2``
packages/core/legoesm/parallel/latlon_mpi.py:1333:    # like serial does).  Interior partition cuts MPI sendrecv with
packages/core/legoesm/parallel/latlon_mpi.py-1334-    # the neighbour rank.
packages/core/legoesm/parallel/latlon_mpi.py-1335-    #
packages/core/legoesm/parallel/latlon_mpi.py-1336-    # We do NOT call ``exchange_halo_latlon`` here because that
--
packages/core/legoesm/parallel/latlon_mpi.py-1340-    # — neither matches the full-pad result serial expects.
packages/core/legoesm/parallel/latlon_mpi.py-1341-    if halo > data.shape[0]:
packages/core/legoesm/parallel/latlon_mpi.py-1342-        raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py:1343:            f"pad_halo_latlon_mpi: halo={halo} exceeds n_lat_local="
packages/core/legoesm/parallel/latlon_mpi.py-1344-            f"{data.shape[0]} on rank {layout.rank}."
packages/core/legoesm/parallel/latlon_mpi.py-1345-        )
packages/core/legoesm/parallel/latlon_mpi.py-1346-
--
packages/core/legoesm/parallel/latlon_mpi.py-1357-                lon_padded, halo, negate=is_vector_v,
packages/core/legoesm/parallel/latlon_mpi.py-1358-            )
packages/core/legoesm/parallel/latlon_mpi.py-1359-    else:
packages/core/legoesm/parallel/latlon_mpi.py:1360:        # MPI sendrecv with south neighbour — exchanges the
packages/core/legoesm/parallel/latlon_mpi.py:1361:        # lon-padded boundary rows.  AD-safe via get_sendrecv_vjp.
packages/core/legoesm/parallel/latlon_mpi.py-1362-        try:
packages/core/legoesm/parallel/latlon_mpi.py-1363-            import mpi4jax
packages/core/legoesm/parallel/latlon_mpi.py-1364-            from mpi4py import MPI
--
packages/core/legoesm/parallel/latlon_mpi.py-1368-                "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1369-            ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1370-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1371:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1372-        send_bot = lon_padded[:halo].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py-1373-        recv_template = jnp.zeros_like(send_bot)
packages/core/legoesm/parallel/latlon_mpi.py:1374:        recv_south = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1375-            send_bot, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-1376-            layout.south_rank, layout.south_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1377-            layout.rank, layout.south_rank, comm,
--
packages/core/legoesm/parallel/latlon_mpi.py-1411-                "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1412-            ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1413-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1414:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1415-        send_top = lon_padded[-halo:].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py-1416-        recv_template = jnp.zeros_like(send_top)
packages/core/legoesm/parallel/latlon_mpi.py:1417:        recv_north = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1418-            send_top, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-1419-            layout.north_rank, layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1420-            layout.rank, layout.north_rank, comm,
--
packages/core/legoesm/parallel/latlon_mpi.py-1435-    can't be reused directly.  This helper builds the (n_lat_local +
packages/core/legoesm/parallel/latlon_mpi.py-1436-    2*halo,) result by:
packages/core/legoesm/parallel/latlon_mpi.py-1437-
packages/core/legoesm/parallel/latlon_mpi.py:1438:      * inter-rank sendrecv on the lat axis (just like
packages/core/legoesm/parallel/latlon_mpi.py-1439-        ``exchange_halo_latlon``), and
packages/core/legoesm/parallel/latlon_mpi.py-1440-      * constant ``south_value`` / ``north_value`` pad at pole-touching
packages/core/legoesm/parallel/latlon_mpi.py-1441-        boundaries (instead of pole-fold).
--
packages/core/legoesm/parallel/latlon_mpi.py-1468-                "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1469-            ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1470-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1471:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1472-        send_bot = interior[:halo]
packages/core/legoesm/parallel/latlon_mpi.py-1473-        recv_template = jnp.zeros_like(send_bot)
packages/core/legoesm/parallel/latlon_mpi.py:1474:        south_slab = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1475-            send_bot, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-1476-            layout.south_rank, layout.south_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1477-            layout.rank, layout.south_rank, comm,
--
packages/core/legoesm/parallel/latlon_mpi.py-1492-                "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1493-            ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1494-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1495:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1496-        send_top = interior[-halo:]
packages/core/legoesm/parallel/latlon_mpi.py-1497-        recv_template = jnp.zeros_like(send_top)
packages/core/legoesm/parallel/latlon_mpi.py:1498:        north_slab = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1499-            send_top, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-1500-            layout.north_rank, layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1501-            layout.rank, layout.north_rank, comm,
--
packages/core/legoesm/parallel/latlon_mpi.py-1512-
packages/core/legoesm/parallel/latlon_mpi.py-1513-    Grid metrics (``grid.lat``, ``sin_lat``, tripolar ``dx_T`` rows, …)
packages/core/legoesm/parallel/latlon_mpi.py-1514-    are closed-over constants inside the jitted step, so their wall-BC
packages/core/legoesm/parallel/latlon_mpi.py:1515:    pads are static — yet the traced ``sendrecv`` op cannot be
packages/core/legoesm/parallel/latlon_mpi.py-1516-    constant-folded by XLA, so every operator call re-exchanged the
packages/core/legoesm/parallel/latlon_mpi.py-1517-    same bytes every step (census job 8459289: 14 metric pads/step, two
packages/core/legoesm/parallel/latlon_mpi.py-1518-    of them inside the barotropic PCG ``fori_loop`` body = 120 executed
packages/core/legoesm/parallel/latlon_mpi.py:1519:    sendrecv pairs/step at M=60).  This helper performs the exchange
packages/core/legoesm/parallel/latlon_mpi.py-1520-    ONCE at trace time with host mpi4py; the result is a compile-time
packages/core/legoesm/parallel/latlon_mpi.py-1521-    constant and the per-step collective disappears.
packages/core/legoesm/parallel/latlon_mpi.py-1522-
--
packages/core/legoesm/parallel/latlon_mpi.py-1579-
packages/core/legoesm/parallel/latlon_mpi.py-1580-    Pole-touching south rank → ``south_value`` constant pad.
packages/core/legoesm/parallel/latlon_mpi.py-1581-    Pole-touching north rank → ``north_value`` constant pad.
packages/core/legoesm/parallel/latlon_mpi.py:1582:    Interior partition cuts → MPI sendrecv with neighbour.
packages/core/legoesm/parallel/latlon_mpi.py-1583-
packages/core/legoesm/parallel/latlon_mpi.py-1584-    Reuses :func:`exchange_halo_latlon` for the inter-rank exchange;
packages/core/legoesm/parallel/latlon_mpi.py-1585-    overrides the pole-fold result with the requested constants on
packages/core/legoesm/parallel/latlon_mpi.py-1586-    the boundary ranks afterwards (so the wall-BC semantics is
packages/core/legoesm/parallel/latlon_mpi.py-1587-    preserved instead of pole-folding).  This is the SIMPLEST way
packages/core/legoesm/parallel/latlon_mpi.py:1588:    to reuse the existing AD-safe sendrecv machinery — we let
packages/core/legoesm/parallel/latlon_mpi.py-1589-    ``exchange_halo_latlon`` do the heavy lifting and then patch
packages/core/legoesm/parallel/latlon_mpi.py-1590-    the boundary halo slabs.
packages/core/legoesm/parallel/latlon_mpi.py-1591-
--
packages/core/legoesm/parallel/latlon_mpi.py-1607-    # Static-metric constant folding (census job 8459289): a concrete
packages/core/legoesm/parallel/latlon_mpi.py-1608-    # (non-Tracer) 1-D input is a closed-over compile-time constant —
packages/core/legoesm/parallel/latlon_mpi.py-1609-    # its wall-BC pad is exchanged ONCE at trace time via host MPI
packages/core/legoesm/parallel/latlon_mpi.py:1610:    # instead of a per-step traced sendrecv that XLA cannot fold.
packages/core/legoesm/parallel/latlon_mpi.py-1611-    # Codex-hardened gate (review 2026-06-10):
packages/core/legoesm/parallel/latlon_mpi.py-1612-    #   * ndim == 1 only — covers every measured static pad (the
packages/core/legoesm/parallel/latlon_mpi.py-1613-    #     1-D lat metrics; census 8459289) while keeping the rarely-
packages/core/legoesm/parallel/latlon_mpi.py-1614-    #     trodden 2-D tripolar-metric pads on the traced path;
packages/core/legoesm/parallel/latlon_mpi.py-1615-    #   * boundary constants must ALSO be non-Tracers (a traced
packages/core/legoesm/parallel/latlon_mpi.py-1616-    #     south/north value must not be constant-folded);
packages/core/legoesm/parallel/latlon_mpi.py:1617:    #   * same halo <= n_lat_local guard as the traced path (a silent
packages/core/legoesm/parallel/latlon_mpi.py-1618-    #     short send would truncate/hang instead of raising).
packages/core/legoesm/parallel/latlon_mpi.py-1619-    # INVARIANT (same class as every traced mpi4jax collective in this
packages/core/legoesm/parallel/latlon_mpi.py-1620-    # module): tracing is SPMD-symmetric — every rank traces the same
packages/core/legoesm/parallel/latlon_mpi.py-1621-    # jitted functions in the same order.  Rank-subset tracing would
packages/core/legoesm/parallel/latlon_mpi.py-1622-    # block in the trace-time Sendrecv exactly like rank-subset
packages/core/legoesm/parallel/latlon_mpi.py:1623:    # EXECUTION blocks the traced sendrecv.  Set
packages/core/legoesm/parallel/latlon_mpi.py-1624-    # ``LEGOESM_LATLON_STATIC_METRIC_PAD=0`` to restore the traced
packages/core/legoesm/parallel/latlon_mpi.py-1625-    # exchange (A/B + kill-switch lever).
packages/core/legoesm/parallel/latlon_mpi.py-1626-    import os
--
packages/core/legoesm/parallel/latlon_mpi.py-1643-
packages/core/legoesm/parallel/latlon_mpi.py-1644-    # 1D lat-axis metrics (sin_lat, cos_lat_v_interior, dx_cell, …)
packages/core/legoesm/parallel/latlon_mpi.py-1645-    # don't have a lon axis to pole-fold over.  Build the result
packages/core/legoesm/parallel/latlon_mpi.py:1646:    # directly: constant pad at pole-touching boundaries, sendrecv
packages/core/legoesm/parallel/latlon_mpi.py-1647-    # at interior cuts — no call to ``exchange_halo_latlon`` (which
packages/core/legoesm/parallel/latlon_mpi.py-1648-    # assumes axis 1 = lon and would fail on ndim=1).
packages/core/legoesm/parallel/latlon_mpi.py-1649-    if interior.ndim == 1:
--
packages/core/legoesm/parallel/latlon_mpi.py-1653-        )
packages/core/legoesm/parallel/latlon_mpi.py-1654-
packages/core/legoesm/parallel/latlon_mpi.py-1655-    # Step 1: run the standard halo exchange.  Boundary ranks will
packages/core/legoesm/parallel/latlon_mpi.py:1656:    # get pole-folded values; interior cuts will get sendrecv'd
packages/core/legoesm/parallel/latlon_mpi.py-1657-    # neighbour values (which is what we want — those are NOT to
packages/core/legoesm/parallel/latlon_mpi.py-1658-    # be overwritten).
packages/core/legoesm/parallel/latlon_mpi.py-1659-    padded = exchange_halo_latlon(
--
packages/core/legoesm/parallel/latlon_mpi.py-1690-# ============================================================================
packages/core/legoesm/parallel/latlon_mpi.py-1691-#
packages/core/legoesm/parallel/latlon_mpi.py-1692-# The lat-lon band MPI step issues O(30) independent wall-BC cell pads per
packages/core/legoesm/parallel/latlon_mpi.py:1693:# step, each paying its own token-serialized sendrecv pair (~200-300 us
packages/core/legoesm/parallel/latlon_mpi.py-1694-# latency on Ginsburg CPU nodes — the measured rank-growing term of the
packages/core/legoesm/parallel/latlon_mpi.py:1695:# baroclinic phase, jobs 8458934/8458989).  mpi4jax sendrecvs do NOT
packages/core/legoesm/parallel/latlon_mpi.py-1696-# overlap (token chain), so N independent pads cost N x latency.  Fusing
packages/core/legoesm/parallel/latlon_mpi.py:1697:# independent same-dataflow-level pads into ONE concatenated sendrecv per
packages/core/legoesm/parallel/latlon_mpi.py-1698-# cut per dtype group cuts that latency term by the cluster size while
packages/core/legoesm/parallel/latlon_mpi.py:1699:# exchanging bit-identical bytes (concat -> sendrecv -> split is value-
packages/core/legoesm/parallel/latlon_mpi.py:1700:# identical to per-field sendrecvs; no arithmetic).
packages/core/legoesm/parallel/latlon_mpi.py-1701-#
packages/core/legoesm/parallel/latlon_mpi.py-1702-# Scope: scalar wall-BC fields ONLY (``pad_ns_zero`` /
packages/core/legoesm/parallel/latlon_mpi.py-1703-# ``pad_with_pole_bc_lat`` with constant boundary values, no
--
packages/core/legoesm/parallel/latlon_mpi.py-1707-# the single-field path.
packages/core/legoesm/parallel/latlon_mpi.py-1708-
packages/core/legoesm/parallel/latlon_mpi.py-1709-
packages/core/legoesm/parallel/latlon_mpi.py:1710:def pad_with_pole_bc_lat_multi_2d(
packages/core/legoesm/parallel/latlon_mpi.py-1711-    fields,
packages/core/legoesm/parallel/latlon_mpi.py:1712:    layout: "LatLon2DLayout",
packages/core/legoesm/parallel/latlon_mpi.py-1713-    halo: int = 1,
packages/core/legoesm/parallel/latlon_mpi.py-1714-    south_values=None,
packages/core/legoesm/parallel/latlon_mpi.py-1715-    north_values=None,
packages/core/legoesm/parallel/latlon_mpi.py-1716-):
packages/core/legoesm/parallel/latlon_mpi.py-1717-    """Fused lat-axis wall pad for a 2-D PENCIL layout (codex consult #3).
packages/core/legoesm/parallel/latlon_mpi.py-1718-
packages/core/legoesm/parallel/latlon_mpi.py:1719:    The 2-D twin of :func:`pad_with_pole_bc_lat_multi_mpi`: N independent
packages/core/legoesm/parallel/latlon_mpi.py:1720:    wall-BC scalars share ONE lat sendrecv pair per cut per dtype group
packages/core/legoesm/parallel/latlon_mpi.py-1721-    instead of one pair per field, exactly as the band lane does. Only
packages/core/legoesm/parallel/latlon_mpi.py-1722-    the lat axis is touched (the pencil's lon halo is a separate
packages/core/legoesm/parallel/latlon_mpi.py-1723-    dispatched exchange) — the same contract as the single-field
packages/core/legoesm/parallel/latlon_mpi.py:1724:    :func:`pad_with_pole_bc_lat_2d`, which this is value-identical to
packages/core/legoesm/parallel/latlon_mpi.py:1725:    (both fill pole rows with the constants and sendrecv interior cuts;
packages/core/legoesm/parallel/latlon_mpi.py-1726-    concatenate/slice carry native VJPs and the exchange is the shared
packages/core/legoesm/parallel/latlon_mpi.py:1727:    AD-safe :func:`get_sendrecv_vjp`).
packages/core/legoesm/parallel/latlon_mpi.py-1728-
packages/core/legoesm/parallel/latlon_mpi.py-1729-    Guard: the halo must fit the SMALLEST local lat block, the same
packages/core/legoesm/parallel/latlon_mpi.py:1730:    condition :func:`_pad_lat_wall_2d` enforces — a neighbour owning
packages/core/legoesm/parallel/latlon_mpi.py-1731-    fewer rows would send a mismatched slab and hang.
packages/core/legoesm/parallel/latlon_mpi.py-1732-    """
packages/core/legoesm/parallel/latlon_mpi.py-1733-    fields = tuple(fields)
--
packages/core/legoesm/parallel/latlon_mpi.py-1744-    north_values = tuple(north_values)
packages/core/legoesm/parallel/latlon_mpi.py-1745-    if len(south_values) != n or len(north_values) != n:
packages/core/legoesm/parallel/latlon_mpi.py-1746-        raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py:1747:            "pad_with_pole_bc_lat_multi_2d: south_values/north_values must "
packages/core/legoesm/parallel/latlon_mpi.py-1748-            f"match len(fields)={n}; got {len(south_values)}/"
packages/core/legoesm/parallel/latlon_mpi.py-1749-            f"{len(north_values)}.")
packages/core/legoesm/parallel/latlon_mpi.py-1750-    n_lat_local = fields[0].shape[0]
packages/core/legoesm/parallel/latlon_mpi.py-1751-    for i, f in enumerate(fields):
packages/core/legoesm/parallel/latlon_mpi.py-1752-        if f.shape[0] != n_lat_local:
packages/core/legoesm/parallel/latlon_mpi.py-1753-            raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py:1754:                "pad_with_pole_bc_lat_multi_2d: all fields must share "
packages/core/legoesm/parallel/latlon_mpi.py-1755-                f"n_lat_local (axis 0); field 0 has {n_lat_local}, field "
packages/core/legoesm/parallel/latlon_mpi.py-1756-                f"{i} has {f.shape[0]}.")
packages/core/legoesm/parallel/latlon_mpi.py-1757-    min_lat_block = layout.n_lat_global // layout.proc_lat
packages/core/legoesm/parallel/latlon_mpi.py:1758:    if halo > min_lat_block:
packages/core/legoesm/parallel/latlon_mpi.py-1759-        raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py:1760:            f"pad_with_pole_bc_lat_multi_2d: halo={halo} exceeds the "
packages/core/legoesm/parallel/latlon_mpi.py-1761-            f"smallest local lat block ({min_lat_block}); a neighbour "
packages/core/legoesm/parallel/latlon_mpi.py-1762-            f"would send/recv a mismatched halo and the exchange would "
packages/core/legoesm/parallel/latlon_mpi.py-1763-            f"abort/hang.")
--
packages/core/legoesm/parallel/latlon_mpi.py-1780-        from mpi4py import MPI
packages/core/legoesm/parallel/latlon_mpi.py-1781-
packages/core/legoesm/parallel/latlon_mpi.py-1782-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1783:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1784-        # dtype groups in first-appearance order — trace-deterministic, so
packages/core/legoesm/parallel/latlon_mpi.py-1785-        # every rank issues the same fused schedule (pairing depends on it).
packages/core/legoesm/parallel/latlon_mpi.py-1786-        groups: dict = {}
--
packages/core/legoesm/parallel/latlon_mpi.py-1795-            if layout.south_rank is not None:
packages/core/legoesm/parallel/latlon_mpi.py-1796-                send_bot = jnp.concatenate(
packages/core/legoesm/parallel/latlon_mpi.py-1797-                    [fields[i][:halo].reshape(-1) for i in idxs])
packages/core/legoesm/parallel/latlon_mpi.py:1798:                recv_south = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1799-                    send_bot, jnp.zeros_like(send_bot),
packages/core/legoesm/parallel/latlon_mpi.py-1800-                    layout.south_rank, layout.south_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1801-                    layout.rank, layout.south_rank, comm)
--
packages/core/legoesm/parallel/latlon_mpi.py-1806-            if layout.north_rank is not None:
packages/core/legoesm/parallel/latlon_mpi.py-1807-                send_top = jnp.concatenate(
packages/core/legoesm/parallel/latlon_mpi.py-1808-                    [fields[i][-halo:].reshape(-1) for i in idxs])
packages/core/legoesm/parallel/latlon_mpi.py:1809:                recv_north = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1810-                    send_top, jnp.zeros_like(send_top),
packages/core/legoesm/parallel/latlon_mpi.py-1811-                    layout.north_rank, layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1812-                    layout.rank, layout.north_rank, comm)
--
packages/core/legoesm/parallel/latlon_mpi.py-1821-    )
packages/core/legoesm/parallel/latlon_mpi.py-1822-
packages/core/legoesm/parallel/latlon_mpi.py-1823-
packages/core/legoesm/parallel/latlon_mpi.py:1824:def pad_with_pole_bc_lat_multi_mpi(
packages/core/legoesm/parallel/latlon_mpi.py-1825-    fields,
packages/core/legoesm/parallel/latlon_mpi.py-1826-    layout: LatLonBandLayout,
packages/core/legoesm/parallel/latlon_mpi.py-1827-    halo: int = 1,
--
packages/core/legoesm/parallel/latlon_mpi.py-1832-
packages/core/legoesm/parallel/latlon_mpi.py-1833-    Pads every field in ``fields`` along the lat axis (axis 0) with
packages/core/legoesm/parallel/latlon_mpi.py-1834-    ``halo`` rows per side: constant ``south_values[i]`` /
packages/core/legoesm/parallel/latlon_mpi.py:1835:    ``north_values[i]`` at pole-touching boundaries, MPI-sendrecv'd
packages/core/legoesm/parallel/latlon_mpi.py-1836-    neighbour rows at interior partition cuts.  All fields must share
packages/core/legoesm/parallel/latlon_mpi.py-1837-    ``n_lat_local`` (axis 0); trailing shapes and dtypes may differ
packages/core/legoesm/parallel/latlon_mpi.py-1838-    (fields are flattened and concatenated per dtype group — ONE
packages/core/legoesm/parallel/latlon_mpi.py:1839:    sendrecv pair per cut per dtype group instead of one per field).
packages/core/legoesm/parallel/latlon_mpi.py-1840-
packages/core/legoesm/parallel/latlon_mpi.py-1841-    Value-identical to ``tuple(pad_with_pole_bc_lat_mpi(f, layout,
packages/core/legoesm/parallel/latlon_mpi.py-1842-    halo, sv, nv) for ...)`` for wall-BC scalars: the single-field path
--
packages/core/legoesm/parallel/latlon_mpi.py-1845-    directly produces the same result with less local compute.
packages/core/legoesm/parallel/latlon_mpi.py-1846-
packages/core/legoesm/parallel/latlon_mpi.py-1847-    AD-safe: the fused buffer goes through the same
packages/core/legoesm/parallel/latlon_mpi.py:1848:    :func:`get_sendrecv_vjp` custom-vjp as the single-field path;
packages/core/legoesm/parallel/latlon_mpi.py-1849-    ``concatenate``/``slice`` carry native JAX VJPs.
packages/core/legoesm/parallel/latlon_mpi.py-1850-
packages/core/legoesm/parallel/latlon_mpi.py-1851-    Returns a tuple of padded arrays, in input order.
--
packages/core/legoesm/parallel/latlon_mpi.py-1862-    north_values = tuple(north_values)
packages/core/legoesm/parallel/latlon_mpi.py-1863-    if len(south_values) != n or len(north_values) != n:
packages/core/legoesm/parallel/latlon_mpi.py-1864-        raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py:1865:            "pad_with_pole_bc_lat_multi_mpi: south_values/north_values "
packages/core/legoesm/parallel/latlon_mpi.py-1866-            f"must match len(fields)={n}; got {len(south_values)}/"
packages/core/legoesm/parallel/latlon_mpi.py-1867-            f"{len(north_values)}."
packages/core/legoesm/parallel/latlon_mpi.py-1868-        )
--
packages/core/legoesm/parallel/latlon_mpi.py-1873-    for i, f in enumerate(fields):
packages/core/legoesm/parallel/latlon_mpi.py-1874-        if f.shape[0] != n_lat_local:
packages/core/legoesm/parallel/latlon_mpi.py-1875-            raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py:1876:                "pad_with_pole_bc_lat_multi_mpi: all fields must share "
packages/core/legoesm/parallel/latlon_mpi.py-1877-                f"n_lat_local (axis 0); field 0 has {n_lat_local}, field "
packages/core/legoesm/parallel/latlon_mpi.py-1878-                f"{i} has {f.shape[0]}."
packages/core/legoesm/parallel/latlon_mpi.py-1879-            )
packages/core/legoesm/parallel/latlon_mpi.py:1880:    if halo > n_lat_local:
packages/core/legoesm/parallel/latlon_mpi.py-1881-        raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py:1882:            f"pad_with_pole_bc_lat_multi_mpi: halo={halo} exceeds "
packages/core/legoesm/parallel/latlon_mpi.py-1883-            f"n_lat_local={n_lat_local} on rank {layout.rank}."
packages/core/legoesm/parallel/latlon_mpi.py-1884-        )
packages/core/legoesm/parallel/latlon_mpi.py-1885-
--
packages/core/legoesm/parallel/latlon_mpi.py-1900-                jnp.asarray(north_values[i], dtype=f.dtype),
packages/core/legoesm/parallel/latlon_mpi.py-1901-            )
packages/core/legoesm/parallel/latlon_mpi.py-1902-
packages/core/legoesm/parallel/latlon_mpi.py:1903:    # Interior partition cuts: ONE fused sendrecv per cut per dtype group.
packages/core/legoesm/parallel/latlon_mpi.py-1904-    if layout.south_rank is not None or layout.north_rank is not None:
packages/core/legoesm/parallel/latlon_mpi.py-1905-        try:
packages/core/legoesm/parallel/latlon_mpi.py-1906-            import mpi4jax
--
packages/core/legoesm/parallel/latlon_mpi.py-1911-                "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1912-            ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1913-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1914:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1915-
packages/core/legoesm/parallel/latlon_mpi.py-1916-        # Group field indices by dtype in first-appearance order — the
packages/core/legoesm/parallel/latlon_mpi.py-1917-        # order is trace-deterministic, so every rank issues the same
packages/core/legoesm/parallel/latlon_mpi.py:1918:        # fused-message schedule (sendrecv pairing relies on it).
packages/core/legoesm/parallel/latlon_mpi.py-1919-        groups: dict = {}
packages/core/legoesm/parallel/latlon_mpi.py-1920-        for i, f in enumerate(fields):
packages/core/legoesm/parallel/latlon_mpi.py-1921-            groups.setdefault(jnp.dtype(f.dtype), []).append(i)
--
packages/core/legoesm/parallel/latlon_mpi.py-1931-                send_bot = jnp.concatenate(
packages/core/legoesm/parallel/latlon_mpi.py-1932-                    [fields[i][:halo].reshape(-1) for i in idxs]
packages/core/legoesm/parallel/latlon_mpi.py-1933-                )
packages/core/legoesm/parallel/latlon_mpi.py:1934:                recv_south = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1935-                    send_bot, jnp.zeros_like(send_bot),
packages/core/legoesm/parallel/latlon_mpi.py-1936-                    layout.south_rank,      # source
packages/core/legoesm/parallel/latlon_mpi.py-1937-                    layout.south_rank,      # dest
--
packages/core/legoesm/parallel/latlon_mpi.py-1948-                send_top = jnp.concatenate(
packages/core/legoesm/parallel/latlon_mpi.py-1949-                    [fields[i][-halo:].reshape(-1) for i in idxs]
packages/core/legoesm/parallel/latlon_mpi.py-1950-                )
packages/core/legoesm/parallel/latlon_mpi.py:1951:                recv_north = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1952-                    send_top, jnp.zeros_like(send_top),
packages/core/legoesm/parallel/latlon_mpi.py-1953-                    layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1954-                    layout.north_rank,
--
packages/core/legoesm/parallel/latlon_mpi.py-2011-    )
packages/core/legoesm/parallel/latlon_mpi.py-2012-
packages/core/legoesm/parallel/latlon_mpi.py-2013-
packages/core/legoesm/parallel/latlon_mpi.py:2014:def scatter_state_latlon_2d(state, layout: LatLon2DLayout):
packages/core/legoesm/parallel/latlon_mpi.py-2015-    """Extract the rank-local 2-D block from a global C-grid lat-lon state.
packages/core/legoesm/parallel/latlon_mpi.py-2016-
packages/core/legoesm/parallel/latlon_mpi.py-2017-    The 2-D analog of :func:`scatter_state_latlon`: slice BOTH the latitude
--
packages/core/legoesm/parallel/latlon_mpi.py-2573-
packages/core/legoesm/parallel/latlon_mpi.py-2574-
packages/core/legoesm/parallel/latlon_mpi.py-2575-def slice_latlon_grid_to_block_2d(
packages/core/legoesm/parallel/latlon_mpi.py:2576:    grid, layout: LatLon2DLayout, *, skip_total_area_reduce: bool = False,
packages/core/legoesm/parallel/latlon_mpi.py-2577-):
packages/core/legoesm/parallel/latlon_mpi.py-2578-    """Slice a global ``LatLonGrid`` to this rank's 2-D lat x lon block.
packages/core/legoesm/parallel/latlon_mpi.py-2579-
--
packages/core/legoesm/parallel/latlon_mpi.py-2702-    s, e = layout.lat_start, layout.lat_end
packages/core/legoesm/parallel/latlon_mpi.py-2703-    n_lat_g = layout.n_lat_global
packages/core/legoesm/parallel/latlon_mpi.py-2704-    s_pad = max(s - halo, 0)
packages/core/legoesm/parallel/latlon_mpi.py:2705:    e_pad = min(e + halo, n_lat_g)
packages/core/legoesm/parallel/latlon_mpi.py-2706-    pad_s = halo - (s - s_pad)   # rows to fabricate past south pole
packages/core/legoesm/parallel/latlon_mpi.py-2707-    pad_n = halo - (e_pad - e)   # rows to fabricate past north pole
packages/core/legoesm/parallel/latlon_mpi.py-2708-
--
packages/core/legoesm/parallel/latlon_mpi.py-2786-    backend-dispatched
packages/core/legoesm/parallel/latlon_mpi.py-2787-    :func:`legoesm.grids.halo_latlon.pad_halo_latlon` family — so
packages/core/legoesm/parallel/latlon_mpi.py-2788-    every operator that needs halo data fetches it via inter-rank
packages/core/legoesm/parallel/latlon_mpi.py:2789:    sendrecv at partition cuts and pole-fold / wall-BC constants at
packages/core/legoesm/parallel/latlon_mpi.py-2790-    boundary ranks, with no pre-padding of state on the caller side.
packages/core/legoesm/parallel/latlon_mpi.py-2791-
packages/core/legoesm/parallel/latlon_mpi.py-2792-    Architecture
--
packages/core/legoesm/parallel/latlon_mpi.py-2794-    1. ``set_halo_backend("mpi", layout)`` is activated once at
packages/core/legoesm/parallel/latlon_mpi.py-2795-       factory time.  Every subsequent ``pad_halo_latlon*``,
packages/core/legoesm/parallel/latlon_mpi.py-2796-       ``pad_ns_zero``, and ``pad_with_pole_bc_lat`` call inside the
packages/core/legoesm/parallel/latlon_mpi.py:2797:       dycore operators dispatches through the MPI sendrecv path.
packages/core/legoesm/parallel/latlon_mpi.py-2798-       ``is_distributed()`` returns True, which gates the
packages/core/legoesm/parallel/latlon_mpi.py-2799-       allreduce inside ``batch_global_area_sums`` and
packages/core/legoesm/parallel/latlon_mpi.py-2800-       ``zero_mean_tendency`` (the mass-fixer reductions).
--
packages/core/legoesm/parallel/latlon_mpi.py-2898-
packages/core/legoesm/parallel/latlon_mpi.py-2899-    # Activate the MPI halo backend.  Every pad_halo_latlon /
packages/core/legoesm/parallel/latlon_mpi.py-2900-    # pad_with_pole_bc_lat call inside the dycore now dispatches
packages/core/legoesm/parallel/latlon_mpi.py:2901:    # through inter-rank sendrecv at partition cuts and constant /
packages/core/legoesm/parallel/latlon_mpi.py-2902-    # pole-fold pads at boundary ranks.  This also flips
packages/core/legoesm/parallel/latlon_mpi.py-2903-    # ``is_distributed()`` so the mass-fixer reductions allreduce.
packages/core/legoesm/parallel/latlon_mpi.py-2904-    set_halo_backend("mpi", layout)
--
packages/core/legoesm/parallel/latlon_mpi.py-2976-    def step_fn(local_state, dt, *, target_mass=None):
packages/core/legoesm/parallel/latlon_mpi.py-2977-        # No pre-pad, no strip.  The operators inside ``_step_cgrid``
packages/core/legoesm/parallel/latlon_mpi.py-2978-        # call backend-aware halo helpers that, under
packages/core/legoesm/parallel/latlon_mpi.py:2979:        # ``_halo_backend == "mpi"``, MPI-sendrecv with neighbouring
packages/core/legoesm/parallel/latlon_mpi.py-2980-        # ranks for halo cells at partition cuts and apply the
packages/core/legoesm/parallel/latlon_mpi.py-2981-        # serial pole-fold / wall-BC constants at boundary ranks.
packages/core/legoesm/parallel/latlon_mpi.py-2982-        # The mass fixer's allreduce fires through the same
--
packages/core/legoesm/parallel/latlon_mpi.py-3010-
packages/core/legoesm/parallel/latlon_mpi.py-3011-def make_latlon_2d_mpi_step(
packages/core/legoesm/parallel/latlon_mpi.py-3012-    model,
packages/core/legoesm/parallel/latlon_mpi.py:3013:    layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-3014-    *,
packages/core/legoesm/parallel/latlon_mpi.py-3015-    physics_fn: Callable | None = None,
packages/core/legoesm/parallel/latlon_mpi.py-3016-) -> Callable:
--
packages/core/legoesm/parallel/latlon_mpi.py-3018-
packages/core/legoesm/parallel/latlon_mpi.py-3019-    The 2-D ``(proc_lat × proc_lon)`` analogue of :func:`make_latlon_mpi_step`.
packages/core/legoesm/parallel/latlon_mpi.py-3020-    Identical architecture — activate the MPI halo backend once with the
packages/core/legoesm/parallel/latlon_mpi.py:3021:    :class:`LatLon2DLayout`, build a rank-local model with the global
packages/core/legoesm/parallel/latlon_mpi.py-3022-    sphere area (mass-fixer divisor) and ``pole_v_bc`` tracking which lat
packages/core/legoesm/parallel/latlon_mpi.py-3023-    ends touch a physical pole, then delegate each step to
packages/core/legoesm/parallel/latlon_mpi.py-3024-    ``model._step_cgrid`` whose backend-aware operators fetch halo data via
packages/core/legoesm/parallel/latlon_mpi.py-3025-    the 2-D dispatch (``pad_halo_latlon`` → :func:`pad_halo_latlon_2d`,
packages/core/legoesm/parallel/latlon_mpi.py:3026:    ``pad_with_pole_bc_lat`` → :func:`pad_with_pole_bc_lat_2d`,
packages/core/legoesm/parallel/latlon_mpi.py-3027-    ``zero_polar_lat_ends`` → the 2-D pole-touch test).
packages/core/legoesm/parallel/latlon_mpi.py-3028-
packages/core/legoesm/parallel/latlon_mpi.py-3029-    Wall poles only — a labeled midlatitude throughput benchmark, NOT the
--
packages/core/legoesm/parallel/distributed.py-63-    reductions run through pure-JAX ``ppermute``/``psum`` inside ``shard_map``,
packages/core/legoesm/parallel/distributed.py-64-    NOT mpi4jax.  So this helper requires mpi4py only, unlike
packages/core/legoesm/parallel/distributed.py-65-    :func:`legoesm.parallel.reductions.require_mpi_stack`
packages/core/legoesm/parallel/distributed.py:66:    (which also requires mpi4jax for the cubed-sphere ``sendrecv`` halo).
packages/core/legoesm/parallel/distributed.py-67-    """
packages/core/legoesm/parallel/distributed.py-68-    import importlib.util
packages/core/legoesm/parallel/distributed.py-69-
--
packages/core/legoesm/parallel/distributed.py-454-    SCVT/cubed-sphere grids — separate because the lat-lon path
packages/core/legoesm/parallel/distributed.py-455-    needs neither the face-topology dance nor JAX's device-mesh
packages/core/legoesm/parallel/distributed.py-456-    SPMD machinery.  All the lat-lon MPI work happens through
packages/core/legoesm/parallel/distributed.py:457:    mpi4jax sendrecv (see
packages/core/legoesm/parallel/distributed.py-458-    :mod:`legoesm.parallel.latlon_mpi`) and the backend-dispatched
packages/core/legoesm/parallel/distributed.py-459-    halo helpers (see :mod:`legoesm.grids.halo_latlon`).
packages/core/legoesm/parallel/distributed.py-460-
--
packages/core/legoesm/parallel/distributed.py-465-       owns.
packages/core/legoesm/parallel/distributed.py-466-    3. Activates ``set_halo_backend("mpi", layout)`` — every
packages/core/legoesm/parallel/distributed.py-467-       subsequent ``pad_halo_latlon`` / ``pad_with_pole_bc_lat``
packages/core/legoesm/parallel/distributed.py:468:       call inside the dycore dispatches through MPI sendrecv at
packages/core/legoesm/parallel/distributed.py-469-       partition cuts + pole-fold / wall-BC constants at boundary
packages/core/legoesm/parallel/distributed.py-470-       ranks, and ``is_distributed()`` returns True (gating
packages/core/legoesm/parallel/distributed.py-471-       conservation reductions).
--
tests/distributed/test_cube_face_scatter_mpi.py-352-    lifts the end-to-end gate (``..._gradient_matches_reference``) to ~1e-6, so
tests/distributed/test_cube_face_scatter_mpi.py-353-    this gate certifies the substep's OWN VJPs are exact:
tests/distributed/test_cube_face_scatter_mpi.py-354-
tests/distributed/test_cube_face_scatter_mpi.py:355:    * the 4D ``pad_halo_mpi_4d`` cross-face ``sendrecv`` (``_sendrecv_vjp``);
tests/distributed/test_cube_face_scatter_mpi.py-356-    * ``_conserving_rescale``'s global mass reduction — which feeds the SHARED
tests/distributed/test_cube_face_scatter_mpi.py-357-      ``scale`` that rescales every face, so its reduction MUST use the broadcast
tests/distributed/test_cube_face_scatter_mpi.py-358-      VJP (``global_face_sum_if_scattered(..., differentiable_broadcast=True)``);
--
tests/distributed/test_mpi_fv3_step_fidelity.py-276-        # (or the conftest ``cube_face_layout`` fixture, at
tests/distributed/test_mpi_fv3_step_fidelity.py-277-        # ``global_n=max(n_procs, 2)``) may have left the module-level
tests/distributed/test_mpi_fv3_step_fidelity.py-278-        # halo backend as ``"mpi"`` at test start.  Without resetting
tests/distributed/test_mpi_fv3_step_fidelity.py:279:        # first, the MPI sendrecv runs with a stale topology
tests/distributed/test_mpi_fv3_step_fidelity.py-280-        # (set up for ``global_n=2``) on
tests/distributed/test_mpi_fv3_step_fidelity.py-281-        # ``(6, 8, 8, nlev)`` data → mis-sized halo strips → corrupt
tests/distributed/test_mpi_fv3_step_fidelity.py-282-        # ``state_global`` → single-rank reference blew up to ``1e+53``
--
tests/distributed/test_mpi_fv3_step_fidelity.py-327-        PE counterpart to NH ``test_nh_3_step_with_corner_div_damp``.
tests/distributed/test_mpi_fv3_step_fidelity.py-328-        Exercises the iter-1044 4D-native ``fv3_corner_laplacian_iteration``
tests/distributed/test_mpi_fv3_step_fidelity.py-329-        + the iter-1044 4D-native ``fv3_divergence_corner_3d`` (no vmap
tests/distributed/test_mpi_fv3_step_fidelity.py:330:        around any pad_halo / sendrecv).
tests/distributed/test_mpi_fv3_step_fidelity.py-331-        """
tests/distributed/test_mpi_fv3_step_fidelity.py-332-        rank = MPI.COMM_WORLD.Get_rank()
tests/distributed/test_mpi_fv3_step_fidelity.py-333-        size = MPI.COMM_WORLD.Get_size()
--
tests/distributed/test_mpi_sw_sync.py-2-
tests/distributed/test_mpi_sw_sync.py-3-FV3_3D iter-1052 ported the SW corner-staggered edge + vertex sync
tests/distributed/test_mpi_sw_sync.py-4-to MPI (iter-1050 audit found this as an open gap).  The MPI variant
tests/distributed/test_mpi_sw_sync.py:5:uses batched-per-peer ``sendrecv`` for edge sync and
tests/distributed/test_mpi_sw_sync.py-6-``allreduce(SUM)`` for the 8 cube-vertex broadcasts.
tests/distributed/test_mpi_sw_sync.py-7-
tests/distributed/test_mpi_sw_sync.py-8-Run with::
--
tests/distributed/test_mpi_sw_sync.py-117-        exercised with DISTINCT values per (face, edge).
tests/distributed/test_mpi_sw_sync.py-118-
tests/distributed/test_mpi_sw_sync.py-119-        The baseline test uses random init, but if any cube-edge
tests/distributed/test_mpi_sw_sync.py:120:        sendrecv silently picks up the wrong neighbour strip (e.g.,
tests/distributed/test_mpi_sw_sync.py-121-        ``is_reversed`` flag drop, swapped face indices), randomness
tests/distributed/test_mpi_sw_sync.py-122-        could still pass.  This test fills each face's u_d, v_d with
tests/distributed/test_mpi_sw_sync.py-123-        ``(face_index + 1) * 1000`` so the post-sync owned-face
--
packages/core/legoesm/parallel/voronoi_partition.py-457-
packages/core/legoesm/parallel/voronoi_partition.py-458-    The per-neighbor message sequence stays in sorted-rank order on every
packages/core/legoesm/parallel/voronoi_partition.py-459-    rank, exactly like the entity schedules it replaces, so the blocking
packages/core/legoesm/parallel/voronoi_partition.py:460:    ``sendrecv`` pairing properties of the existing exchange carry over
packages/core/legoesm/parallel/voronoi_partition.py-461-    unchanged.
packages/core/legoesm/parallel/voronoi_partition.py-462-    """
packages/core/legoesm/parallel/voronoi_partition.py-463-
--
packages/core/legoesm/parallel/voronoi_partition.py-674-    # holds an edge/vertex I own in its halo — without its cell-halo reaching my
packages/core/legoesm/parallel/voronoi_partition.py-675-    # cells, so it is absent from `neighbor_ranks` (= cell-recv owners) and the
packages/core/legoesm/parallel/voronoi_partition.py-676-    # cell-neighbour-only `cell_to_nbr` would never mark it as needing that edge:
packages/core/legoesm/parallel/voronoi_partition.py:677:    # I would not send, its blocking sendrecv to me would hang.  This is the
packages/core/legoesm/parallel/voronoi_partition.py-678-    # np>=64 multi-node deadlock (asymmetric edge schedule; cells were fine).
packages/core/legoesm/parallel/voronoi_partition.py-679-    # An edge/vertex spans exactly one cell-ring beyond the cell halo, so owners
packages/core/legoesm/parallel/voronoi_partition.py-680-    # of cells within (halo_depth + 1) rings of my owned cells are a provably
--
packages/core/legoesm/parallel/async_halo.py-7-    4. Compute boundary points (needs halo)
packages/core/legoesm/parallel/async_halo.py-8-    5. Merge interior + boundary results
packages/core/legoesm/parallel/async_halo.py-9-
packages/core/legoesm/parallel/async_halo.py:10:Since ``mpi4jax`` only exposes blocking ``sendrecv`` (no ``isend``/``irecv``),
packages/core/legoesm/parallel/async_halo.py-11-true non-blocking overlap at the MPI level is not available.  Instead, we
packages/core/legoesm/parallel/async_halo.py-12-implement the overlap at a *higher level*:
packages/core/legoesm/parallel/async_halo.py-13-
--
packages/core/legoesm/parallel/async_halo.py-924-    )
packages/core/legoesm/parallel/async_halo.py-925-
packages/core/legoesm/parallel/async_halo.py-926-    # ------------------------------------------------------------------
packages/core/legoesm/parallel/async_halo.py:927:    # Step 2: MPI halo exchange (synchronous sendrecv).
packages/core/legoesm/parallel/async_halo.py-928-    # ------------------------------------------------------------------
packages/core/legoesm/parallel/async_halo.py-929-    padded_mpi = start_halo_exchange(state, topology, halo_width=h)
packages/core/legoesm/parallel/async_halo.py-930-
--
tests/distributed/test_latlon_2d_dispatch_mpi.py-1-"""MPI parity + AD for the lat-lon 2-D pencil DISPATCH + the lat-only wall
tests/distributed/test_latlon_2d_dispatch_mpi.py:2:pad (``pad_with_pole_bc_lat_2d``).
tests/distributed/test_latlon_2d_dispatch_mpi.py-3-
tests/distributed/test_latlon_2d_dispatch_mpi.py-4-Two layers, both run under ``mpirun -np {2,3,6} python -m pytest <thisfile>``
tests/distributed/test_latlon_2d_dispatch_mpi.py-5-(wired into the MPI CI workflow next to ``test_latlon_2d_pad_wall_mpi.py``):
tests/distributed/test_latlon_2d_dispatch_mpi.py-6-
tests/distributed/test_latlon_2d_dispatch_mpi.py:7:1. ``pad_with_pole_bc_lat_2d`` — the lat-axis-ONLY wall pad (the 2-D twin of
tests/distributed/test_latlon_2d_dispatch_mpi.py-8-   the band ``pad_with_pole_bc_lat`` MPI path; longitude is NOT padded).  The
tests/distributed/test_latlon_2d_dispatch_mpi.py-9-   full lat+lon ``pad_halo_latlon_2d`` is already MPI-tested; this pins the
tests/distributed/test_latlon_2d_dispatch_mpi.py:10:   lat-only variant (interior-cut sendrecv + pole wall, lon untouched) +
tests/distributed/test_latlon_2d_dispatch_mpi.py:11:   its sendrecv VJP.
tests/distributed/test_latlon_2d_dispatch_mpi.py:12:2. The BACKEND DISPATCH: with ``set_halo_backend("mpi", LatLon2DLayout)``
tests/distributed/test_latlon_2d_dispatch_mpi.py-13-   armed, the public ``halo_latlon`` entry points must route to the 2-D
tests/distributed/test_latlon_2d_dispatch_mpi.py-14-   primitives under a REAL distributed layout (the 1×1 serial lane in
tests/distributed/test_latlon_2d_dispatch_mpi.py-15-   ``tests/parallel/test_latlon_2d_dispatch_serial.py`` cannot exercise the
tests/distributed/test_latlon_2d_dispatch_mpi.py:16:   sendrecv branch / deadlock-freedom).
tests/distributed/test_latlon_2d_dispatch_mpi.py-17-
tests/distributed/test_latlon_2d_dispatch_mpi.py-18-Single rank => skip (the serial routing is the parallel-lane file).
tests/distributed/test_latlon_2d_dispatch_mpi.py-19-Size-adaptive process grid (1×n ring for primes, balanced 2-D for
--
tests/distributed/test_latlon_2d_dispatch_mpi.py-46-    make_latlon_band_layout,
tests/distributed/test_latlon_2d_dispatch_mpi.py-47-    pad_halo_latlon_2d,
tests/distributed/test_latlon_2d_dispatch_mpi.py-48-    pad_halo_latlon_mpi,
tests/distributed/test_latlon_2d_dispatch_mpi.py:49:    pad_with_pole_bc_lat_2d,
tests/distributed/test_latlon_2d_dispatch_mpi.py-50-    scatter_field_latlon_2d,
tests/distributed/test_latlon_2d_dispatch_mpi.py-51-    scatter_state_latlon_2d,
tests/distributed/test_latlon_2d_dispatch_mpi.py-52-)
--
tests/distributed/test_latlon_2d_dispatch_mpi.py-65-
tests/distributed/test_latlon_2d_dispatch_mpi.py-66-def _serial_lat_wall(g, halo):
tests/distributed/test_latlon_2d_dispatch_mpi.py-67-    """lat-axis-only wall pad (NO lon pad) — the serial twin of
tests/distributed/test_latlon_2d_dispatch_mpi.py:68:    ``pad_with_pole_bc_lat_2d``."""
tests/distributed/test_latlon_2d_dispatch_mpi.py-69-    n_lon = g.shape[1]
tests/distributed/test_latlon_2d_dispatch_mpi.py-70-    sw = np.full((halo, n_lon) + g.shape[2:], SV, g.dtype)
tests/distributed/test_latlon_2d_dispatch_mpi.py-71-    nw = np.full((halo, n_lon) + g.shape[2:], NV, g.dtype)
--
tests/distributed/test_latlon_2d_dispatch_mpi.py-92-    g = np.random.default_rng(11).standard_normal((n_lat, n_lon))
tests/distributed/test_latlon_2d_dispatch_mpi.py-93-    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
tests/distributed/test_latlon_2d_dispatch_mpi.py-94-    local = scatter_field_latlon_2d(jnp.asarray(g), L)
tests/distributed/test_latlon_2d_dispatch_mpi.py:95:    out = np.asarray(pad_with_pole_bc_lat_2d(
tests/distributed/test_latlon_2d_dispatch_mpi.py-96-        local, L, halo=h, south_value=SV, north_value=NV))
tests/distributed/test_latlon_2d_dispatch_mpi.py-97-    ref = _serial_lat_wall(g, h)[
tests/distributed/test_latlon_2d_dispatch_mpi.py-98-        L.lat_start:L.lat_end + 2 * h, L.lon_start:L.lon_end]
--
tests/distributed/test_latlon_2d_dispatch_mpi.py-106-
tests/distributed/test_latlon_2d_dispatch_mpi.py-107-
tests/distributed/test_latlon_2d_dispatch_mpi.py-108-def test_lat_only_wall_pad_grad_parity():
tests/distributed/test_latlon_2d_dispatch_mpi.py:109:    """Reverse-mode AD through the lat-axis sendrecv VJP.  Loss is
tests/distributed/test_latlon_2d_dispatch_mpi.py-110-    ``global_sum_mpi(sum(pad_local**2))`` so each neighbour's ghost-loss
tests/distributed/test_latlon_2d_dispatch_mpi.py-111-    cotangent flows BACK through the halo VJP into this rank's owned cells
tests/distributed/test_latlon_2d_dispatch_mpi.py-112-    (the same construction as ``test_latlon_2d_pad_wall_mpi`` — without the
--
tests/distributed/test_latlon_2d_dispatch_mpi.py-123-    local = scatter_field_latlon_2d(jnp.asarray(g), L)
tests/distributed/test_latlon_2d_dispatch_mpi.py-124-
tests/distributed/test_latlon_2d_dispatch_mpi.py-125-    def loss(x):
tests/distributed/test_latlon_2d_dispatch_mpi.py:126:        local_loss = jnp.sum(pad_with_pole_bc_lat_2d(
tests/distributed/test_latlon_2d_dispatch_mpi.py-127-            x, L, halo=h, south_value=SV, north_value=NV) ** 2)
tests/distributed/test_latlon_2d_dispatch_mpi.py-128-        return global_sum_mpi(local_loss, comm)
tests/distributed/test_latlon_2d_dispatch_mpi.py-129-
--
tests/distributed/test_latlon_2d_dispatch_mpi.py-153-def test_dispatch_routes_to_2d_under_mpi():
tests/distributed/test_latlon_2d_dispatch_mpi.py-154-    """With the 2-D layout armed as the MPI backend, the public dispatch
tests/distributed/test_latlon_2d_dispatch_mpi.py-155-    entry points must produce the SAME result as calling the 2-D primitive
tests/distributed/test_latlon_2d_dispatch_mpi.py:156:    directly — i.e. the ``LatLon2DLayout`` branch fires (not the serial
tests/distributed/test_latlon_2d_dispatch_mpi.py:157:    pole-fold / band path) AND its sendrecvs do not deadlock."""
tests/distributed/test_latlon_2d_dispatch_mpi.py-158-    comm = MPI.COMM_WORLD
tests/distributed/test_latlon_2d_dispatch_mpi.py-159-    rank, n = comm.Get_rank(), comm.Get_size()
tests/distributed/test_latlon_2d_dispatch_mpi.py-160-    if n < 2:
--
tests/distributed/test_latlon_2d_dispatch_mpi.py-168-
tests/distributed/test_latlon_2d_dispatch_mpi.py-169-    # Direct primitives (the reference the dispatch must reproduce).
tests/distributed/test_latlon_2d_dispatch_mpi.py-170-    want_full = np.asarray(pad_halo_latlon_2d(local, L, halo=h, pole_bc="wall"))
tests/distributed/test_latlon_2d_dispatch_mpi.py:171:    want_lat = np.asarray(pad_with_pole_bc_lat_2d(local, L, halo=h))
tests/distributed/test_latlon_2d_dispatch_mpi.py-172-
tests/distributed/test_latlon_2d_dispatch_mpi.py-173-    set_halo_backend("mpi", L)
tests/distributed/test_latlon_2d_dispatch_mpi.py-174-    try:
--
tests/distributed/test_latlon_2d_dispatch_mpi.py-192-    ``pad_halo_latlon`` dispatch must reuse the validated band POLE-FOLD —
tests/distributed/test_latlon_2d_dispatch_mpi.py-193-    NOT the wall-pole pad.  Compared bit-for-bit against ``pad_halo_latlon_mpi``
tests/distributed/test_latlon_2d_dispatch_mpi.py-194-    on the equivalent ``LatLonBandLayout`` (the path the 2-D dispatch delegates
tests/distributed/test_latlon_2d_dispatch_mpi.py:195:    to).  Exercises the real lat sendrecv at interior cuts AND the local
tests/distributed/test_latlon_2d_dispatch_mpi.py-196-    180-deg fold at pole-touching ranks under proc_lat>1 MPI."""
tests/distributed/test_latlon_2d_dispatch_mpi.py-197-    comm = MPI.COMM_WORLD
tests/distributed/test_latlon_2d_dispatch_mpi.py-198-    rank, n = comm.Get_rank(), comm.Get_size()
--
tests/distributed/test_latlon_2d_dispatch_mpi.py-207-    band_ref = make_latlon_band_layout(rank, pr, n_lat, n_lon)
tests/distributed/test_latlon_2d_dispatch_mpi.py-208-
tests/distributed/test_latlon_2d_dispatch_mpi.py-209-    # Reference: the validated band fold on the equivalent band layout
tests/distributed/test_latlon_2d_dispatch_mpi.py:210:    # (computed BEFORE arming the 2-D backend; both legs sendrecv in the
tests/distributed/test_latlon_2d_dispatch_mpi.py-211-    # same SPMD-symmetric order so there is no deadlock).
tests/distributed/test_latlon_2d_dispatch_mpi.py-212-    want = np.asarray(pad_halo_latlon_mpi(local, band_ref, halo=h))
tests/distributed/test_latlon_2d_dispatch_mpi.py-213-
--
tests/distributed/test_mpi_differentiability.py-7-
tests/distributed/test_mpi_differentiability.py-8-Verifies that ``jax.grad`` flows correctly through:
tests/distributed/test_mpi_differentiability.py-9-- ``global_sum_mpi`` (allreduce with SUM)
tests/distributed/test_mpi_differentiability.py:10:- ``pad_halo`` with MPI backend (sendrecv with custom_vjp)
tests/distributed/test_mpi_differentiability.py-11-- ``pad_halo_4d`` with MPI backend
tests/distributed/test_mpi_differentiability.py-12-- Conservation fixers (``fix_mass_hydrostatic``) under MPI
tests/distributed/test_mpi_differentiability.py-13-- The Voronoi/MPAS halo exchange on BOTH state-exchange paths: the
tests/distributed/test_mpi_differentiability.py-14-  DEFAULT ``batched_halo_exchange`` union-neighbor path and the opt-OUT
tests/distributed/test_mpi_differentiability.py-15-  per-entity path (``VoronoiHaloExchange`` -> ``_exchange_mpi``) (both
tests/distributed/test_mpi_differentiability.py:16:  sendrecv with custom_vjp; per-neighbor send/recv counts differ,
tests/distributed/test_mpi_differentiability.py-17-  exercising the asymmetric-shape backward template)
tests/distributed/test_mpi_differentiability.py-18-
tests/distributed/test_mpi_differentiability.py-19-The single-process equivalents are already tested in:
--
tests/distributed/test_mpi_differentiability.py-105-
tests/distributed/test_mpi_differentiability.py-106-
tests/distributed/test_mpi_differentiability.py-107-class TestPadHaloMPIGrad:
tests/distributed/test_mpi_differentiability.py:108:    """Gradient through MPI halo exchange (sendrecv with custom_vjp)."""
tests/distributed/test_mpi_differentiability.py-109-
tests/distributed/test_mpi_differentiability.py-110-    def test_pad_halo_mpi_grad_finite(self, topology):
tests/distributed/test_mpi_differentiability.py-111-        """jax.grad through pad_halo with MPI backend produces finite results."""
--
tests/distributed/test_mpi_differentiability.py-129-        ``(6, n, n)`` input on every rank, the MPI ``pad_halo``
tests/distributed/test_mpi_differentiability.py-130-        forward fills each rank's local-face interior from its own
tests/distributed/test_mpi_differentiability.py-131-        data and the cross-face halos from neighboring ranks via
tests/distributed/test_mpi_differentiability.py:132:        ``mpi4jax.sendrecv``.  The backward VJP propagates cotangents
tests/distributed/test_mpi_differentiability.py:133:        through ``sendrecv`` (via ``_sendrecv_vjp`` custom_vjp), but
tests/distributed/test_mpi_differentiability.py-134-        does not allreduce the upstream cotangent — each rank's
tests/distributed/test_mpi_differentiability.py-135-        gradient at a given face position is correct ONLY for the
tests/distributed/test_mpi_differentiability.py-136-        faces it owns.
--
tests/distributed/test_mpi_differentiability.py-226-        per-rank MPI gradient must recover the full local-backend
tests/distributed/test_mpi_differentiability.py-227-        gradient on ALL 6 faces.
tests/distributed/test_mpi_differentiability.py-228-
tests/distributed/test_mpi_differentiability.py:229:        Without this stronger test, a bug in ``_sendrecv_vjp`` that
tests/distributed/test_mpi_differentiability.py-230-        zeros out non-owned-face cotangents (or fails to propagate
tests/distributed/test_mpi_differentiability.py-231-        them via MPI) would pass ``test_pad_halo_mpi_grad_matches_local``
tests/distributed/test_mpi_differentiability.py-232-        (which only inspects owned faces) but corrupt the canonical
--
tests/distributed/test_mpi_differentiability.py-264-            err_msg=(
tests/distributed/test_mpi_differentiability.py-265-                "Allreduced per-rank MPI gradient does not recover the "
tests/distributed/test_mpi_differentiability.py-266-                "full local-backend gradient.  Likely cause: "
tests/distributed/test_mpi_differentiability.py:267:                "_sendrecv_vjp drops or mis-routes non-owned-face "
tests/distributed/test_mpi_differentiability.py-268-                "cotangents."
tests/distributed/test_mpi_differentiability.py-269-            ),
tests/distributed/test_mpi_differentiability.py-270-        )
--
tests/distributed/test_mpi_differentiability.py-318-
tests/distributed/test_mpi_differentiability.py-319-    - ``per_entity_legacy`` — the opt-OUT model-step path
tests/distributed/test_mpi_differentiability.py-320-      (``VoronoiHaloExchange`` -> ``_exchange_mpi`` ->
tests/distributed/test_mpi_differentiability.py:321:      ``get_sendrecv_vjp``), composed exactly as the legacy
tests/distributed/test_mpi_differentiability.py-322-      ``_exchange_mpas_state`` packs fields (u edge; T+p_s packed cell;
tests/distributed/test_mpi_differentiability.py-323-      tracers stacked cell);
tests/distributed/test_mpi_differentiability.py-324-    - ``batched`` — the DEFAULT union-neighbor exchange
tests/distributed/test_mpi_differentiability.py-325-      (``batched_halo_exchange``), called as a direct closure (no env
tests/distributed/test_mpi_differentiability.py-326-      var needed).  Its UNEQUAL per-neighbor send/recv counts also gate
tests/distributed/test_mpi_differentiability.py:327:      the asymmetric-shape backward recv template in ``_sendrecv_vjp``
tests/distributed/test_mpi_differentiability.py-328-      (cotangents must come back SEND-shaped, not recv-shaped).
tests/distributed/test_mpi_differentiability.py-329-
tests/distributed/test_mpi_differentiability.py-330-    Real gate: ``mpirun -np 2`` (and ``-np 3``/``-np 6``).  At np=1 the
--
tests/distributed/test_mpi_differentiability.py-367-        over all partitions, so the serial gradient at global entity e is
tests/distributed/test_mpi_differentiability.py-368-        ``2*x[e] * (#partitions whose local set contains e)``.  Under MPI
tests/distributed/test_mpi_differentiability.py-369-        the extra copies live on neighbor ranks, and their cotangents
tests/distributed/test_mpi_differentiability.py:370:        reach the owner ONLY through the sendrecv VJP — a dropped or
tests/distributed/test_mpi_differentiability.py-371-        mis-routed backward message fails this test.  Halo-row gradients
tests/distributed/test_mpi_differentiability.py-372-        must be exactly zero (the forward overwrites every halo row on
tests/distributed/test_mpi_differentiability.py-373-        both paths: per-entity and union recv_idx cover the same rows).
--
tests/distributed/test_mpi_differentiability.py-393-            # ``voronoi_mpi._exchange_mpas_state`` does — u edge exchange;
tests/distributed/test_mpi_differentiability.py-394-            # T+p_s packed into ONE cell exchange; tracers stacked into
tests/distributed/test_mpi_differentiability.py-395-            # one further cell exchange — through ``VoronoiHaloExchange``
tests/distributed/test_mpi_differentiability.py:396:            # -> ``_exchange_mpi`` -> ``get_sendrecv_vjp``.
tests/distributed/test_mpi_differentiability.py-397-            halo = VoronoiHaloExchange(part, backend="mpi")
tests/distributed/test_mpi_differentiability.py-398-
tests/distributed/test_mpi_differentiability.py-399-            def _exchange(u, T, ps, qv):
--
tests/distributed/test_mpi_differentiability.py-471-                atol=1e-12,
tests/distributed/test_mpi_differentiability.py-472-                err_msg=(
tests/distributed/test_mpi_differentiability.py-473-                    f"{name}: owned-row MPI gradient != serial gradient "
tests/distributed/test_mpi_differentiability.py:474:                    f"(rank {rank}) — sendrecv VJP dropped/mis-routed a "
tests/distributed/test_mpi_differentiability.py-475-                    f"halo cotangent"
tests/distributed/test_mpi_differentiability.py-476-                ),
tests/distributed/test_mpi_differentiability.py-477-            )
--
tests/distributed/test_mpi_differentiability.py-485-
tests/distributed/test_mpi_differentiability.py-486-        # Production composition: the real step traces the exchange under
tests/distributed/test_mpi_differentiability.py-487-        # @jax.jit, so gate jit(grad(...)) too — XLA must compile the
tests/distributed/test_mpi_differentiability.py:488:        # custom_vjp sendrecv pair (fwd+bwd) into one program on every
tests/distributed/test_mpi_differentiability.py-489-        # rank without reordering the matched message sequence.
tests/distributed/test_mpi_differentiability.py-490-        g_jit = jax.jit(jax.grad(mpi_loss, argnums=(0, 1, 2, 3)))(
tests/distributed/test_mpi_differentiability.py-491-            u_l, T_l, ps_l, qv_l)
--
tests/distributed/test_mpi_differentiability.py-518-        qv = jnp.ones((part.n_local_cells, nlev), dtype=jnp.float64)
tests/distributed/test_mpi_differentiability.py-519-
tests/distributed/test_mpi_differentiability.py-520-        # Message-count instrumentation (counter increments per posted
tests/distributed/test_mpi_differentiability.py:521:        # sendrecv when running eagerly).
tests/distributed/test_mpi_differentiability.py-522-        reset_halo_message_count()
tests/distributed/test_mpi_differentiability.py-523-        _ = batched_halo_exchange((u,), (T, ps, qv), sched, rank)
tests/distributed/test_mpi_differentiability.py-524-        posted = get_halo_message_count()
--
tests/distributed/test_latlon_2d_operators_mpi.py-8-(local-backend) operator on the global field — the same per-rank-window
tests/distributed/test_latlon_2d_operators_mpi.py-9-strategy as ``test_latlon_2d_pad_wall_mpi``, so no face-field gather is
tests/distributed/test_latlon_2d_operators_mpi.py-10-needed.  A genuine longitude split (proc_lon>1 for np={2,3,6}) drives the
tests/distributed/test_latlon_2d_operators_mpi.py:11:``exchange_halo_lon`` ring; matching the serial window proves the local
tests/distributed/test_latlon_2d_operators_mpi.py-12-``jnp.roll`` wrap was correctly replaced (and stays bit-identical where lon
tests/distributed/test_latlon_2d_operators_mpi.py-13-is full).
tests/distributed/test_latlon_2d_operators_mpi.py-14-
tests/distributed/test_latlon_2d_operators_mpi.py-15-The serial reference is computed FIRST on every rank under the LOCAL backend
tests/distributed/test_latlon_2d_operators_mpi.py-16-(deterministic, collective-free) BEFORE arming the 2-D MPI backend — a serial
tests/distributed/test_latlon_2d_operators_mpi.py:17:op traced after arming would embed sendrecvs no other rank matches and
tests/distributed/test_latlon_2d_operators_mpi.py-18-deadlock (same ordering contract as ``test_latlon_mpi_step``).
tests/distributed/test_latlon_2d_operators_mpi.py-19-
tests/distributed/test_latlon_2d_operators_mpi.py-20-Run: ``mpirun -np {2,3,6} python -m pytest <thisfile>``.
--
tests/distributed/test_latlon_2d_operators_mpi.py-168-
tests/distributed/test_latlon_2d_operators_mpi.py-169-
tests/distributed/test_latlon_2d_operators_mpi.py-170-def test_2d_gradient_x_grad_finite():
tests/distributed/test_latlon_2d_operators_mpi.py:171:    """Reverse-mode AD through the lon-ring exchange inside gradient_x stays
tests/distributed/test_latlon_2d_operators_mpi.py-172-    finite under a real 2-D split (the exchange_halo_lon VJP is exercised via
tests/distributed/test_latlon_2d_operators_mpi.py-173-    the operator)."""
tests/distributed/test_latlon_2d_operators_mpi.py-174-    comm = MPI.COMM_WORLD
--
tests/distributed/test_latlon_2d_operators_mpi.py-188-
tests/distributed/test_latlon_2d_operators_mpi.py-189-    def loss(fb):
tests/distributed/test_latlon_2d_operators_mpi.py-190-        # global_sum_mpi(sum**2) so neighbour cotangents flow back through the
tests/distributed/test_latlon_2d_operators_mpi.py:191:        # lon-ring VJP into this rank's owned cells (else grad collapses to a
tests/distributed/test_latlon_2d_operators_mpi.py-192-        # local form — same construction as the pad-wall AD test).
tests/distributed/test_latlon_2d_operators_mpi.py-193-        return global_sum_mpi(jnp.sum(gradient_x_cgrid(fb, gblk) ** 2), comm)
tests/distributed/test_latlon_2d_operators_mpi.py-194-
--
tests/distributed/test_mpi_synchronize_cgrid_fluxes.py-2-
tests/distributed/test_mpi_synchronize_cgrid_fluxes.py-3-iter-1049 made ``synchronize_cgrid_fluxes`` MPI-aware: under
tests/distributed/test_mpi_synchronize_cgrid_fluxes.py-4-``_halo_backend == "mpi"`` it dispatches to
tests/distributed/test_mpi_synchronize_cgrid_fluxes.py:5:``_synchronize_cgrid_fluxes_mpi`` which uses ``mpi4jax.sendrecv``
tests/distributed/test_mpi_synchronize_cgrid_fluxes.py-6-to swap boundary flux strips across ranks before averaging.
tests/distributed/test_mpi_synchronize_cgrid_fluxes.py-7-
tests/distributed/test_mpi_synchronize_cgrid_fluxes.py-8-This test verifies that the MPI path produces bit-for-bit
tests/distributed/test_mpi_synchronize_cgrid_fluxes.py-9-identical owned-face boundary fluxes vs the single-device path.
tests/distributed/test_mpi_synchronize_cgrid_fluxes.py:10:A regression in the sendrecv tag scheme, reversal logic, or
tests/distributed/test_mpi_synchronize_cgrid_fluxes.py-11-sign-flip table would cause the MPI averages to drift.
tests/distributed/test_mpi_synchronize_cgrid_fluxes.py-12-
tests/distributed/test_mpi_synchronize_cgrid_fluxes.py-13-Run with::
--
tests/distributed/test_plane_slow_tend_halo_mpi.py-3-Splits a global state across ranks, runs halo-aware slow tendency
tests/distributed/test_plane_slow_tend_halo_mpi.py-4-locally, gathers, compares to single-process original on the SAME
tests/distributed/test_plane_slow_tend_halo_mpi.py-5-global state. Catches halo-edge bugs (off-by-one slicing, wrong
tests/distributed/test_plane_slow_tend_halo_mpi.py:6:direction in sendrecv, etc.).
tests/distributed/test_plane_slow_tend_halo_mpi.py-7-"""
tests/distributed/test_plane_slow_tend_halo_mpi.py-8-
tests/distributed/test_plane_slow_tend_halo_mpi.py-9-from __future__ import annotations
--
packages/core/legoesm/parallel/halo_exchange.py-32-Optimization
packages/core/legoesm/parallel/halo_exchange.py-33-------------
packages/core/legoesm/parallel/halo_exchange.py-34-Face-only mode: edges destined for the same neighbor rank are packed
packages/core/legoesm/parallel/halo_exchange.py:35:into a single send buffer and exchanged via one sendrecv per neighbor
packages/core/legoesm/parallel/halo_exchange.py-36-(at most 4 point-to-point messages per rank).
packages/core/legoesm/parallel/halo_exchange.py-37-
packages/core/legoesm/parallel/halo_exchange.py:38:Sub-face tiling mode: one sendrecv per edge direction (4 total).
packages/core/legoesm/parallel/halo_exchange.py-39-
packages/core/legoesm/parallel/halo_exchange.py-40-Limitations and Known Bottlenecks
packages/core/legoesm/parallel/halo_exchange.py-41-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
packages/core/legoesm/parallel/halo_exchange.py:42:1. **Blocking sendrecv**: All halo exchanges use blocking ``sendrecv``
packages/core/legoesm/parallel/halo_exchange.py-43-   via ``mpi4jax``.  ``mpi4jax`` does not yet expose ``Isend``/``Irecv``
packages/core/legoesm/parallel/halo_exchange.py-44-   non-blocking primitives, so true computation-communication overlap
packages/core/legoesm/parallel/halo_exchange.py-45-   is not possible at this level.
packages/core/legoesm/parallel/halo_exchange.py-46-
packages/core/legoesm/parallel/halo_exchange.py:47:2. **Face-only mode uses neighbor sendrecv**: When rank count ≤ 6,
packages/core/legoesm/parallel/halo_exchange.py:48:   edges are grouped by neighbor rank and exchanged via one sendrecv
packages/core/legoesm/parallel/halo_exchange.py-49-   per unique neighbor (≤4 messages).  For >6 ranks, the tiled mode
packages/core/legoesm/parallel/halo_exchange.py:50:   uses point-to-point sendrecv with only actual neighbors.
packages/core/legoesm/parallel/halo_exchange.py-51-
packages/core/legoesm/parallel/halo_exchange.py-52-3. **Tiled mode is sequential per edge**: The 4 edge directions are
packages/core/legoesm/parallel/halo_exchange.py:53:   exchanged in sequence (not pipelined).  Each sendrecv blocks until
packages/core/legoesm/parallel/halo_exchange.py-54-   both send and receive complete.
packages/core/legoesm/parallel/halo_exchange.py-55-
packages/core/legoesm/parallel/halo_exchange.py-56-Near-term improvement path:
packages/core/legoesm/parallel/halo_exchange.py-57-  - Batch edge packing into a single contiguous buffer per neighbor
packages/core/legoesm/parallel/halo_exchange.py:58:    rank and do one sendrecv per neighbor (reduces from 4 to ≤4
packages/core/legoesm/parallel/halo_exchange.py-59-    messages per rank, with larger messages for better bandwidth).
packages/core/legoesm/parallel/halo_exchange.py-60-  - When ``mpi4jax`` gains ``Isend``/``Irecv``, convert to non-blocking
packages/core/legoesm/parallel/halo_exchange.py-61-    with ``Waitall`` after all sends/receives are posted.
--
packages/core/legoesm/parallel/halo_exchange.py-145-
packages/core/legoesm/parallel/halo_exchange.py-146-
packages/core/legoesm/parallel/halo_exchange.py-147-# ======================================================================
packages/core/legoesm/parallel/halo_exchange.py:148:# AD-safe sendrecv wrapper (custom_vjp)
packages/core/legoesm/parallel/halo_exchange.py-149-# ======================================================================
packages/core/legoesm/parallel/halo_exchange.py-150-
packages/core/legoesm/parallel/halo_exchange.py:151:def _make_sendrecv_vjp(mpi4jax_mod):
packages/core/legoesm/parallel/halo_exchange.py:152:    """Build an AD-safe sendrecv once mpi4jax is imported.
packages/core/legoesm/parallel/halo_exchange.py-153-
packages/core/legoesm/parallel/halo_exchange.py:154:    mpi4jax.sendrecv has a transpose rule that swaps source/dest, but
packages/core/legoesm/parallel/halo_exchange.py-155-    the XLA lowering raises RuntimeError when ``_must_transpose=True``.
packages/core/legoesm/parallel/halo_exchange.py-156-    This wrapper bypasses that by using ``@jax.custom_vjp``: the forward
packages/core/legoesm/parallel/halo_exchange.py:157:    calls sendrecv normally, and the backward calls sendrecv with
packages/core/legoesm/parallel/halo_exchange.py-158-    swapped endpoints as a fresh forward call (no transpose flag).
packages/core/legoesm/parallel/halo_exchange.py-159-
packages/core/legoesm/parallel/halo_exchange.py-160-    Non-JAX arguments (source, dest, sendtag, recvtag, comm) are
--
packages/core/legoesm/parallel/halo_exchange.py-162-    them.  They are passed through to fwd/bwd as leading static args.
packages/core/legoesm/parallel/halo_exchange.py-163-    """
packages/core/legoesm/parallel/halo_exchange.py-164-
packages/core/legoesm/parallel/halo_exchange.py:165:    def _sendrecv_impl(send_buf, recv_template, source, dest,
packages/core/legoesm/parallel/halo_exchange.py-166-                       sendtag, recvtag, comm):
packages/core/legoesm/parallel/halo_exchange.py-167-        return mpi4jax_array_result(
packages/core/legoesm/parallel/halo_exchange.py:168:            mpi4jax_mod.sendrecv(
packages/core/legoesm/parallel/halo_exchange.py-169-                send_buf, recv_template,
packages/core/legoesm/parallel/halo_exchange.py-170-                source=source, dest=dest,
packages/core/legoesm/parallel/halo_exchange.py-171-                sendtag=sendtag, recvtag=recvtag, comm=comm,
packages/core/legoesm/parallel/halo_exchange.py-172-            )
packages/core/legoesm/parallel/halo_exchange.py-173-        )
packages/core/legoesm/parallel/halo_exchange.py-174-
packages/core/legoesm/parallel/halo_exchange.py:175:    _sendrecv = jax.custom_vjp(
packages/core/legoesm/parallel/halo_exchange.py:176:        _sendrecv_impl, nondiff_argnums=(2, 3, 4, 5, 6),
packages/core/legoesm/parallel/halo_exchange.py-177-    )
packages/core/legoesm/parallel/halo_exchange.py-178-
packages/core/legoesm/parallel/halo_exchange.py-179-    def _fwd(send_buf, recv_template, source, dest, sendtag, recvtag, comm):
packages/core/legoesm/parallel/halo_exchange.py-180-        # fwd has the same signature as the primal function.
packages/core/legoesm/parallel/halo_exchange.py-181-        result = mpi4jax_array_result(
packages/core/legoesm/parallel/halo_exchange.py:182:            mpi4jax_mod.sendrecv(
packages/core/legoesm/parallel/halo_exchange.py-183-                send_buf, recv_template,
packages/core/legoesm/parallel/halo_exchange.py-184-                source=source, dest=dest,
packages/core/legoesm/parallel/halo_exchange.py-185-                sendtag=sendtag, recvtag=recvtag, comm=comm,
--
packages/core/legoesm/parallel/halo_exchange.py-195-        # NOTE: a metadata-only residual (shape, dtype) to avoid retaining the
packages/core/legoesm/parallel/halo_exchange.py-196-        # send activation (codex integration review 2026-06-10, MINOR) was
packages/core/legoesm/parallel/halo_exchange.py-197-        # tried and REVERTED — both numpy-dtype-object and dtype-name-str
packages/core/legoesm/parallel/halo_exchange.py:198:        # forms hit "not a valid JAX type" in the backward zeros/sendrecv on
packages/core/legoesm/parallel/halo_exchange.py-199-        # this jax/mpi4jax stack.  Correctness of this shared AD primitive
packages/core/legoesm/parallel/halo_exchange.py-200-        # outranks the memory micro-opt; revisit with an on-device-verified
packages/core/legoesm/parallel/halo_exchange.py-201-        # ShapeDtypeStruct idiom, not a login-node guess.
--
packages/core/legoesm/parallel/halo_exchange.py-209-        # the template must be SEND-shaped.
packages/core/legoesm/parallel/halo_exchange.py-210-        (send_buf,) = res
packages/core/legoesm/parallel/halo_exchange.py-211-        d_send = mpi4jax_array_result(
packages/core/legoesm/parallel/halo_exchange.py:212:            mpi4jax_mod.sendrecv(
packages/core/legoesm/parallel/halo_exchange.py-213-                g, jnp.zeros_like(send_buf),
packages/core/legoesm/parallel/halo_exchange.py-214-                source=dest, dest=source,
packages/core/legoesm/parallel/halo_exchange.py-215-                sendtag=sendtag, recvtag=recvtag, comm=comm,
--
packages/core/legoesm/parallel/halo_exchange.py-217-        )
packages/core/legoesm/parallel/halo_exchange.py-218-        return d_send, jnp.zeros_like(g)
packages/core/legoesm/parallel/halo_exchange.py-219-
packages/core/legoesm/parallel/halo_exchange.py:220:    _sendrecv.defvjp(_fwd, _bwd)
packages/core/legoesm/parallel/halo_exchange.py:221:    return _sendrecv
packages/core/legoesm/parallel/halo_exchange.py-222-
packages/core/legoesm/parallel/halo_exchange.py-223-
packages/core/legoesm/parallel/halo_exchange.py-224-# Module-level cache: built lazily on first MPI import.
packages/core/legoesm/parallel/halo_exchange.py:225:_sendrecv_vjp_fn = None
packages/core/legoesm/parallel/halo_exchange.py-226-
packages/core/legoesm/parallel/halo_exchange.py-227-
packages/core/legoesm/parallel/halo_exchange.py:228:def get_sendrecv_vjp(mpi4jax_mod):
packages/core/legoesm/parallel/halo_exchange.py:229:    """Return the cached AD-safe sendrecv wrapper.
packages/core/legoesm/parallel/halo_exchange.py-230-
packages/core/legoesm/parallel/halo_exchange.py:231:    This is the single choke point every halo ``sendrecv`` routes through, so
packages/core/legoesm/parallel/halo_exchange.py-232-    it is where the mpi4jax GPU-transport preflight runs — on EVERY call, BEFORE
packages/core/legoesm/parallel/halo_exchange.py-233-    the (memoised) wrapper is returned. Halo entry points reach mpi4jax by a
packages/core/legoesm/parallel/halo_exchange.py-234-    direct ``import mpi4jax`` rather than :func:`require_mpi_stack`, so without
--
packages/core/legoesm/parallel/halo_exchange.py-237-    device-buffer exchange.
packages/core/legoesm/parallel/halo_exchange.py-238-    """
packages/core/legoesm/parallel/halo_exchange.py-239-    check_mpi4jax_transport(mpi4jax_mod)
packages/core/legoesm/parallel/halo_exchange.py:240:    global _sendrecv_vjp_fn
packages/core/legoesm/parallel/halo_exchange.py:241:    if _sendrecv_vjp_fn is None:
packages/core/legoesm/parallel/halo_exchange.py:242:        _sendrecv_vjp_fn = _make_sendrecv_vjp(mpi4jax_mod)
packages/core/legoesm/parallel/halo_exchange.py:243:    return _sendrecv_vjp_fn
packages/core/legoesm/parallel/halo_exchange.py-244-
packages/core/legoesm/parallel/halo_exchange.py-245-
packages/core/legoesm/parallel/halo_exchange.py-246-def _place_strip(padded: jax.Array, face: int, edge: int, strip: jax.Array) -> jax.Array:
--
packages/core/legoesm/parallel/halo_exchange.py-347-    MPI,
packages/core/legoesm/parallel/halo_exchange.py-348-    interp_offsets: jax.Array | None = None,
packages/core/legoesm/parallel/halo_exchange.py-349-) -> jax.Array:
packages/core/legoesm/parallel/halo_exchange.py:350:    """Face-only halo exchange using batched neighbor sendrecv.
packages/core/legoesm/parallel/halo_exchange.py-351-
packages/core/legoesm/parallel/halo_exchange.py:352:    Groups remote edges by neighbor rank and issues one sendrecv per
packages/core/legoesm/parallel/halo_exchange.py-353-    unique neighbor (at most 4 for face-only decomposition), rather
packages/core/legoesm/parallel/halo_exchange.py-354-    than an O(world_size) allgather.
packages/core/legoesm/parallel/halo_exchange.py-355-
--
packages/core/legoesm/parallel/halo_exchange.py-357-
packages/core/legoesm/parallel/halo_exchange.py-358-    * Replicated dynamics — ``data.shape[0] == 6``.  Every rank holds
packages/core/legoesm/parallel/halo_exchange.py-359-      all six faces; only owned faces are written by remote-edge
packages/core/legoesm/parallel/halo_exchange.py:360:      sendrecv (the rest stay as the local copy).  Backward-compatible
packages/core/legoesm/parallel/halo_exchange.py-361-      with the iter 1/2 driver path.
packages/core/legoesm/parallel/halo_exchange.py-362-    * Scattered dynamics — ``data.shape[0] == len(local_face_ids)``.
packages/core/legoesm/parallel/halo_exchange.py-363-      Each rank only owns its faces; ``_build_global_to_local`` maps
--
packages/core/legoesm/parallel/halo_exchange.py-436-            padded = fill_corners_h3(padded)
packages/core/legoesm/parallel/halo_exchange.py-437-        return padded
packages/core/legoesm/parallel/halo_exchange.py-438-
packages/core/legoesm/parallel/halo_exchange.py:439:    # --- Batched neighbor sendrecv for remote edges ---
packages/core/legoesm/parallel/halo_exchange.py-440-    # Group remote edges by neighbor rank so we send one message per
packages/core/legoesm/parallel/halo_exchange.py-441-    # unique neighbor instead of an O(world_size) allgather.
packages/core/legoesm/parallel/halo_exchange.py-442-    from collections import defaultdict
--
packages/core/legoesm/parallel/halo_exchange.py-451-        by_nbr_rank[entry[5]].append(entry)
packages/core/legoesm/parallel/halo_exchange.py-452-
packages/core/legoesm/parallel/halo_exchange.py-453-    # FV3_3D iter-1054: iterate peers in ascending rank order so all
packages/core/legoesm/parallel/halo_exchange.py:454:    # ranks issue sendrecv calls in the same global peer-sequence.
packages/core/legoesm/parallel/halo_exchange.py-455-    # Dict-insertion order produces cyclic-wait deadlock at np=6 face-
packages/core/legoesm/parallel/halo_exchange.py-456-    # only (4 peers per rank).  Same fix as iter-1053 for
packages/core/legoesm/parallel/halo_exchange.py-457-    # synchronize_cgrid_fluxes / _sync_dgrid_boundary.
--
packages/core/legoesm/parallel/halo_exchange.py-480-                    )
packages/core/legoesm/parallel/halo_exchange.py-481-        send_buf = jnp.concatenate(send_parts)
packages/core/legoesm/parallel/halo_exchange.py-482-
packages/core/legoesm/parallel/halo_exchange.py:483:        # Single sendrecv per neighbor.  Tags + neighbor ranks remain
packages/core/legoesm/parallel/halo_exchange.py-484-        # global — they identify the *MPI peer*, not the array slot.
packages/core/legoesm/parallel/halo_exchange.py-485-        send_tag = rank
packages/core/legoesm/parallel/halo_exchange.py-486-        recv_tag = nbr_rank
packages/core/legoesm/parallel/halo_exchange.py:487:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/halo_exchange.py:488:        recv_buf = sendrecv(
packages/core/legoesm/parallel/halo_exchange.py-489-            send_buf, jnp.zeros_like(send_buf),
packages/core/legoesm/parallel/halo_exchange.py-490-            nbr_rank, nbr_rank,
packages/core/legoesm/parallel/halo_exchange.py-491-            send_tag, recv_tag, comm,
--
packages/core/legoesm/parallel/halo_exchange.py-551-    mpi4jax,
packages/core/legoesm/parallel/halo_exchange.py-552-    MPI,
packages/core/legoesm/parallel/halo_exchange.py-553-) -> jax.Array:
packages/core/legoesm/parallel/halo_exchange.py:554:    """Sub-face tiling halo exchange using batched point-to-point sendrecv.
packages/core/legoesm/parallel/halo_exchange.py-555-
packages/core/legoesm/parallel/halo_exchange.py-556-    Edges destined for the same neighbor rank are packed into a single
packages/core/legoesm/parallel/halo_exchange.py-557-    contiguous send buffer, reducing the number of MPI messages.  A tile
packages/core/legoesm/parallel/halo_exchange.py:558:    with 4 unique neighbor ranks still does 4 sendrecv calls, but when
packages/core/legoesm/parallel/halo_exchange.py-559-    2 edges share a neighbor (corner tiles), this batching halves the
packages/core/legoesm/parallel/halo_exchange.py-560-    message count for those edges.
packages/core/legoesm/parallel/halo_exchange.py-561-
--
packages/core/legoesm/parallel/halo_exchange.py-633-                    )
packages/core/legoesm/parallel/halo_exchange.py-634-        send_buf = jnp.concatenate(send_parts)
packages/core/legoesm/parallel/halo_exchange.py-635-
packages/core/legoesm/parallel/halo_exchange.py:636:        # Single sendrecv for all edges to this neighbor.
packages/core/legoesm/parallel/halo_exchange.py-637-        send_tag = rank
packages/core/legoesm/parallel/halo_exchange.py-638-        recv_tag = nbr_rank
packages/core/legoesm/parallel/halo_exchange.py:639:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/halo_exchange.py:640:        recv_buf = sendrecv(
packages/core/legoesm/parallel/halo_exchange.py-641-            send_buf, jnp.zeros_like(send_buf),
packages/core/legoesm/parallel/halo_exchange.py-642-            nbr_rank, nbr_rank,
packages/core/legoesm/parallel/halo_exchange.py-643-            send_tag, recv_tag, comm,
--
packages/core/legoesm/parallel/halo_exchange.py-924-    MPI,
packages/core/legoesm/parallel/halo_exchange.py-925-    interp_offsets: jax.Array | None = None,
packages/core/legoesm/parallel/halo_exchange.py-926-) -> jax.Array:
packages/core/legoesm/parallel/halo_exchange.py:927:    """Face-only 4D halo exchange: one sendrecv per neighbor for all levels."""
packages/core/legoesm/parallel/halo_exchange.py-928-    # Single Pad HLO op replaces alloc-zeros + scatter (4D face-only
packages/core/legoesm/parallel/halo_exchange.py-929-    # path).
packages/core/legoesm/parallel/halo_exchange.py-930-    padded = jnp.pad(
--
packages/core/legoesm/parallel/halo_exchange.py-998-            padded = fill_corners_h3(padded)
packages/core/legoesm/parallel/halo_exchange.py-999-        return padded
packages/core/legoesm/parallel/halo_exchange.py-1000-
packages/core/legoesm/parallel/halo_exchange.py:1001:    # Batched neighbor sendrecv — one message per neighbor for all levels
packages/core/legoesm/parallel/halo_exchange.py-1002-    from collections import defaultdict
packages/core/legoesm/parallel/halo_exchange.py-1003-    n = data.shape[1]
packages/core/legoesm/parallel/halo_exchange.py-1004-    nlev = data.shape[3]
--
packages/core/legoesm/parallel/halo_exchange.py-1036-
packages/core/legoesm/parallel/halo_exchange.py-1037-        send_tag = rank
packages/core/legoesm/parallel/halo_exchange.py-1038-        recv_tag = nbr_rank
packages/core/legoesm/parallel/halo_exchange.py:1039:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/halo_exchange.py:1040:        recv_buf = sendrecv(
packages/core/legoesm/parallel/halo_exchange.py-1041-            send_buf, jnp.zeros_like(send_buf),
packages/core/legoesm/parallel/halo_exchange.py-1042-            nbr_rank, nbr_rank,
packages/core/legoesm/parallel/halo_exchange.py-1043-            send_tag, recv_tag, comm,
--
packages/core/legoesm/parallel/halo_exchange.py-1101-    mpi4jax,
packages/core/legoesm/parallel/halo_exchange.py-1102-    MPI,
packages/core/legoesm/parallel/halo_exchange.py-1103-) -> jax.Array:
packages/core/legoesm/parallel/halo_exchange.py:1104:    """Sub-face tiling 4D halo exchange: one sendrecv per neighbor for all levels.
packages/core/legoesm/parallel/halo_exchange.py-1105-
packages/core/legoesm/parallel/halo_exchange.py-1106-    Iter 30: tiled mode owns one tile of one face per rank; ``face`` is
packages/core/legoesm/parallel/halo_exchange.py-1107-    the global id (used for connectivity lookups) but data/padded are
--
packages/core/legoesm/parallel/halo_exchange.py-1181-
packages/core/legoesm/parallel/halo_exchange.py-1182-        send_tag = rank
packages/core/legoesm/parallel/halo_exchange.py-1183-        recv_tag = nbr_rank
packages/core/legoesm/parallel/halo_exchange.py:1184:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/halo_exchange.py:1185:        recv_buf = sendrecv(
packages/core/legoesm/parallel/halo_exchange.py-1186-            send_buf, jnp.zeros_like(send_buf),
packages/core/legoesm/parallel/halo_exchange.py-1187-            nbr_rank, nbr_rank,
packages/core/legoesm/parallel/halo_exchange.py-1188-            send_tag, recv_tag, comm,
--
tests/distributed/test_voronoi_batched_halo.py-3-No MPI launcher required: pure schedule math, pack/unpack round-trips
tests/distributed/test_voronoi_batched_halo.py-4-through a simulated exchange, message-count accounting, the np=1
tests/distributed/test_voronoi_batched_halo.py-5-identity degeneracy, and an AST guard that the Voronoi halo module never
tests/distributed/test_voronoi_batched_halo.py:6:calls raw ``mpi4jax.sendrecv`` (every message must route through the
tests/distributed/test_voronoi_batched_halo.py:7:AD-safe ``get_sendrecv_vjp`` wrapper).
tests/distributed/test_voronoi_batched_halo.py-8-
tests/distributed/test_voronoi_batched_halo.py-9-Companions:
tests/distributed/test_voronoi_batched_halo.py-10-
--
tests/distributed/test_voronoi_batched_halo.py-176-        GLOBAL ids of A's send rows equal the global ids of B's recv rows
tests/distributed/test_voronoi_batched_halo.py-177-        elementwise (both sides sort by global index).  This is the
tests/distributed/test_voronoi_batched_halo.py-178-        uniform-collective-schedule requirement: matching blocking
tests/distributed/test_voronoi_batched_halo.py:179:        sendrecv posts with matching sizes on both ends.
tests/distributed/test_voronoi_batched_halo.py-180-        """
tests/distributed/test_voronoi_batched_halo.py-181-        parts, scheds = _build_all(mesh, n_ranks)
tests/distributed/test_voronoi_batched_halo.py-182-        for a, (pa, sa) in enumerate(zip(parts, scheds)):
--
tests/distributed/test_voronoi_batched_halo.py-438-
tests/distributed/test_voronoi_batched_halo.py-439-
tests/distributed/test_voronoi_batched_halo.py-440-# ========================================================================
tests/distributed/test_voronoi_batched_halo.py:441:# AD routing: no raw mpi4jax.sendrecv left in the Voronoi halo module
tests/distributed/test_voronoi_batched_halo.py-442-# ========================================================================
tests/distributed/test_voronoi_batched_halo.py-443-
tests/distributed/test_voronoi_batched_halo.py:444:def _raw_sendrecv_call_lines(source: str) -> list[int]:
tests/distributed/test_voronoi_batched_halo.py:445:    """Line numbers of attribute-style ``*.sendrecv(...)`` calls.
tests/distributed/test_voronoi_batched_halo.py-446-
tests/distributed/test_voronoi_batched_halo.py-447-    The AD-safe path calls the wrapper as a bare name
tests/distributed/test_voronoi_batched_halo.py:448:    (``sendrecv = get_sendrecv_vjp(mpi4jax); sendrecv(...)``), so any
tests/distributed/test_voronoi_batched_halo.py:449:    attribute call ``<module>.sendrecv(...)`` is a raw, forward-only
tests/distributed/test_voronoi_batched_halo.py-450-    mpi4jax primitive — a regression of the custom-VJP routing.
tests/distributed/test_voronoi_batched_halo.py-451-    """
tests/distributed/test_voronoi_batched_halo.py-452-    tree = ast.parse(source)
--
tests/distributed/test_voronoi_batched_halo.py-455-        for node in ast.walk(tree)
tests/distributed/test_voronoi_batched_halo.py-456-        if isinstance(node, ast.Call)
tests/distributed/test_voronoi_batched_halo.py-457-        and isinstance(node.func, ast.Attribute)
tests/distributed/test_voronoi_batched_halo.py:458:        and node.func.attr == "sendrecv"
tests/distributed/test_voronoi_batched_halo.py-459-    ]
tests/distributed/test_voronoi_batched_halo.py-460-
tests/distributed/test_voronoi_batched_halo.py-461-
tests/distributed/test_voronoi_batched_halo.py-462-class TestADRoutingStatic:
tests/distributed/test_voronoi_batched_halo.py:463:    def test_no_raw_mpi4jax_sendrecv_in_voronoi_halo(self):
tests/distributed/test_voronoi_batched_halo.py-464-        source = Path(hev.__file__).read_text()
tests/distributed/test_voronoi_batched_halo.py:465:        hits = _raw_sendrecv_call_lines(source)
tests/distributed/test_voronoi_batched_halo.py-466-        assert hits == [], (
tests/distributed/test_voronoi_batched_halo.py:467:            f"raw *.sendrecv call(s) at line(s) {hits} of "
tests/distributed/test_voronoi_batched_halo.py:468:            f"{hev.__file__} — route through get_sendrecv_vjp instead"
tests/distributed/test_voronoi_batched_halo.py-469-        )
tests/distributed/test_voronoi_batched_halo.py-470-        # And the AD-safe wrapper is actually what the module uses.
tests/distributed/test_voronoi_batched_halo.py:471:        assert "get_sendrecv_vjp" in source
tests/distributed/test_voronoi_batched_halo.py-472-
tests/distributed/test_voronoi_batched_halo.py:473:    def test_raw_sendrecv_detector_catches_violation(self):
tests/distributed/test_voronoi_batched_halo.py-474-        """Tripwire self-test: the AST guard is not vacuous."""
tests/distributed/test_voronoi_batched_halo.py-475-        bad = (
tests/distributed/test_voronoi_batched_halo.py-476-            "import mpi4jax\n"
tests/distributed/test_voronoi_batched_halo.py-477-            "def f(a, b, c):\n"
tests/distributed/test_voronoi_batched_halo.py:478:            "    return mpi4jax.sendrecv(a, b, source=c, dest=c)\n"
tests/distributed/test_voronoi_batched_halo.py-479-        )
tests/distributed/test_voronoi_batched_halo.py:480:        assert _raw_sendrecv_call_lines(bad) == [3]
tests/distributed/test_voronoi_batched_halo.py-481-        good = (
tests/distributed/test_voronoi_batched_halo.py:482:            "from legoesm.parallel.halo_exchange import get_sendrecv_vjp\n"
tests/distributed/test_voronoi_batched_halo.py-483-            "def f(mpi4jax, a, b, c):\n"
tests/distributed/test_voronoi_batched_halo.py:484:            "    sendrecv = get_sendrecv_vjp(mpi4jax)\n"
tests/distributed/test_voronoi_batched_halo.py:485:            "    return sendrecv(a, b, c, c, 0, 0, None)\n"
tests/distributed/test_voronoi_batched_halo.py-486-        )
tests/distributed/test_voronoi_batched_halo.py:487:        assert _raw_sendrecv_call_lines(good) == []
tests/distributed/test_voronoi_batched_halo.py-488-
tests/distributed/test_voronoi_batched_halo.py-489-
tests/distributed/test_voronoi_batched_halo.py-490-class TestRankIndependentTags:
--
tests/distributed/test_voronoi_batched_halo.py-492-
tests/distributed/test_voronoi_batched_halo.py-493-    The old per-entity scheme encoded ``rank * 1000 + nbr`` and hard-raised at
tests/distributed/test_voronoi_batched_halo.py-494-    >= 1000 ranks (and the batched path's ``rank * n_ranks`` + 3e6 base pushed
tests/distributed/test_voronoi_batched_halo.py:495:    tags past MPI_TAG_UB).  mpi4jax ``sendrecv`` already matches on (source,
tests/distributed/test_voronoi_batched_halo.py-496-    dest), so the rank pair is redundant in the tag; dropping it removes the
tests/distributed/test_voronoi_batched_halo.py-497-    ceiling.  These tests lock that the scheme can never reintroduce a
tests/distributed/test_voronoi_batched_halo.py-498-    rank-dependent tag.
--
tests/distributed/test_latlon_2d_mpi_step.py-124-    dt = 100.0
tests/distributed/test_latlon_2d_mpi_step.py-125-
tests/distributed/test_latlon_2d_mpi_step.py-126-    # Both legs are 2-D MPI (no serial leg), every rank in lockstep — the
tests/distributed/test_latlon_2d_mpi_step.py:127:    # mass allreduce + halo sendrecvs are SPMD-symmetric, no deadlock.
tests/distributed/test_latlon_2d_mpi_step.py-128-    mb22, ma22, g22 = _run_layout(gstate, grid, sigma, config, 2, 2, dt)
tests/distributed/test_latlon_2d_mpi_step.py-129-    mb14, ma14, g14 = _run_layout(gstate, grid, sigma, config, 1, 4, dt)
tests/distributed/test_latlon_2d_mpi_step.py-130-
--
packages/core/legoesm/parallel/plane_mpi.py-13-
packages/core/legoesm/parallel/plane_mpi.py-14-AD safety
packages/core/legoesm/parallel/plane_mpi.py-15----------
packages/core/legoesm/parallel/plane_mpi.py:16:Every ``sendrecv`` call goes through ``get_sendrecv_vjp`` from
packages/core/legoesm/parallel/plane_mpi.py-17-:mod:`legoesm.parallel.halo_exchange` (existing ``@jax.custom_vjp``
packages/core/legoesm/parallel/plane_mpi.py-18-wrapper that swaps source / dest in the backward pass per
packages/core/legoesm/parallel/plane_mpi.py-19-``CLAUDE.md`` "MPI AD compat" rule). ``allreduce(SUM)`` reductions
--
packages/core/legoesm/parallel/plane_mpi.py-30-
packages/core/legoesm/parallel/plane_mpi.py-31-Multi-rank coverage
packages/core/legoesm/parallel/plane_mpi.py-32--------------------
packages/core/legoesm/parallel/plane_mpi.py:33:``layout.n_ranks > 1`` runs a two-stage sendrecv: N/S along the
packages/core/legoesm/parallel/plane_mpi.py-34-``y`` axis first, then E/W of the *already-NS-padded* array so the
packages/core/legoesm/parallel/plane_mpi.py-35-four corner halos arrive via two-axis composition (mirrors the
packages/core/legoesm/parallel/plane_mpi.py:36:cubed-sphere ``_pad_halo_mpi`` pattern). All sendrecv calls flow
packages/core/legoesm/parallel/plane_mpi.py:37:through ``get_sendrecv_vjp`` so :func:`jax.grad` works through
packages/core/legoesm/parallel/plane_mpi.py-38-the exchange. Validated by ``tests/distributed/test_plane_pencil_mpi.py``
packages/core/legoesm/parallel/plane_mpi.py-39-under the OpenMPI ``mpi-distributed.yml`` CI job (np = 2, 4).
packages/core/legoesm/parallel/plane_mpi.py-40-
--
packages/core/legoesm/parallel/plane_mpi.py-233-
packages/core/legoesm/parallel/plane_mpi.py-234-    Output shape ``(ny_local + 2h, nx_local + 2h, ...)``. The four
packages/core/legoesm/parallel/plane_mpi.py-235-    interior edges are filled from the matching neighbour rank via
packages/core/legoesm/parallel/plane_mpi.py:236:    ``_sendrecv_vjp`` (AD-safe); the four corners come from
packages/core/legoesm/parallel/plane_mpi.py-237-    composing the two-axis wrap. Single-rank ``n_ranks == 1`` path
packages/core/legoesm/parallel/plane_mpi.py-238-    uses ``jnp.pad(mode='wrap')`` and skips MPI entirely.
packages/core/legoesm/parallel/plane_mpi.py-239-
--
packages/core/legoesm/parallel/plane_mpi.py-272-    # Multi-rank path. Each axis (y, x) handled INDEPENDENTLY:
packages/core/legoesm/parallel/plane_mpi.py-273-    #   * n_ranks_axis == 1 → local periodic wrap on that axis (no
packages/core/legoesm/parallel/plane_mpi.py-274-    #     MPI). Codex review 2026-05-24: previously the multi-rank
packages/core/legoesm/parallel/plane_mpi.py:275:    #     path always issued sendrecv even when the periodic
packages/core/legoesm/parallel/plane_mpi.py-276-    #     neighbour was the current rank. That triggered two
packages/core/legoesm/parallel/plane_mpi.py:277:    #     self-sendrecv on the same tag base with identical
packages/core/legoesm/parallel/plane_mpi.py-278-    #     `(source, dest)` for both directions — MPI matched the
packages/core/legoesm/parallel/plane_mpi.py-279-    #     messages arbitrarily and filled the east halo with the
packages/core/legoesm/parallel/plane_mpi.py-280-    #     east boundary (instead of the west) or vice versa.
packages/core/legoesm/parallel/plane_mpi.py:281:    #   * n_ranks_axis >  1 → mpi4jax sendrecv with the periodic
packages/core/legoesm/parallel/plane_mpi.py-282-    #     neighbour rank.
packages/core/legoesm/parallel/plane_mpi.py-283-    # NS stage always runs first; the EW stage operates on the
packages/core/legoesm/parallel/plane_mpi.py-284-    # NS-padded array so the four corners arrive via two-axis
--
packages/core/legoesm/parallel/plane_mpi.py-295-                "mpi4py (install with `pip install -e \".[mpi]\"` and "
packages/core/legoesm/parallel/plane_mpi.py-296-                "have OpenMPI available)."
packages/core/legoesm/parallel/plane_mpi.py-297-            ) from exc
packages/core/legoesm/parallel/plane_mpi.py:298:        from legoesm.parallel.halo_exchange import get_sendrecv_vjp
packages/core/legoesm/parallel/plane_mpi.py-299-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/plane_mpi.py:300:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/plane_mpi.py-301-    else:
packages/core/legoesm/parallel/plane_mpi.py:302:        sendrecv = None  # never used; single-rank shortcut above
packages/core/legoesm/parallel/plane_mpi.py-303-
packages/core/legoesm/parallel/plane_mpi.py-304-    trailing = field_yxz.shape[2:]
packages/core/legoesm/parallel/plane_mpi.py-305-    # iter-57 Codex LOW#2 doc: MPI tag namespace. N/S exchanges use
--
packages/core/legoesm/parallel/plane_mpi.py-327-        recv_template = jnp.zeros_like(send_to_north)
packages/core/legoesm/parallel/plane_mpi.py-328-        # Directional ring-shift exchange — correct AND deadlock-free even
packages/core/legoesm/parallel/plane_mpi.py-329-        # when an axis has exactly 2 ranks (north_rank == south_rank).  The
packages/core/legoesm/parallel/plane_mpi.py:330:        # old ``send-to-X & recv-from-X`` pattern paired both sendrecvs to the
packages/core/legoesm/parallel/plane_mpi.py-331-        # same neighbour under a SHARED tag, so at n_ranks==2 MPI matched the
packages/core/legoesm/parallel/plane_mpi.py-332-        # WRONG message and delivered the neighbour's opposite edge (silent
packages/core/legoesm/parallel/plane_mpi.py-333-        # halo corruption — du_dt off by ~0.5 vs single-process).
--
packages/core/legoesm/parallel/plane_mpi.py-335-        # (its north edge).  Fill NORTH halo: send my SOUTH edge -> south_rank,
packages/core/legoesm/parallel/plane_mpi.py-336-        # recv <- north_rank.  One tag per shift direction (a consistent ring
packages/core/legoesm/parallel/plane_mpi.py-337-        # shift, so send/recv pair within the SAME call across the ring).
packages/core/legoesm/parallel/plane_mpi.py:338:        from_south = sendrecv(
packages/core/legoesm/parallel/plane_mpi.py-339-            send_to_north, recv_template,
packages/core/legoesm/parallel/plane_mpi.py-340-            layout.south_rank, layout.north_rank,
packages/core/legoesm/parallel/plane_mpi.py-341-            _TAG_NS, _TAG_NS, comm,
packages/core/legoesm/parallel/plane_mpi.py-342-        )
packages/core/legoesm/parallel/plane_mpi.py:343:        from_north = sendrecv(
packages/core/legoesm/parallel/plane_mpi.py-344-            send_to_south, recv_template,
packages/core/legoesm/parallel/plane_mpi.py-345-            layout.north_rank, layout.south_rank,
packages/core/legoesm/parallel/plane_mpi.py-346-            _TAG_NS + 1, _TAG_NS + 1, comm,
--
packages/core/legoesm/parallel/plane_mpi.py-371-        # at n_ranks_x == 2 where west_rank == east_rank).  Fill WEST halo:
packages/core/legoesm/parallel/plane_mpi.py-372-        # send my EAST edge -> east_rank, recv <- west_rank (its east edge).
packages/core/legoesm/parallel/plane_mpi.py-373-        # Fill EAST halo: send my WEST edge -> west_rank, recv <- east_rank.
packages/core/legoesm/parallel/plane_mpi.py:374:        from_west = sendrecv(
packages/core/legoesm/parallel/plane_mpi.py-375-            send_to_east, recv_template_ew,
packages/core/legoesm/parallel/plane_mpi.py-376-            layout.west_rank, layout.east_rank,
packages/core/legoesm/parallel/plane_mpi.py-377-            _TAG_EW, _TAG_EW, comm,
packages/core/legoesm/parallel/plane_mpi.py-378-        )
packages/core/legoesm/parallel/plane_mpi.py:379:        from_east = sendrecv(
packages/core/legoesm/parallel/plane_mpi.py-380-            send_to_west, recv_template_ew,
packages/core/legoesm/parallel/plane_mpi.py-381-            layout.east_rank, layout.west_rank,
packages/core/legoesm/parallel/plane_mpi.py-382-            _TAG_EW + 1, _TAG_EW + 1, comm,
--
packages/core/legoesm/parallel/plane_mpi.py-407-    + reshapes. Reduces MPI message count from ``len(fields)``
packages/core/legoesm/parallel/plane_mpi.py-408-    exchanges to 1 — critical on macOS shared-memory MPI where the
packages/core/legoesm/parallel/plane_mpi.py-409-    per-call mpi4jax dispatch overhead dominates many-small-message
packages/core/legoesm/parallel/plane_mpi.py:410:    workloads (~2-5 ms per sendrecv vs ~50 μs on cluster IB).
packages/core/legoesm/parallel/plane_mpi.py-411-
packages/core/legoesm/parallel/plane_mpi.py-412-    All fields must share the same ``(ny_local, nx_local)`` prefix +
packages/core/legoesm/parallel/plane_mpi.py-413-    same trailing-axis dimensions (typically same ``nlev``). If
--
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-54-
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-55-def test_grad_flows_through_mpi_dgrid_halo():
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-56-    """FV3_3D iter-1090: jax.grad must flow through the MPI dgrid
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py:57:    halo.  Raw mpi4jax.sendrecv chokes on the symbolic Zero
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py:58:    cotangent; the iter-1090 fix routes through ``_sendrecv_vjp``
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-59-    (the same custom_vjp wrapper the iter-1040+ MPI halo uses)."""
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-60-    from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d_replicated_mpi
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-61-    from legoesm.parallel.comm import build_comm_topology
--
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-90-    is the canonical training pipeline pattern.  iter-1090 verified
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-91-    plain jax.grad works; this test verifies the composition with JIT
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-92-    survives — both the forward primitive and the custom_vjp's bwd
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py:93:    sendrecv must JIT-trace cleanly.
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-94-    """
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-95-    from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d_replicated_mpi
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-96-    from legoesm.parallel.comm import build_comm_topology
--
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-180-        err_msg=(
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-181-            f"rank {rank} allreduced MPI g_u does not match single-"
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-182-            f"device reference.  Likely cause: miswired custom_vjp "
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py:183:            f"in _sendrecv_vjp through DGRID halo."
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-184-        ),
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-185-    )
tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py-186-    np.testing.assert_allclose(
--
tests/distributed/test_distributed_multigrid_banded.py-8-The banded V-cycle (``_make_multigrid_preconditioner_banded``) coarsens each
tests/distributed/test_distributed_multigrid_banded.py-9-rank's LATITUDE BAND with band-local 2x2 transfers and a comm-free zonal-line
tests/distributed/test_distributed_multigrid_banded.py-10-smoother; the ONLY communication is the neighbour halo inside each Helmholtz
tests/distributed/test_distributed_multigrid_banded.py:11:``A_op`` (``gradient_y_cgrid`` -> MPI sendrecv).  There is NO global allreduce
tests/distributed/test_distributed_multigrid_banded.py-12-in the V-cycle — that is the multinode reduction-latency win over the flat
tests/distributed/test_distributed_multigrid_banded.py-13-fixed-M Jacobi-PCG (2M allreduces/step).
tests/distributed/test_distributed_multigrid_banded.py-14-
--
tests/distributed/test_latlon_mpi_halo.py-240-
tests/distributed/test_latlon_mpi_halo.py-241-
tests/distributed/test_latlon_mpi_halo.py-242-# ---------------------------------------------------------------------------
tests/distributed/test_latlon_mpi_halo.py:243:# AD safety: jax.grad through MPI sendrecv returns a finite gradient
tests/distributed/test_latlon_mpi_halo.py-244-# ---------------------------------------------------------------------------
tests/distributed/test_latlon_mpi_halo.py-245-
tests/distributed/test_latlon_mpi_halo.py-246-
tests/distributed/test_latlon_mpi_halo.py-247-class TestADUnderMPI:
tests/distributed/test_latlon_mpi_halo.py-248-
tests/distributed/test_latlon_mpi_halo.py-249-    def test_grad_through_mpi_halo_is_finite(self):
tests/distributed/test_latlon_mpi_halo.py:250:        """Reverse-mode AD through the AD-safe ``_sendrecv_vjp`` halo
tests/distributed/test_latlon_mpi_halo.py-251-        must produce finite gradients on every rank.  We just check
tests/distributed/test_latlon_mpi_halo.py-252-        finiteness + shape — exact value vs. serial reference is hard
tests/distributed/test_latlon_mpi_halo.py-253-        to compute under MPI without re-implementing the backward in
--
packages/core/legoesm/parallel/runtime.py-216-    """Halo exchange backend identifiers."""
packages/core/legoesm/parallel/runtime.py-217-    LOCAL = "local"       # pad_halo from grids.halo (single process)
packages/core/legoesm/parallel/runtime.py-218-    JAX_SPMD = "jax"      # implicit via XLA when data is sharded
packages/core/legoesm/parallel/runtime.py:219:    MPI = "mpi"           # mpi4jax sendrecv / allgather
packages/core/legoesm/parallel/runtime.py-220-    HYBRID = "hybrid"     # MPI between ranks + JAX within rank
packages/core/legoesm/parallel/runtime.py-221-
packages/core/legoesm/parallel/runtime.py-222-
--
packages/core/legoesm/parallel/runtime.py-442-            # the MPI halo backend via the standalone helper.  This
packages/core/legoesm/parallel/runtime.py-443-            # is what makes ``pad_halo_latlon*`` / ``pad_ns_zero`` /
packages/core/legoesm/parallel/runtime.py-444-            # ``pad_with_pole_bc_lat`` inside the dycore dispatch
packages/core/legoesm/parallel/runtime.py:445:            # through MPI sendrecv at partition cuts and pole-fold /
packages/core/legoesm/parallel/runtime.py-446-            # wall-BC constants at boundary ranks — see
packages/core/legoesm/parallel/runtime.py-447-            # legoesm.parallel.distributed.initialize_distributed_latlon
packages/core/legoesm/parallel/runtime.py-448-            # for the full activation contract.
--
packages/core/legoesm/parallel/halo_exchange_voronoi.py-4-halo (ghost) entities from their owning ranks.  Also provides a
packages/core/legoesm/parallel/halo_exchange_voronoi.py-5-simulated local exchange for testing without MPI.
packages/core/legoesm/parallel/halo_exchange_voronoi.py-6-
packages/core/legoesm/parallel/halo_exchange_voronoi.py:7:All MPI messages route through the AD-safe ``@jax.custom_vjp`` sendrecv
packages/core/legoesm/parallel/halo_exchange_voronoi.py:8:wrapper (:func:`legoesm.parallel.halo_exchange.get_sendrecv_vjp`) so the
packages/core/legoesm/parallel/halo_exchange_voronoi.py-9-Voronoi halo exchange is reverse-mode differentiable, like the lat-lon
packages/core/legoesm/parallel/halo_exchange_voronoi.py-10-and cubed-sphere halos.
packages/core/legoesm/parallel/halo_exchange_voronoi.py-11-
--
packages/core/legoesm/parallel/halo_exchange_voronoi.py-57-# Message-count instrumentation
packages/core/legoesm/parallel/halo_exchange_voronoi.py-58-# ============================================================================
packages/core/legoesm/parallel/halo_exchange_voronoi.py-59-
packages/core/legoesm/parallel/halo_exchange_voronoi.py:60:# Incremented once per MPI sendrecv POST.  The increment happens when the
packages/core/legoesm/parallel/halo_exchange_voronoi.py-61-# exchange function body runs — i.e. at TRACE time under ``jax.jit`` (once
packages/core/legoesm/parallel/halo_exchange_voronoi.py-62-# per compilation, not per execution) and per call when running eagerly.
packages/core/legoesm/parallel/halo_exchange_voronoi.py-63-# That is exactly the "messages per exchange" schedule quantity the
--
packages/core/legoesm/parallel/halo_exchange_voronoi.py-67-
packages/core/legoesm/parallel/halo_exchange_voronoi.py-68-
packages/core/legoesm/parallel/halo_exchange_voronoi.py-69-def reset_halo_message_count() -> None:
packages/core/legoesm/parallel/halo_exchange_voronoi.py:70:    """Reset the traced-sendrecv message counter to zero."""
packages/core/legoesm/parallel/halo_exchange_voronoi.py-71-    global _halo_message_count
packages/core/legoesm/parallel/halo_exchange_voronoi.py-72-    _halo_message_count = 0
packages/core/legoesm/parallel/halo_exchange_voronoi.py-73-
packages/core/legoesm/parallel/halo_exchange_voronoi.py-74-
packages/core/legoesm/parallel/halo_exchange_voronoi.py-75-def get_halo_message_count() -> int:
packages/core/legoesm/parallel/halo_exchange_voronoi.py:76:    """Number of MPI sendrecv posts traced since the last reset."""
packages/core/legoesm/parallel/halo_exchange_voronoi.py-77-    return _halo_message_count
packages/core/legoesm/parallel/halo_exchange_voronoi.py-78-
packages/core/legoesm/parallel/halo_exchange_voronoi.py-79-
--
packages/core/legoesm/parallel/halo_exchange_voronoi.py-145-    rank: int,
packages/core/legoesm/parallel/halo_exchange_voronoi.py-146-    entity_type: int,
packages/core/legoesm/parallel/halo_exchange_voronoi.py-147-) -> jnp.ndarray:
packages/core/legoesm/parallel/halo_exchange_voronoi.py:148:    """Perform MPI halo exchange using mpi4jax sendrecv.
packages/core/legoesm/parallel/halo_exchange_voronoi.py-149-
packages/core/legoesm/parallel/halo_exchange_voronoi.py-150-    Gathers every neighbor's send buffer from the ORIGINAL field, issues
packages/core/legoesm/parallel/halo_exchange_voronoi.py:151:    one blocking sendrecv per neighbor, then scatters all received data
packages/core/legoesm/parallel/halo_exchange_voronoi.py-152-    back in a SINGLE functional update.
packages/core/legoesm/parallel/halo_exchange_voronoi.py-153-
packages/core/legoesm/parallel/halo_exchange_voronoi.py-154-    Works for fields of any shape ``(n_local, ...)`` — multi-level
--
packages/core/legoesm/parallel/halo_exchange_voronoi.py-161-    field copied 6× per exchange) and (b) created a false
packages/core/legoesm/parallel/halo_exchange_voronoi.py-162-    read-after-write dependency: ``send_buf = field[send_idx]`` for the
packages/core/legoesm/parallel/halo_exchange_voronoi.py-163-    next neighbor textually read the just-mutated ``field``, forcing XLA
packages/core/legoesm/parallel/halo_exchange_voronoi.py:164:    to serialize the blocking sendrecvs even though ``send_idx`` are
packages/core/legoesm/parallel/halo_exchange_voronoi.py-165-    OWNED entities that never overlap the halo ``recv_idx`` written by any
packages/core/legoesm/parallel/halo_exchange_voronoi.py-166-    neighbor.  Collecting all sends from the original field and scattering
packages/core/legoesm/parallel/halo_exchange_voronoi.py-167-    once removes both costs.  Measured comm overhead was 22 ms/step at
packages/core/legoesm/parallel/halo_exchange_voronoi.py:168:    np=4 (18 serialized sendrecv) and 30 ms at np=8 (36) — see
packages/core/legoesm/parallel/halo_exchange_voronoi.py-169-    docs/performance/scaling/amip_mpi_scaling.md.
packages/core/legoesm/parallel/halo_exchange_voronoi.py-170-
packages/core/legoesm/parallel/halo_exchange_voronoi.py-171-    Correctness: ``send_idx`` ⊂ owned, ``recv_idx`` ⊂ halo (disjoint), and
--
packages/core/legoesm/parallel/halo_exchange_voronoi.py-175-    the previous result exactly, order-independent.
packages/core/legoesm/parallel/halo_exchange_voronoi.py-176-
packages/core/legoesm/parallel/halo_exchange_voronoi.py-177-    Differentiability: every message goes through the ``@jax.custom_vjp``
packages/core/legoesm/parallel/halo_exchange_voronoi.py:178:    sendrecv wrapper (:func:`legoesm.parallel.halo_exchange.get_sendrecv_vjp`)
packages/core/legoesm/parallel/halo_exchange_voronoi.py-179-    — the same AD-safe path the lat-lon / cubed-sphere halos use — so
packages/core/legoesm/parallel/halo_exchange_voronoi.py-180-    ``jax.grad`` routes halo cotangents back to the owning rank instead of
packages/core/legoesm/parallel/halo_exchange_voronoi.py-181-    hitting mpi4jax's broken transpose rule.  Never call raw
packages/core/legoesm/parallel/halo_exchange_voronoi.py:182:    ``mpi4jax.sendrecv`` here (enforced by the AST guard in
packages/core/legoesm/parallel/halo_exchange_voronoi.py-183-    ``tests/distributed/test_voronoi_batched_halo.py``).
packages/core/legoesm/parallel/halo_exchange_voronoi.py-184-    """
packages/core/legoesm/parallel/halo_exchange_voronoi.py:185:    from legoesm.parallel.halo_exchange import get_sendrecv_vjp
packages/core/legoesm/parallel/halo_exchange_voronoi.py-186-    from legoesm.parallel.reductions import require_mpi_stack
packages/core/legoesm/parallel/halo_exchange_voronoi.py-187-
packages/core/legoesm/parallel/halo_exchange_voronoi.py-188-    mpi4jax, MPI = require_mpi_stack()
packages/core/legoesm/parallel/halo_exchange_voronoi.py:189:    sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/halo_exchange_voronoi.py-190-
packages/core/legoesm/parallel/halo_exchange_voronoi.py-191-    # Tag = the entity type ALONE (cell=0 / edge=1 / vertex=2).  mpi4jax
packages/core/legoesm/parallel/halo_exchange_voronoi.py:192:    # ``sendrecv`` already matches on (source, dest), so the rank PAIR is never
packages/core/legoesm/parallel/halo_exchange_voronoi.py-193-    # encoded in the tag -- the only thing source/dest does NOT separate is two
packages/core/legoesm/parallel/halo_exchange_voronoi.py-194-    # messages of DIFFERENT entity type between the same pair (a cell vs an edge
packages/core/legoesm/parallel/halo_exchange_voronoi.py-195-    # buffer, different sizes).  Same-entity multi-field exchanges between a pair
--
packages/core/legoesm/parallel/halo_exchange_voronoi.py-223-            )
packages/core/legoesm/parallel/halo_exchange_voronoi.py-224-
packages/core/legoesm/parallel/halo_exchange_voronoi.py-225-        _count_message()
packages/core/legoesm/parallel/halo_exchange_voronoi.py:226:        recv_data = sendrecv(
packages/core/legoesm/parallel/halo_exchange_voronoi.py-227-            send_buf, recv_buf,
packages/core/legoesm/parallel/halo_exchange_voronoi.py-228-            nbr_rank, nbr_rank,
packages/core/legoesm/parallel/halo_exchange_voronoi.py-229-            tag,
--
packages/core/legoesm/parallel/halo_exchange_voronoi.py-257-# path, and the VJP (split/scatter-add duals of the pack, gather duals of
packages/core/legoesm/parallel/halo_exchange_voronoi.py-258-# the unpack) is exact.
packages/core/legoesm/parallel/halo_exchange_voronoi.py-259-
packages/core/legoesm/parallel/halo_exchange_voronoi.py:260:# MPI tags here are RANK-INDEPENDENT: mpi4jax ``sendrecv`` matches on
packages/core/legoesm/parallel/halo_exchange_voronoi.py-261-# (source, dest), so the rank pair is never encoded in the tag and the scheme
packages/core/legoesm/parallel/halo_exchange_voronoi.py-262-# scales to ANY rank count (no < 1000-rank ceiling).  A tag only separates
packages/core/legoesm/parallel/halo_exchange_voronoi.py-263-# concurrent messages between the SAME (source, dest) pair: per-entity
--
packages/core/legoesm/parallel/halo_exchange_voronoi.py-273-# messages that share (comm, source, dest, tag) -- e.g. the production MPAS
packages/core/legoesm/parallel/halo_exchange_voronoi.py-274-# step's two cell exchanges (T+p_s, then tracers) between one pair -- are
packages/core/legoesm/parallel/halo_exchange_voronoi.py-275-# disambiguated NOT by the tag but by ORDER.  Every rank runs the same SPMD
packages/core/legoesm/parallel/halo_exchange_voronoi.py:276:# program order; mpi4jax's ordered effect keeps the sendrecvs from being
packages/core/legoesm/parallel/halo_exchange_voronoi.py-277-# reordered/parallelised; MPI's non-overtaking guarantee then pairs them 1:1.
packages/core/legoesm/parallel/halo_exchange_voronoi.py-278-# A NEW halo path that issues same-(source,dest,tag) messages MUST preserve that
packages/core/legoesm/parallel/halo_exchange_voronoi.py-279-# program-order property (regression: test_voronoi_mpi
--
packages/core/legoesm/parallel/halo_exchange_voronoi.py-284-def _entity_tag(entity_type: int) -> int:
packages/core/legoesm/parallel/halo_exchange_voronoi.py-285-    """MPI tag for a per-entity halo message: the entity type ALONE.
packages/core/legoesm/parallel/halo_exchange_voronoi.py-286-
packages/core/legoesm/parallel/halo_exchange_voronoi.py:287:    Rank-independent: mpi4jax ``sendrecv`` matches on (source, dest), so the
packages/core/legoesm/parallel/halo_exchange_voronoi.py-288-    pair is never encoded in the tag.  The entity type is the only thing
packages/core/legoesm/parallel/halo_exchange_voronoi.py-289-    source/dest does not separate (cell vs edge vs vertex between one pair).
packages/core/legoesm/parallel/halo_exchange_voronoi.py-290-    """
--
packages/core/legoesm/parallel/halo_exchange_voronoi.py-512-                          sched: BatchedHaloSchedule, rank: int):
packages/core/legoesm/parallel/halo_exchange_voronoi.py-513-    """One MPI halo exchange for edge AND cell fields together.
packages/core/legoesm/parallel/halo_exchange_voronoi.py-514-
packages/core/legoesm/parallel/halo_exchange_voronoi.py:515:    Issues ONE blocking sendrecv per union neighbor per dtype group
packages/core/legoesm/parallel/halo_exchange_voronoi.py-516-    (vs one per neighbor per entity exchange), via the AD-safe
packages/core/legoesm/parallel/halo_exchange_voronoi.py:517:    ``custom_vjp`` sendrecv wrapper — fully reverse-mode differentiable.
packages/core/legoesm/parallel/halo_exchange_voronoi.py-518-    Pack/unpack are linear gather/scatter, so results are bit-identical
packages/core/legoesm/parallel/halo_exchange_voronoi.py-519-    to the per-entity exchanges for the production same-compute-dtype
packages/core/legoesm/parallel/halo_exchange_voronoi.py-520-    state (T, p_s, and tracers share the compute dtype; the legacy path's
--
packages/core/legoesm/parallel/halo_exchange_voronoi.py-550-        # np=1 / no-neighbor degeneracy: identity, no MPI dependency.
packages/core/legoesm/parallel/halo_exchange_voronoi.py-551-        return edge_fields, cell_fields
packages/core/legoesm/parallel/halo_exchange_voronoi.py-552-
packages/core/legoesm/parallel/halo_exchange_voronoi.py:553:    from legoesm.parallel.halo_exchange import get_sendrecv_vjp
packages/core/legoesm/parallel/halo_exchange_voronoi.py-554-    from legoesm.parallel.reductions import require_mpi_stack
packages/core/legoesm/parallel/halo_exchange_voronoi.py-555-
packages/core/legoesm/parallel/halo_exchange_voronoi.py-556-    mpi4jax, MPI = require_mpi_stack()
packages/core/legoesm/parallel/halo_exchange_voronoi.py:557:    sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/halo_exchange_voronoi.py-558-
packages/core/legoesm/parallel/halo_exchange_voronoi.py-559-    groups, plan = _message_schedule(edge_fields, cell_fields, sched)
packages/core/legoesm/parallel/halo_exchange_voronoi.py-560-    send_buffers = pack_batched_sends(edge_fields, cell_fields, sched)
--
packages/core/legoesm/parallel/halo_exchange_voronoi.py-578-                g_recv.append(jnp.zeros(0, dtype=dtype))
packages/core/legoesm/parallel/halo_exchange_voronoi.py-579-                continue
packages/core/legoesm/parallel/halo_exchange_voronoi.py-580-            _count_message()
packages/core/legoesm/parallel/halo_exchange_voronoi.py:581:            g_recv.append(sendrecv(
packages/core/legoesm/parallel/halo_exchange_voronoi.py-582-                send_buffers[g][i],
packages/core/legoesm/parallel/halo_exchange_voronoi.py-583-                jnp.zeros(r_sz, dtype=dtype),
packages/core/legoesm/parallel/halo_exchange_voronoi.py-584-                nbr_rank, nbr_rank,
--
packages/core/legoesm/parallel/reductions.py-240-    """``True`` if the installed jax/mpi4jax fall outside legoESM's tested MPI
packages/core/legoesm/parallel/reductions.py-241-    range, so distributed *numerics* may be unreliable.
packages/core/legoesm/parallel/reductions.py-242-
packages/core/legoesm/parallel/reductions.py:243:    Halo exchange (point-to-point ``sendrecv``) stays bit-correct across the
packages/core/legoesm/parallel/reductions.py-244-    range, but the global-allreduce path (mass fixer, ``global_sum_mpi``) can
packages/core/legoesm/parallel/reductions.py-245-    drift ~1e-9 vs serial under an incompatible custom-call ABI — enough to
packages/core/legoesm/parallel/reductions.py-246-    break the tight serial-vs-MPI equivalence pins.  Tests that assert that
--
packages/core/legoesm/parallel/reductions.py-272-
packages/core/legoesm/parallel/reductions.py-273-
packages/core/legoesm/parallel/reductions.py-274-# --- mpi4jax GPU transport (device-direct vs host-staged) -------------------
packages/core/legoesm/parallel/reductions.py:275:# mpi4jax decides, once per process, whether a halo ``sendrecv`` hands the
packages/core/legoesm/parallel/reductions.py-276-# on-device buffer straight to (GPU-aware) MPI or first copies
packages/core/legoesm/parallel/reductions.py-277-# device->host->device.  The switch is the env var ``MPI4JAX_USE_CUDA_MPI``
packages/core/legoesm/parallel/reductions.py-278-# (read in mpi4jax's decorators): unset/falsy -> HOST-STAGED (always safe, but
packages/core/legoesm/parallel/reductions.py-279-# the per-exchange host round-trip caps multi-GPU scaling); truthy ->
packages/core/legoesm/parallel/reductions.py-280-# GPU-DIRECT (fast, but segfaults if mpi4jax has no CUDA extension or the MPI is
packages/core/legoesm/parallel/reductions.py:281:# not GPU-aware).  legoESM hands device arrays to ``sendrecv`` unconditionally,
packages/core/legoesm/parallel/reductions.py-282-# so on a GPU backend this choice is otherwise INVISIBLE -- a silent host-stage
packages/core/legoesm/parallel/reductions.py-283-# looks like "the halo works but doesn't scale".  ``mpi4jax.has_cuda_support()``
packages/core/legoesm/parallel/reductions.py-284-# reports whether the CUDA extension was built in; a missing or raising
--
packages/core/legoesm/parallel/reductions.py-290-    "MPI4JAX_USE_CUDA_MPI is set (GPU-direct halo exchange requested) on a GPU "
packages/core/legoesm/parallel/reductions.py-291-    "backend, but this mpi4jax does not report usable CUDA support "
packages/core/legoesm/parallel/reductions.py-292-    "(mpi4jax.has_cuda_support() returned False or could not be called): handing "
packages/core/legoesm/parallel/reductions.py:293:    "an on-device sendrecv buffer to MPI would error or segfault. Rebuild "
packages/core/legoesm/parallel/reductions.py-294-    "mpi4jax with CUDA against the GPU-aware Cray MPICH "
packages/core/legoesm/parallel/reductions.py-295-    "(scripts/cluster/scaling_derecho/README.md), or unset MPI4JAX_USE_CUDA_MPI "
packages/core/legoesm/parallel/reductions.py-296-    "to use the (slower) host-staged path."
packages/core/legoesm/parallel/reductions.py-297-)
packages/core/legoesm/parallel/reductions.py-298-_MPI4JAX_HOST_STAGED_MSG = (
packages/core/legoesm/parallel/reductions.py-299-    "GPU backend with a CUDA-capable mpi4jax, but MPI4JAX_USE_CUDA_MPI is unset: "
packages/core/legoesm/parallel/reductions.py:300:    "every halo sendrecv copies device->host->device, which caps multi-GPU "
packages/core/legoesm/parallel/reductions.py-301-    "scaling. Export MPI4JAX_USE_CUDA_MPI=1 for GPU-direct halo exchange (also "
packages/core/legoesm/parallel/reductions.py-302-    "needs MPICH_GPU_SUPPORT_ENABLED=1 and the craype-accel-nvidia80 GTL on "
packages/core/legoesm/parallel/reductions.py-303-    "Cray/Slingshot)."
--
packages/core/legoesm/parallel/reductions.py-313-
packages/core/legoesm/parallel/reductions.py-314-    * ``"raise"`` -- GPU backend with GPU-direct REQUESTED (``use_cuda_mpi``) but
packages/core/legoesm/parallel/reductions.py-315-      CUDA support not POSITIVELY proven (``cuda_built`` is ``False`` *or*
packages/core/legoesm/parallel/reductions.py:316:      ``None``/unintrospectable): handing an on-device ``sendrecv`` buffer to MPI
packages/core/legoesm/parallel/reductions.py-317-      would error or segfault, so fail CLOSED before the first halo exchange.
packages/core/legoesm/parallel/reductions.py-318-    * ``"warn"``  -- GPU backend, the toggle UNSET, and a CUDA-capable mpi4jax:
packages/core/legoesm/parallel/reductions.py-319-      the halo silently host-stages (device<->host copy), capping GPU scaling.
--
packages/core/legoesm/parallel/reductions.py-336-    """Surface the mpi4jax GPU halo transport mode.
packages/core/legoesm/parallel/reductions.py-337-
packages/core/legoesm/parallel/reductions.py-338-    Called from every checked entry point that can issue a device-array MPI op:
packages/core/legoesm/parallel/reductions.py:339:    :func:`require_mpi_stack` (collectives) and the halo sendrecv choke point
packages/core/legoesm/parallel/reductions.py:340:    :func:`legoesm.parallel.halo_exchange.get_sendrecv_vjp`.  Trace-safe: pure
packages/core/legoesm/parallel/reductions.py-341-    Python, no JAX ops.
packages/core/legoesm/parallel/reductions.py-342-    The hard-error condition is RECOMPUTED on every call (never latched) so a
packages/core/legoesm/parallel/reductions.py-343-    later ``MPI4JAX_USE_CUDA_MPI`` / backend / build change cannot slip a
--
packages/core/legoesm/parallel/voronoi_mpi.py-13-(job 8488057) and I6 f64 np8/np16 -4.6%/-3.5% (job 8488023); the historical
packages/core/legoesm/parallel/voronoi_mpi.py-14-18x regression (435.9 ms I5/f32, 2026-06-10) was fixed by the one-scatter
packages/core/legoesm/parallel/voronoi_mpi.py-15-pack/unpack rework.  On both paths every message routes through the AD-safe
packages/core/legoesm/parallel/voronoi_mpi.py:16:``@jax.custom_vjp`` sendrecv wrapper (reverse-mode differentiable, never raw
packages/core/legoesm/parallel/voronoi_mpi.py:17:``mpi4jax.sendrecv``), and pack/unpack is pure gather/scatter so the two
packages/core/legoesm/parallel/voronoi_mpi.py-18-paths are bit-identical.
packages/core/legoesm/parallel/voronoi_mpi.py-19-
packages/core/legoesm/parallel/voronoi_mpi.py-20-Design
--
packages/core/legoesm/parallel/voronoi_mpi.py-96-#   * ``"0"`` -> legacy per-entity exchange (opt-OUT): per neighbor, one u edge
packages/core/legoesm/parallel/voronoi_mpi.py-97-#     message + ONE packed T/p_s cell message + ONE packed tracer cell message,
packages/core/legoesm/parallel/voronoi_mpi.py-98-#     via ``VoronoiHaloExchange`` -> ``_exchange_mpi`` -> the AD-safe
packages/core/legoesm/parallel/voronoi_mpi.py:99:#     ``get_sendrecv_vjp`` wrapper.  Pack/unpack is pure gather/scatter, so the
packages/core/legoesm/parallel/voronoi_mpi.py-100-#     two paths are bit-identical (kept as a fallback + parity reference).
packages/core/legoesm/parallel/voronoi_mpi.py-101-#
packages/core/legoesm/parallel/voronoi_mpi.py-102-# Trace-time semantics: the flag is consulted as a static Python bool while
--
packages/core/legoesm/parallel/voronoi_mpi.py-366-    message per neighbor per dtype group for the whole prognostic state
packages/core/legoesm/parallel/voronoi_mpi.py-367-    — ``u`` (edge) plus ``T``, ``S``, ``eta``, ``w`` (cell) — via
packages/core/legoesm/parallel/voronoi_mpi.py-368-    :func:`legoesm.parallel.halo_exchange_voronoi.batched_halo_exchange`
packages/core/legoesm/parallel/voronoi_mpi.py:369:    (AD-safe ``custom_vjp`` sendrecv; identity when the schedule has no
packages/core/legoesm/parallel/voronoi_mpi.py-370-    neighbors, i.e. np=1).  ``H_bathy``, ``land_mask`` and ``rho_ref_z``
packages/core/legoesm/parallel/voronoi_mpi.py-371-    are static after :func:`scatter_state_mpas_ocean` (halo filled at
packages/core/legoesm/parallel/voronoi_mpi.py-372-    scatter) and pass through unexchanged.
--
packages/core/legoesm/parallel/voronoi_mpi.py-437-
packages/core/legoesm/parallel/voronoi_mpi.py-438-    Each callable issues ONE batched union-neighbor message per neighbor
packages/core/legoesm/parallel/voronoi_mpi.py-439-    per dtype group (:func:`batched_halo_exchange` — AD-safe ``custom_vjp``
packages/core/legoesm/parallel/voronoi_mpi.py:440:    sendrecv, safe inside ``lax.scan``; identity when the schedule has no
packages/core/legoesm/parallel/voronoi_mpi.py-441-    neighbors).  ``None`` (the serial default everywhere) keeps every
packages/core/legoesm/parallel/voronoi_mpi.py-442-    consumer byte-identical — the refresh sites are static Python
packages/core/legoesm/parallel/voronoi_mpi.py-443-    ``if halo_refresh is not None`` branches.
--
packages/core/legoesm/parallel/voronoi_mpi.py-458-        cell/edge message (the batched schedule carries no vertex lane;
packages/core/legoesm/parallel/voronoi_mpi.py-459-        the ONLY vertex consumer is the K_zeta_bih vorticity-biharmonic
packages/core/legoesm/parallel/voronoi_mpi.py-460-        intermediate, one field per tendency call, so a per-field
packages/core/legoesm/parallel/voronoi_mpi.py:461:        exchange is proportionate).  Same AD-safe custom_vjp sendrecv.
packages/core/legoesm/parallel/voronoi_mpi.py-462-    """
packages/core/legoesm/parallel/voronoi_mpi.py-463-
packages/core/legoesm/parallel/voronoi_mpi.py-464-    edges: Callable
--
packages/core/legoesm/parallel/voronoi_mpi.py-733-    .. note::
packages/core/legoesm/parallel/voronoi_mpi.py-734-       **Differentiable halo path.**  The Voronoi halo exchange
packages/core/legoesm/parallel/voronoi_mpi.py-735-       (:mod:`legoesm.parallel.halo_exchange_voronoi`) routes every
packages/core/legoesm/parallel/voronoi_mpi.py:736:       message through the AD-safe ``@jax.custom_vjp`` sendrecv wrapper
packages/core/legoesm/parallel/voronoi_mpi.py:737:       (:func:`legoesm.parallel.halo_exchange.get_sendrecv_vjp`) — the
packages/core/legoesm/parallel/voronoi_mpi.py-738-       same path the lat-lon / cubed-sphere halos use — so ``jax.grad``
packages/core/legoesm/parallel/voronoi_mpi.py-739-       through this step propagates halo cotangents back to the owning
packages/core/legoesm/parallel/voronoi_mpi.py-740-       rank (see ``tests/distributed/test_mpi_differentiability.py::
--
packages/core/legoesm/parallel/voronoi_mpi.py-866-    # Static Python switch (module-level ``_USE_BATCHED_HALO``, env
packages/core/legoesm/parallel/voronoi_mpi.py-867-    # ``LEGOESM_VORONOI_BATCHED_HALO`` read at import): both branches bind
packages/core/legoesm/parallel/voronoi_mpi.py-868-    # the same ``_exchange_mpas_state`` name, and BOTH route every message
packages/core/legoesm/parallel/voronoi_mpi.py:869:    # through the AD-safe ``get_sendrecv_vjp`` sendrecv wrapper.  The choice
packages/core/legoesm/parallel/voronoi_mpi.py-870-    # is baked into the traced/jitted ``_step`` — the dynamics RK stages AND
packages/core/legoesm/parallel/voronoi_mpi.py-871-    # the post-physics exchange below use this same closure.
packages/core/legoesm/parallel/voronoi_mpi.py-872-    if _USE_BATCHED_HALO:
--
packages/core/legoesm/parallel/voronoi_mpi.py-939-    else:
packages/core/legoesm/parallel/voronoi_mpi.py-940-        # opt-OUT (``LEGOESM_VORONOI_BATCHED_HALO=0``): legacy per-entity
packages/core/legoesm/parallel/voronoi_mpi.py-941-        # exchange.  ``halo_ex`` dispatches through ``_exchange_mpi`` — every
packages/core/legoesm/parallel/voronoi_mpi.py:942:        # message via the AD-safe ``get_sendrecv_vjp`` custom-VJP wrapper, same
packages/core/legoesm/parallel/voronoi_mpi.py-943-        # as the (now default) batched path; kept as the parity reference.
packages/core/legoesm/parallel/voronoi_mpi.py-944-        halo_ex = layout.halo_exchange
packages/core/legoesm/parallel/voronoi_mpi.py-945-
--
packages/core/legoesm/parallel/sharded_dynamics.py-2285-            local_state = MPASHydrostaticState(
packages/core/legoesm/parallel/sharded_dynamics.py-2286-                u=Field(data=u_local, name="u",
packages/core/legoesm/parallel/sharded_dynamics.py-2287-                        dims=("nEdges", "nlev"), units="m/s",
packages/core/legoesm/parallel/sharded_dynamics.py:2288:                        long_name="normal velocity", staggering="edge"),
packages/core/legoesm/parallel/sharded_dynamics.py-2289-                T=Field(data=T_local, name="T",
packages/core/legoesm/parallel/sharded_dynamics.py-2290-                        dims=("nCells", "nlev"), units="K",
packages/core/legoesm/parallel/sharded_dynamics.py:2291:                        long_name="temperature", staggering="cell"),
packages/core/legoesm/parallel/sharded_dynamics.py-2292-                p_s=Field(data=ps_local, name="p_s",
packages/core/legoesm/parallel/sharded_dynamics.py-2293-                          dims=("nCells",), units="Pa",
packages/core/legoesm/parallel/sharded_dynamics.py:2294:                          long_name="surface pressure", staggering="cell"),
packages/core/legoesm/parallel/sharded_dynamics.py-2295-                phis=Field(data=phis_local, name="phis",
packages/core/legoesm/parallel/sharded_dynamics.py-2296-                           dims=("nCells",), units="m^2/s^2",
packages/core/legoesm/parallel/sharded_dynamics.py-2297-                           long_name="surface geopotential",
--
tests/distributed/test_latlon_2d_fused_pad.py-1-"""2-rank MPI equivalence for the fused 2-D pencil wall pad.
tests/distributed/test_latlon_2d_fused_pad.py-2-
tests/distributed/test_latlon_2d_fused_pad.py:3:``pad_with_pole_bc_lat_multi_2d`` (codex consult #3) must be
tests/distributed/test_latlon_2d_fused_pad.py:4:value-identical to N single-field ``pad_with_pole_bc_lat_2d`` calls
tests/distributed/test_latlon_2d_fused_pad.py:5:while issuing ONE sendrecv pair per cut per dtype group.
tests/distributed/test_latlon_2d_fused_pad.py-6-
tests/distributed/test_latlon_2d_fused_pad.py-7-Run: mpirun -np 2 python -m pytest tests/distributed/test_latlon_2d_fused_pad.py
tests/distributed/test_latlon_2d_fused_pad.py-8-"""
--
tests/distributed/test_latlon_2d_fused_pad.py-16-
tests/distributed/test_latlon_2d_fused_pad.py-17-from legoesm.parallel.latlon_mpi import (  # noqa: E402
tests/distributed/test_latlon_2d_fused_pad.py-18-    make_latlon_2d_layout,
tests/distributed/test_latlon_2d_fused_pad.py:19:    pad_with_pole_bc_lat_2d,
tests/distributed/test_latlon_2d_fused_pad.py:20:    pad_with_pole_bc_lat_multi_2d,
tests/distributed/test_latlon_2d_fused_pad.py-21-)
tests/distributed/test_latlon_2d_fused_pad.py-22-
tests/distributed/test_latlon_2d_fused_pad.py-23-
--
tests/distributed/test_latlon_2d_fused_pad.py-42-    fields = (f32, f32b, f64)
tests/distributed/test_latlon_2d_fused_pad.py-43-    sv, nv = (0.0, 1.5, -2.0), (0.5, 0.0, 3.0)
tests/distributed/test_latlon_2d_fused_pad.py-44-
tests/distributed/test_latlon_2d_fused_pad.py:45:    fused = pad_with_pole_bc_lat_multi_2d(
tests/distributed/test_latlon_2d_fused_pad.py-46-        fields, lay, halo=1, south_values=sv, north_values=nv)
tests/distributed/test_latlon_2d_fused_pad.py-47-    ref = tuple(
tests/distributed/test_latlon_2d_fused_pad.py:48:        pad_with_pole_bc_lat_2d(f, lay, halo=1,
tests/distributed/test_latlon_2d_fused_pad.py-49-                                south_value=sv[i], north_value=nv[i])
tests/distributed/test_latlon_2d_fused_pad.py-50-        for i, f in enumerate(fields))
tests/distributed/test_latlon_2d_fused_pad.py-51-    for i, (a, b) in enumerate(zip(fused, ref)):
--
tests/distributed/test_latlon_2d_fused_pad.py-69-                    dtype=jnp.float64)
tests/distributed/test_latlon_2d_fused_pad.py-70-
tests/distributed/test_latlon_2d_fused_pad.py-71-    def loss(a):
tests/distributed/test_latlon_2d_fused_pad.py:72:        out = pad_with_pole_bc_lat_multi_2d((a, a * 2.0), lay, halo=1)
tests/distributed/test_latlon_2d_fused_pad.py-73-        return sum(jnp.sum(o ** 2) for o in out)
tests/distributed/test_latlon_2d_fused_pad.py-74-
tests/distributed/test_latlon_2d_fused_pad.py-75-    g = jax.grad(loss)(x)
--
tests/distributed/test_latlon_mpi_step.py-85-    itself (deterministically identical — same global state, no comm)
tests/distributed/test_latlon_mpi_step.py-86-    *before* ``make_latlon_mpi_step`` arms the MPI halo backend.  A
tests/distributed/test_latlon_mpi_step.py-87-    rank-0-only serial reference computed *after* arming deadlocks:
tests/distributed/test_latlon_mpi_step.py:88:    the serial ``_step_cgrid`` traces MPI sendrecv/allreduce ops
tests/distributed/test_latlon_mpi_step.py-89-    (``fix_mass`` → ``batch_global_area_sums``; operator pads → band
tests/distributed/test_latlon_mpi_step.py:90:    sendrecv) that no other rank matches.  That deadlock is exactly
tests/distributed/test_latlon_mpi_step.py-91-    how every historical np>1 run of this file timed out while rank 1
tests/distributed/test_latlon_mpi_step.py-92-    printed a trivial early-return "passed"."""
tests/distributed/test_latlon_mpi_step.py-93-    return CGridLatLonPrimitiveEquationModel(
--
tests/distributed/test_latlon_mpi_step.py-231-                    reason=(
tests/distributed/test_latlon_mpi_step.py-232-                        "Multi-rank PPM tracer transport under MPI: "
tests/distributed/test_latlon_mpi_step.py-233-                        "the JIT-compile graph fuses PPM mass-flux "
tests/distributed/test_latlon_mpi_step.py:234:                        "reconstruction × per-direction sendrecv × "
tests/distributed/test_latlon_mpi_step.py-235-                        "per-tracer reduction into a single XLA module "
tests/distributed/test_latlon_mpi_step.py-236-                        "that exceeds 90 min of wallclock to compile "
tests/distributed/test_latlon_mpi_step.py-237-                        "on Ginsburg compute nodes.  The dry case "
--
tests/distributed/test_latlon_mpi_step.py-274-        # --- Serial reference FIRST, on EVERY rank, LOCAL backend ---
tests/distributed/test_latlon_mpi_step.py-275-        # Must precede ``make_latlon_mpi_step`` (which arms the band-
tests/distributed/test_latlon_mpi_step.py-276-        # layout MPI halo backend as a global side effect): a serial
tests/distributed/test_latlon_mpi_step.py:277:        # ``_step_cgrid`` traced after arming embeds band sendrecvs +
tests/distributed/test_latlon_mpi_step.py-278-        # the ``fix_mass`` allreduce, which deadlocks when only rank 0
tests/distributed/test_latlon_mpi_step.py-279-        # runs it.  Under the local backend the graph is collective-
tests/distributed/test_latlon_mpi_step.py-280-        # free (``is_distributed()`` is False at trace time), so every
--
tests/distributed/test_latlon_mpi_step.py-386-        # materialized under the local halo backend BEFORE
tests/distributed/test_latlon_mpi_step.py-387-        # ``make_latlon_mpi_step`` arms the band-layout MPI backend —
tests/distributed/test_latlon_mpi_step.py-388-        # otherwise the rank-0-only serial graph embeds band
tests/distributed/test_latlon_mpi_step.py:389:        # sendrecvs + the ``fix_mass`` allreduce and deadlocks at
tests/distributed/test_latlon_mpi_step.py-390-        # np>1.  Every rank computes it (deterministic, comm-free).
tests/distributed/test_latlon_mpi_step.py-391-        set_halo_backend("local")
tests/distributed/test_latlon_mpi_step.py-392-        serial_out = global_state
--
packages/core/legoesm/timestepping/leapfrog_ab2.py-51-Integration into ``LatLonCGridOceanModel`` and the ``timestepping/
packages/core/legoesm/timestepping/leapfrog_ab2.py-52-split_explicit.py`` ``outer_integrator`` dispatch is tracked under
packages/core/legoesm/timestepping/leapfrog_ab2.py-53-Phase G.2 in ``docs/ocean/fidelity/phase_g_veros_recipe_audit.md`` —
packages/core/legoesm/timestepping/leapfrog_ab2.py:54:this module provides the standalone closure ready for that wiring.
packages/core/legoesm/timestepping/leapfrog_ab2.py-55-
packages/core/legoesm/timestepping/leapfrog_ab2.py-56-References
packages/core/legoesm/timestepping/leapfrog_ab2.py-57-----------
--
packages/core/legoesm/parallel/latlon_spmd.py-60-    return perm_north, perm_south
packages/core/legoesm/parallel/latlon_spmd.py-61-
packages/core/legoesm/parallel/latlon_spmd.py-62-
packages/core/legoesm/parallel/latlon_spmd.py:63:def latlon_lon_ring_perms(p_lon: int):
packages/core/legoesm/parallel/latlon_spmd.py:64:    """Static (src, dst) permutation pairs over the periodic ``lon`` ring axis.
packages/core/legoesm/parallel/latlon_spmd.py-65-
packages/core/legoesm/parallel/latlon_spmd.py-66-    Longitude is a periodic RING (unlike the pole-terminated lat LINE), so
packages/core/legoesm/parallel/latlon_spmd.py-67-    every tile is both a source and a destination — the perms are full cyclic
--
packages/core/legoesm/parallel/latlon_spmd.py-77-    return perm_to_west, perm_to_east
packages/core/legoesm/parallel/latlon_spmd.py-78-
packages/core/legoesm/parallel/latlon_spmd.py-79-
packages/core/legoesm/parallel/latlon_spmd.py:80:def lon_ring_ghosts_spmd(f, mesh, halo: int = 1):
packages/core/legoesm/parallel/latlon_spmd.py-81-    """Periodic LONGITUDE ghosts (axis 1) under the 2-D lat-lon SPMD mesh.
packages/core/legoesm/parallel/latlon_spmd.py-82-
packages/core/legoesm/parallel/latlon_spmd.py-83-    The SPMD twin of the local ``jnp.pad(mode="wrap")`` lon wrap (band path)
--
packages/core/legoesm/parallel/latlon_spmd.py-85-    SECTOR, so its east/west ghost columns are the neighbouring tiles' edge
packages/core/legoesm/parallel/latlon_spmd.py-86-    columns, moved by ``jax.lax.ppermute`` over the ``"lon"`` mesh axis (the
packages/core/legoesm/parallel/latlon_spmd.py-87-    periodic wrap IS the cyclic ring permutation —
packages/core/legoesm/parallel/latlon_spmd.py:88:    :func:`latlon_lon_ring_perms`).  ``p_lon == 1`` (a degenerate lon axis /
packages/core/legoesm/parallel/latlon_spmd.py-89-    the 1-D-band-equivalent (N, 1) mesh) is the LOCAL wrap, chosen by a
packages/core/legoesm/parallel/latlon_spmd.py-90-    STATIC Python branch — bit-identical to the band path's ``jnp.pad`` (no
packages/core/legoesm/parallel/latlon_spmd.py-91-    collective is emitted at all).
--
packages/core/legoesm/parallel/latlon_spmd.py-104-    if p_lon == 1:
packages/core/legoesm/parallel/latlon_spmd.py-105-        pad = ((0, 0), (halo, halo)) + ((0, 0),) * (f.ndim - 2)
packages/core/legoesm/parallel/latlon_spmd.py-106-        return jnp.pad(f, pad, mode="wrap")
packages/core/legoesm/parallel/latlon_spmd.py:107:    perm_to_west, perm_to_east = latlon_lon_ring_perms(p_lon)
packages/core/legoesm/parallel/latlon_spmd.py-108-    # My EAST ghost = east neighbour's west edge (sources send WEST edges to
packages/core/legoesm/parallel/latlon_spmd.py-109-    # their west neighbour); my WEST ghost = west neighbour's east edge.
packages/core/legoesm/parallel/latlon_spmd.py-110-    east_ghost = jax.lax.ppermute(f[:, :halo], "lon", perm_to_west)
--
packages/core/legoesm/parallel/latlon_spmd.py-216-def reconstruct_uface_left(u_left, axis: str, p_lon: int):
packages/core/legoesm/parallel/latlon_spmd.py-217-    """Rebuild the ``n_lon_local+1`` staggered u-faces from the
packages/core/legoesm/parallel/latlon_spmd.py-218-    ``n_lon_local``-column ``u_left`` representation, INSIDE a ``shard_map``
packages/core/legoesm/parallel/latlon_spmd.py:219:    over the periodic ``"lon"`` ring axis — the u-stagger twin of
packages/core/legoesm/parallel/latlon_spmd.py-220-    :func:`reconstruct_vface_lower`.
packages/core/legoesm/parallel/latlon_spmd.py-221-
packages/core/legoesm/parallel/latlon_spmd.py-222-    The zonal velocity ``u`` carries ``n_lon+1`` lon-interface columns whose
--
packages/core/legoesm/parallel/latlon_spmd.py-248-    if p_lon == 1:
packages/core/legoesm/parallel/latlon_spmd.py-249-        boundary = u_left[:, 0:1]
packages/core/legoesm/parallel/latlon_spmd.py-250-    else:
packages/core/legoesm/parallel/latlon_spmd.py:251:        perm_to_west, _ = latlon_lon_ring_perms(p_lon)
packages/core/legoesm/parallel/latlon_spmd.py-252-        boundary = jax.lax.ppermute(u_left[:, 0:1], axis, perm_to_west)
packages/core/legoesm/parallel/latlon_spmd.py-253-    return jnp.concatenate([u_left, boundary], axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-254-
--
packages/core/legoesm/parallel/latlon_spmd.py-384-            f"(the E/W extension strips would exceed the neighbour tile); "
packages/core/legoesm/parallel/latlon_spmd.py-385-            f"use the all_gather fold for this degenerate tiling")
packages/core/legoesm/parallel/latlon_spmd.py-386-    W = w * p_lon
packages/core/legoesm/parallel/latlon_spmd.py:387:    # 1. E/W-extend the edge rows by 2h (one lon ring ppermute pair).
packages/core/legoesm/parallel/latlon_spmd.py:388:    ext = lon_ring_ghosts_spmd(edge, mesh, halo=2 * h)  # (halo, w+4h[, lev])
packages/core/legoesm/parallel/latlon_spmd.py:389:    # 2. ONE antipodal ppermute over the lon ring (shift by p_lon/2 is a
packages/core/legoesm/parallel/latlon_spmd.py-390-    # bijection, and c' != c for every even p_lon >= 2).
packages/core/legoesm/parallel/latlon_spmd.py-391-    perm_anti = [(s, (s + p_lon // 2) % p_lon) for s in range(p_lon)]
packages/core/legoesm/parallel/latlon_spmd.py-392-    recv = jax.lax.ppermute(ext, "lon", perm_anti)
--
packages/core/legoesm/parallel/latlon_spmd.py-490-
packages/core/legoesm/parallel/latlon_spmd.py-491-    * latitude interior cuts: ``ppermute`` of the ``halo`` edge rows over
packages/core/legoesm/parallel/latlon_spmd.py-492-      ``"lat"`` (identical to the band body);
packages/core/legoesm/parallel/latlon_spmd.py:493:    * longitude: :func:`lon_ring_ghosts_spmd` — the periodic wrap as a cyclic
packages/core/legoesm/parallel/latlon_spmd.py-494-      ring ``ppermute`` over ``"lon"`` (``p_lon == 1``: the LOCAL wrap, a
packages/core/legoesm/parallel/latlon_spmd.py-495-      static branch — no collective, bit-identical to the band body);
packages/core/legoesm/parallel/latlon_spmd.py-496-    * corners: filled by SEQUENCING lat-then-lon — the E/W neighbour's edge
--
packages/core/legoesm/parallel/latlon_spmd.py-563-            south_recv = jax.lax.ppermute(tile[-halo:], "lat", perm_south)
packages/core/legoesm/parallel/latlon_spmd.py-564-        ext = jnp.concatenate([south_recv, tile, north_recv], axis=0)
packages/core/legoesm/parallel/latlon_spmd.py-565-
packages/core/legoesm/parallel/latlon_spmd.py:566:        # 2. longitude ring ghosts on the lat-EXTENDED block (fills corners
packages/core/legoesm/parallel/latlon_spmd.py-567-        # from the E/W neighbour's lat-ghost rows == the diagonal tile).
packages/core/legoesm/parallel/latlon_spmd.py:568:        ext = lon_ring_ghosts_spmd(ext, mesh, halo=halo)
packages/core/legoesm/parallel/latlon_spmd.py-569-
packages/core/legoesm/parallel/latlon_spmd.py-570-        # 3. pole fold at the physical pole tiles (ppermute non-targets
packages/core/legoesm/parallel/latlon_spmd.py-571-        # received zeros in step 1; overwritten here on the pole rows).
--
packages/core/legoesm/parallel/latlon_spmd.py-650-    and reshaped back — value-identical to the per-field pads (the exchange
packages/core/legoesm/parallel/latlon_spmd.py-651-    is a bit-copy; flatten/concat/split are layout ops).  This is the SPMD
packages/core/legoesm/parallel/latlon_spmd.py-652-    leg of the message-aggregation lever (audit item 7): the mpi4jax leg
packages/core/legoesm/parallel/latlon_spmd.py:653:    already fuses via ``pad_with_pole_bc_lat_multi_mpi``, the SPMD leg
packages/core/legoesm/parallel/latlon_spmd.py-654-    expanded per field.
packages/core/legoesm/parallel/latlon_spmd.py-655-
packages/core/legoesm/parallel/latlon_spmd.py-656-    STATIC group signature: ``n_fields`` (+ each field's dtype/shape at
--
packages/core/legoesm/parallel/latlon_spmd.py-879-        exchange, moves edge blocks of that depth in ONE ppermute hop).
packages/core/legoesm/parallel/latlon_spmd.py-880-
packages/core/legoesm/parallel/latlon_spmd.py-881-    Score = the per-tile RECEIVED communication volume of one full halo pad,
packages/core/legoesm/parallel/latlon_spmd.py:882:    normalized per unit halo depth (volume / 2h), with ``nl = n_lat/p_lat``,
packages/core/legoesm/parallel/latlon_spmd.py-883-    ``w = n_lon/p_lon``:
packages/core/legoesm/parallel/latlon_spmd.py-884-
packages/core/legoesm/parallel/latlon_spmd.py-885-      * lat cut (``p_lat > 1``): the N/S ppermute pair moves ``2h*w`` cells
packages/core/legoesm/parallel/latlon_spmd.py-886-        -> ``w``;
packages/core/legoesm/parallel/latlon_spmd.py:887:      * lon cut (``p_lon > 1``): the E/W ring ppermute pair moves ``~2h*nl``
packages/core/legoesm/parallel/latlon_spmd.py-888-        -> ``nl``, PLUS the pole-fold term, which depends on ``p_lon``
packages/core/legoesm/parallel/latlon_spmd.py-889-        parity (``make_latlon_2d_pad_body._fold_rows``, uniform program —
packages/core/legoesm/parallel/latlon_spmd.py-890-        the fold collective runs on EVERY tile because both operands of the
--
packages/core/legoesm/grids/voronoi.py-1109-# Levels <= ROUTINE build freely (cheap). ROUTINE+1 .. MAX are
packages/core/legoesm/grids/voronoi.py-1110-# cache-or-prewarm ONLY: a hit loads; a miss raises unless this process is
packages/core/legoesm/grids/voronoi.py-1111-# the designated single builder (env below + exclusive lockfile — the
packages/core/legoesm/grids/voronoi.py:1112:# opt-in alone would be a thundering herd across an MPI launch). > MAX is
packages/core/legoesm/grids/voronoi.py-1113-# hard-refused (42M+ cells; the scipy SphericalVoronoi path does not scale
packages/core/legoesm/grids/voronoi.py-1114-# there). Build cost measured 2026-07-29: one Lloyd iteration = 21.6 s at
packages/core/legoesm/grids/voronoi.py-1115-# subdiv-7, 87.1 s at subdiv-8 (~4x per level).
--
packages/core/legoesm/grids/voronoi.py-1494-                f"(lon_range={lon_range})"
packages/core/legoesm/grids/voronoi.py-1495-            )
packages/core/legoesm/grids/voronoi.py-1496-        # Exact integer number of cells around the periodic extent.
packages/core/legoesm/grids/voronoi.py:1497:        # Force ``n_lon`` even so that the hex row-offset staggering
packages/core/legoesm/grids/voronoi.py-1498-        # (lons shifted by dlon/2 on odd rows) closes consistently at
packages/core/legoesm/grids/voronoi.py-1499-        # the seam.
packages/core/legoesm/grids/voronoi.py-1500-        n_lon = max(2, int(np.round(L_rad / dlon)))
--
packages/core/legoesm/core/operators_cdgrid.py-725-
packages/core/legoesm/core/operators_cdgrid.py-726-    Used by the 4D ``cgrid_mass_flux_divergence`` to compute per-level
packages/core/legoesm/core/operators_cdgrid.py-727-    fluxes inside ``jax.vmap``.  The 4D entry then synchronizes the
packages/core/legoesm/core/operators_cdgrid.py:728:    stacked 4D fluxes (one MPI sendrecv exchange total, vs ``nlev``
packages/core/legoesm/core/operators_cdgrid.py-729-    inside vmap which mpi4jax's batch-axis rule refuses).
packages/core/legoesm/core/operators_cdgrid.py-730-
packages/core/legoesm/core/operators_cdgrid.py-731-    Thin wrapper over :func:`cgrid_ppm_fluxes_core` (``halo_in=2``); no
--
packages/core/legoesm/core/operators_cdgrid.py-750-    FV3_3D iter-1042: 3D path pre-pads halos OUTSIDE the per-level
packages/core/legoesm/core/operators_cdgrid.py-751-    ``jax.vmap`` and threads the padded array through the
packages/core/legoesm/core/operators_cdgrid.py-752-    ``h_pad`` kwarg.  Required for MPI fidelity: ``mpi4jax``'s
packages/core/legoesm/core/operators_cdgrid.py:753:    sendrecv batching rule asserts matching batch axes on the
packages/core/legoesm/core/operators_cdgrid.py-754-    send/recv buffers, which fails when ``pad_halo`` is invoked
packages/core/legoesm/core/operators_cdgrid.py-755-    inside ``vmap``.  Mirrors the pre-existing pattern in
packages/core/legoesm/core/operators_cdgrid.py-756-    :func:`_cgrid_fct_fluxes_2d` (pad-once-then-vmap).
packages/core/legoesm/core/operators_cdgrid.py-757-    """
packages/core/legoesm/core/operators_cdgrid.py-758-    if h.ndim == 4:
packages/core/legoesm/core/operators_cdgrid.py:759:        # 3D: pad halos ONCE for all levels (avoids MPI sendrecv inside vmap)
packages/core/legoesm/core/operators_cdgrid.py-760-        h_pad_4d = _pad_halo_auto_h2(h, cdgrid)  # (6, n+4, n+4, nlev)
packages/core/legoesm/core/operators_cdgrid.py-761-        h_t = jnp.moveaxis(h, -1, 0)
packages/core/legoesm/core/operators_cdgrid.py-762-        u_c_t = jnp.moveaxis(u_c, -1, 0)
--
packages/core/legoesm/core/operators_cdgrid.py-765-
packages/core/legoesm/core/operators_cdgrid.py-766-        # FV3_3D iter-1049: per-level vmap computes fluxes only (no
packages/core/legoesm/core/operators_cdgrid.py-767-        # sync, no divergence) — the duogrid flux synchronization
packages/core/legoesm/core/operators_cdgrid.py:768:        # involves an MPI sendrecv that cannot run inside ``vmap``
packages/core/legoesm/core/operators_cdgrid.py-769-        # (mpi4jax batch-axis assertion).  Lift sync + divergence to
packages/core/legoesm/core/operators_cdgrid.py-770-        # the 4D level after the vmap.
packages/core/legoesm/core/operators_cdgrid.py-771-        def flux_per_level(args):
--
packages/core/legoesm/grids/halo.py-518-    This is a TRACE-TIME switch for wide-halo compute regions: after a code
packages/core/legoesm/grids/halo.py-519-    path has already exchanged a wide halo (width = its full stencil reach),
packages/core/legoesm/grids/halo.py-520-    its interior operators must run their internal pads as plain local
packages/core/legoesm/grids/halo.py:521:    ``jnp.pad`` — re-dispatching them to MPI sendrecv / SPMD ppermute would
packages/core/legoesm/grids/halo.py-522-    re-communicate every substep, defeating the wide exchange (and, on the
packages/core/legoesm/grids/halo.py-523-    extended arrays, exchange the WRONG rows).  The flag is read when the
packages/core/legoesm/grids/halo.py-524-    operators trace, so wrap the traced region, not the runtime call.
--
packages/core/legoesm/grids/halo.py-2420-
packages/core/legoesm/grids/halo.py-2421-    FV3_3D iter-1049: under the MPI backend (``_halo_backend == "mpi"``)
packages/core/legoesm/grids/halo.py-2422-    we dispatch to :func:`_synchronize_cgrid_fluxes_mpi`, which uses
packages/core/legoesm/grids/halo.py:2423:    ``mpi4jax.sendrecv`` to swap boundary flux strips between ranks
packages/core/legoesm/grids/halo.py-2424-    before averaging.  Without that swap, ``fx[nbr_face, ...]`` reads
packages/core/legoesm/grids/halo.py-2425-    of non-owned faces return values computed with zero halos and
packages/core/legoesm/grids/halo.py-2426-    contaminate the averaged result on OWNED faces.
--
packages/core/legoesm/grids/halo.py-2496-    For each owned face's edge, locate the neighbour face and its
packages/core/legoesm/grids/halo.py-2497-    cross-face edge.  If both endpoints live on this rank (local
packages/core/legoesm/grids/halo.py-2498-    edge), read directly from ``fx`` / ``fy`` — same as the
packages/core/legoesm/grids/halo.py:2499:    single-device path.  Otherwise issue an ``mpi4jax.sendrecv``
packages/core/legoesm/grids/halo.py-2500-    that swaps the local boundary flux strip with the neighbour
packages/core/legoesm/grids/halo.py-2501-    rank's strip from the corresponding edge.
packages/core/legoesm/grids/halo.py-2502-
--
packages/core/legoesm/grids/halo.py-2505-    Callers comparing across local/MPI must compare owned faces only
packages/core/legoesm/grids/halo.py-2506-    (the MPI-replicated-mode contract).
packages/core/legoesm/grids/halo.py-2507-    """
packages/core/legoesm/grids/halo.py:2508:    from legoesm.parallel.halo_exchange import get_sendrecv_vjp
packages/core/legoesm/grids/halo.py-2509-    from collections import defaultdict
packages/core/legoesm/grids/halo.py-2510-    try:
packages/core/legoesm/grids/halo.py-2511-        import mpi4jax
--
packages/core/legoesm/grids/halo.py-2514-        raise ImportError(
packages/core/legoesm/grids/halo.py-2515-            "MPI synchronize_cgrid_fluxes requires mpi4jax + mpi4py."
packages/core/legoesm/grids/halo.py-2516-        ) from exc
packages/core/legoesm/grids/halo.py:2517:    sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/grids/halo.py-2518-    comm = _MPI.COMM_WORLD
packages/core/legoesm/grids/halo.py-2519-    rank = topology.rank
packages/core/legoesm/grids/halo.py-2520-
--
packages/core/legoesm/grids/halo.py-2534-    def _bdy_strip(face, edge):
packages/core/legoesm/grids/halo.py-2535-        return _extract_cgrid_boundary(fx, fy, face, edge, n)
packages/core/legoesm/grids/halo.py-2536-
packages/core/legoesm/grids/halo.py:2537:    # FV3_3D iter-1051: batched-per-neighbour sendrecv.  The iter-1049
packages/core/legoesm/grids/halo.py:2538:    # per-edge sendrecv pattern deadlocked at runtime because each
packages/core/legoesm/grids/halo.py:2539:    # ``mpi4jax.sendrecv`` blocks on its own ``recv`` until the peer
packages/core/legoesm/grids/halo.py:2540:    # issues a matching call.  With multiple sendrecvs per neighbour
packages/core/legoesm/grids/halo.py-2541-    # rank, both ranks block on their first call's recv waiting for
packages/core/legoesm/grids/halo.py-2542-    # the other's later send → deadlock.  The pad_halo_mpi pattern
packages/core/legoesm/grids/halo.py-2543-    # packs ALL strips for a given neighbour into ONE contiguous
packages/core/legoesm/grids/halo.py-2544-    # send/recv buffer (sorted canonically: send by ``(nbr_face,
packages/core/legoesm/grids/halo.py:2545:    # nbr_edge)``, recv by ``(face, edge)``).  One sendrecv per
packages/core/legoesm/grids/halo.py-2546-    # peer rank — at most 3 peers in face-only mode, so at most 3
packages/core/legoesm/grids/halo.py-2547-    # MPI calls per sync.
packages/core/legoesm/grids/halo.py-2548-
--
packages/core/legoesm/grids/halo.py-2561-    for entry in remote_edges:
packages/core/legoesm/grids/halo.py-2562-        by_nbr_rank[entry[5]].append(entry)
packages/core/legoesm/grids/halo.py-2563-
packages/core/legoesm/grids/halo.py:2564:    # Per-neighbour single sendrecv with canonically-ordered batched
packages/core/legoesm/grids/halo.py-2565-    # strips.  Send ordered by ``(nbr_face, nbr_edge)`` = the peer's
packages/core/legoesm/grids/halo.py-2566-    # local-edge identity, so the peer's recv-side canonical order
packages/core/legoesm/grids/halo.py-2567-    # (sorted by ``(face, edge)``) matches.  Tag is the peer rank
--
packages/core/legoesm/grids/halo.py-2572-    # 4D).  Pack by stacking on axis 0; unpack by slicing axis 0
packages/core/legoesm/grids/halo.py-2573-    # in chunks of ``n``.
packages/core/legoesm/grids/halo.py-2574-    # FV3_3D iter-1053: iterate peers in ascending rank order so all
packages/core/legoesm/grids/halo.py:2575:    # ranks issue sendrecv calls in the same global peer-sequence
packages/core/legoesm/grids/halo.py-2576-    # (mirror of the iter-1053 fix applied to ``_sync_dgrid_boundary_mpi``).
packages/core/legoesm/grids/halo.py-2577-    for nbr_rank in sorted(by_nbr_rank.keys()):
packages/core/legoesm/grids/halo.py-2578-        entries = by_nbr_rank[nbr_rank]
--
packages/core/legoesm/grids/halo.py-2584-
packages/core/legoesm/grids/halo.py-2585-        send_tag = rank
packages/core/legoesm/grids/halo.py-2586-        recv_tag = nbr_rank
packages/core/legoesm/grids/halo.py:2587:        recv_buf = sendrecv(
packages/core/legoesm/grids/halo.py-2588-            send_buf, jnp.zeros_like(send_buf),
packages/core/legoesm/grids/halo.py-2589-            nbr_rank, nbr_rank,
packages/core/legoesm/grids/halo.py-2590-            send_tag, recv_tag, comm,
--
packages/core/legoesm/grids/fv3_native_halos.py-105-
packages/core/legoesm/grids/fv3_native_halos.py-106-
packages/core/legoesm/grids/fv3_native_halos.py-107-def ed_supergrid_lonlat_ref(n: int):
packages/core/legoesm/grids/fv3_native_halos.py:108:    """Compute-domain ED supergrid lon/lat, REFERENCE face numbering.
packages/core/legoesm/grids/fv3_native_halos.py-109-
packages/core/legoesm/grids/fv3_native_halos.py-110-    Returns ``lon6, lat6`` of shape (6, 2n+1, 2n+1) over supergrid nodes
packages/core/legoesm/grids/fv3_native_halos.py-111-    1..2n+1 (index 0 == supergrid 1) — every stagger's physical positions
--
tests/distributed/test_fused_halo_mpi.py-5-Covers what the serial suite cannot:
tests/distributed/test_fused_halo_mpi.py-6-  * fused multi-field exchange == per-field single exchanges at a REAL
tests/distributed/test_fused_halo_mpi.py-7-    interior partition cut (values from the neighbour rank);
tests/distributed/test_fused_halo_mpi.py:8:  * AD parity through the fused sendrecv (``_sendrecv_vjp`` path);
tests/distributed/test_fused_halo_mpi.py-9-  * static-metric trace-time folding: a concrete (non-Tracer) pad
tests/distributed/test_fused_halo_mpi.py:10:    inside ``jax.jit`` produces NO ``mpi_sendrecv`` custom call in the
tests/distributed/test_fused_halo_mpi.py-11-    compiled HLO (tripwire) while matching the traced-path values.
tests/distributed/test_fused_halo_mpi.py-12-"""
tests/distributed/test_fused_halo_mpi.py-13-
--
tests/distributed/test_fused_halo_mpi.py-56-def test_fused_matches_singles_at_cut():
tests/distributed/test_fused_halo_mpi.py-57-    from legoesm.parallel.latlon_mpi import (
tests/distributed/test_fused_halo_mpi.py-58-        pad_with_pole_bc_lat_mpi,
tests/distributed/test_fused_halo_mpi.py:59:        pad_with_pole_bc_lat_multi_mpi,
tests/distributed/test_fused_halo_mpi.py-60-    )
tests/distributed/test_fused_halo_mpi.py-61-
tests/distributed/test_fused_halo_mpi.py-62-    layout = _layout()
tests/distributed/test_fused_halo_mpi.py-63-    f2d, f3d = _band_fields(layout)
tests/distributed/test_fused_halo_mpi.py-64-    sv, nv = (1.0e30, 0.0), (1.0e30, -3.0)
tests/distributed/test_fused_halo_mpi.py-65-
tests/distributed/test_fused_halo_mpi.py:66:    fused = pad_with_pole_bc_lat_multi_mpi(
tests/distributed/test_fused_halo_mpi.py-67-        (f2d, f3d), layout, halo=1, south_values=sv, north_values=nv,
tests/distributed/test_fused_halo_mpi.py-68-    )
tests/distributed/test_fused_halo_mpi.py-69-    singles = tuple(
--
tests/distributed/test_fused_halo_mpi.py-79-def test_fused_halo2_matches_singles():
tests/distributed/test_fused_halo_mpi.py-80-    from legoesm.parallel.latlon_mpi import (
tests/distributed/test_fused_halo_mpi.py-81-        pad_with_pole_bc_lat_mpi,
tests/distributed/test_fused_halo_mpi.py:82:        pad_with_pole_bc_lat_multi_mpi,
tests/distributed/test_fused_halo_mpi.py-83-    )
tests/distributed/test_fused_halo_mpi.py-84-
tests/distributed/test_fused_halo_mpi.py-85-    layout = _layout()
tests/distributed/test_fused_halo_mpi.py-86-    f2d, f3d = _band_fields(layout, key=1)
tests/distributed/test_fused_halo_mpi.py:87:    fused = pad_with_pole_bc_lat_multi_mpi((f2d, f3d), layout, halo=2)
tests/distributed/test_fused_halo_mpi.py-88-    for got, f in zip(fused, (f2d, f3d)):
tests/distributed/test_fused_halo_mpi.py-89-        want = pad_with_pole_bc_lat_mpi(f, layout, halo=2)
tests/distributed/test_fused_halo_mpi.py-90-        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
--
tests/distributed/test_fused_halo_mpi.py-93-def test_fused_grad_matches_singles_grad():
tests/distributed/test_fused_halo_mpi.py-94-    from legoesm.parallel.latlon_mpi import (
tests/distributed/test_fused_halo_mpi.py-95-        pad_with_pole_bc_lat_mpi,
tests/distributed/test_fused_halo_mpi.py:96:        pad_with_pole_bc_lat_multi_mpi,
tests/distributed/test_fused_halo_mpi.py-97-    )
tests/distributed/test_fused_halo_mpi.py-98-
tests/distributed/test_fused_halo_mpi.py-99-    layout = _layout()
tests/distributed/test_fused_halo_mpi.py-100-    f2d, f3d = _band_fields(layout, key=2)
tests/distributed/test_fused_halo_mpi.py-101-
tests/distributed/test_fused_halo_mpi.py-102-    def loss_fused(a, b):
tests/distributed/test_fused_halo_mpi.py:103:        pa, pb = pad_with_pole_bc_lat_multi_mpi((a, b), layout, halo=1)
tests/distributed/test_fused_halo_mpi.py-104-        return jnp.sum(pa**2) + jnp.sum(pb**2)
tests/distributed/test_fused_halo_mpi.py-105-
tests/distributed/test_fused_halo_mpi.py-106-    def loss_singles(a, b):
--
tests/distributed/test_fused_halo_mpi.py-115-
tests/distributed/test_fused_halo_mpi.py-116-
tests/distributed/test_fused_halo_mpi.py-117-def test_static_metric_pad_folds_to_constant():
tests/distributed/test_fused_halo_mpi.py:118:    """Concrete input -> trace-time host exchange -> NO per-step sendrecv."""
tests/distributed/test_fused_halo_mpi.py-119-    from legoesm.parallel.latlon_mpi import pad_with_pole_bc_lat_mpi
tests/distributed/test_fused_halo_mpi.py-120-
tests/distributed/test_fused_halo_mpi.py-121-    layout = _layout()
--
tests/distributed/test_fused_halo_mpi.py-135-
tests/distributed/test_fused_halo_mpi.py-136-    lowered = f.lower(jnp.float64(1.0))
tests/distributed/test_fused_halo_mpi.py-137-    hlo = lowered.compile().as_text()
tests/distributed/test_fused_halo_mpi.py:138:    assert "mpi_sendrecv" not in hlo, (
tests/distributed/test_fused_halo_mpi.py:139:        "static-metric pad was traced into a per-step sendrecv — "
tests/distributed/test_fused_halo_mpi.py-140-        "constant folding regressed"
tests/distributed/test_fused_halo_mpi.py-141-    )
tests/distributed/test_fused_halo_mpi.py-142-
--
tests/distributed/test_fused_halo_mpi.py-189-
tests/distributed/test_fused_halo_mpi.py-190-
tests/distributed/test_fused_halo_mpi.py-191-def test_static_fold_is_1d_only():
tests/distributed/test_fused_halo_mpi.py:192:    """ndim >= 2 concrete fields keep the traced sendrecv path
tests/distributed/test_fused_halo_mpi.py-193-    (codex CRITICAL hardening: narrow static surface to 1-D metrics)."""
tests/distributed/test_fused_halo_mpi.py-194-    from legoesm.parallel.latlon_mpi import pad_with_pole_bc_lat_mpi
tests/distributed/test_fused_halo_mpi.py-195-
--
tests/distributed/test_fused_halo_mpi.py-205-        )
tests/distributed/test_fused_halo_mpi.py-206-
tests/distributed/test_fused_halo_mpi.py-207-    hlo = f.lower(jnp.float64(1.0)).compile().as_text()
tests/distributed/test_fused_halo_mpi.py:208:    assert "mpi_sendrecv" in hlo  # 2-D concrete stays traced
tests/distributed/test_fused_halo_mpi.py-209-    f(jnp.float64(1.0)).block_until_ready()
tests/distributed/test_fused_halo_mpi.py-210-
tests/distributed/test_fused_halo_mpi.py-211-
--
tests/distributed/test_fused_halo_mpi.py-226-        return x * jnp.sum(pad)
tests/distributed/test_fused_halo_mpi.py-227-
tests/distributed/test_fused_halo_mpi.py-228-    hlo = f.lower(jnp.float64(1.0)).compile().as_text()
tests/distributed/test_fused_halo_mpi.py:229:    assert "mpi_sendrecv" in hlo  # traced path restored
tests/distributed/test_fused_halo_mpi.py-230-    f(jnp.float64(1.0)).block_until_ready()
--
tests/distributed/test_latlon_mpi_tripole.py-15-  * the northernmost rank's north halo == the serial tripolar fold of
tests/distributed/test_latlon_mpi_tripole.py-16-    the global field (``_fold_tripolar_north``) — bit-exact;
tests/distributed/test_latlon_mpi_tripole.py-17-  * an interior rank's north halo == the neighbour's first interior
tests/distributed/test_latlon_mpi_tripole.py:18:    row(s) (sendrecv continuity);
tests/distributed/test_latlon_mpi_tripole.py-19-  * the south halo of a non-south rank == the neighbour's last row(s).
tests/distributed/test_latlon_mpi_tripole.py-20-
tests/distributed/test_latlon_mpi_tripole.py-21-Single-rank counterparts live in
--
packages/core/legoesm/core/field.py-42-        Grid staggering: "cell", "edge", or "vertex". Default "cell".
packages/core/legoesm/core/field.py-43-    """
packages/core/legoesm/core/field.py-44-
packages/core/legoesm/core/field.py:45:    __slots__ = ("data", "name", "dims", "units", "long_name", "staggering")
packages/core/legoesm/core/field.py-46-
packages/core/legoesm/core/field.py-47-    def __init__(
packages/core/legoesm/core/field.py-48-        self,
--
packages/core/legoesm/core/field.py-74-        to preserve metadata compatibility.
packages/core/legoesm/core/field.py-75-        """
packages/core/legoesm/core/field.py-76-        children = (self.data,)
packages/core/legoesm/core/field.py:77:        aux_data = (self.name, self.dims, self.units, self.long_name, self.staggering)
packages/core/legoesm/core/field.py-78-        return children, aux_data
packages/core/legoesm/core/field.py-79-
packages/core/legoesm/core/field.py-80-    @classmethod
packages/core/legoesm/core/field.py-81-    def tree_unflatten(cls, aux_data, children):
packages/core/legoesm/core/field.py-82-        """Reconstruct Field from flattened representation."""
packages/core/legoesm/core/field.py-83-        (data,) = children
packages/core/legoesm/core/field.py:84:        name, dims, units, long_name, staggering = aux_data
packages/core/legoesm/core/field.py-85-        return cls(data=data, name=name, dims=dims, units=units,
packages/core/legoesm/core/field.py:86:                   long_name=long_name, staggering=staggering)
packages/core/legoesm/core/field.py-87-
packages/core/legoesm/core/field.py-88-    # ---- Convenience methods ----
packages/core/legoesm/core/field.py-89-
--
packages/core/legoesm/core/field.py-189-    return Field(
packages/core/legoesm/core/field.py-190-        data=jnp.zeros(shape, dtype=dtype),
packages/core/legoesm/core/field.py-191-        name=name, dims=dims, units=units,
packages/core/legoesm/core/field.py:192:        long_name=long_name, staggering=staggering,
packages/core/legoesm/core/field.py-193-    )
packages/core/legoesm/core/field.py-194-
packages/core/legoesm/core/field.py-195-
--
packages/core/legoesm/core/field.py-212-    return Field(
packages/core/legoesm/core/field.py-213-        data=jnp.ones(shape, dtype=dtype),
packages/core/legoesm/core/field.py-214-        name=name, dims=dims, units=units,
packages/core/legoesm/core/field.py:215:        long_name=long_name, staggering=staggering,
packages/core/legoesm/core/field.py-216-    )
--
packages/core/legoesm/core/fv3_del6_vt_flux.py-131-    offsets = None if dg is not None else cdgrid.base.halo_interp_offsets
packages/core/legoesm/core/fv3_del6_vt_flux.py-132-
packages/core/legoesm/core/fv3_del6_vt_flux.py-133-    # FV3_3D iter-1045: ndim-aware halo dispatch.  For 4D ``q`` use
packages/core/legoesm/core/fv3_del6_vt_flux.py:134:    # ``pad_halo_4d`` (one batched MPI sendrecv per call); for 3D use
packages/core/legoesm/core/fv3_del6_vt_flux.py-135-    # the legacy ``pad_halo``.  3D static metrics ``del6_u``, ``del6_v``,
packages/core/legoesm/core/fv3_del6_vt_flux.py-136-    # ``rarea`` get a trailing ``[..., None]`` axis for broadcasting
packages/core/legoesm/core/fv3_del6_vt_flux.py-137-    # against 4D data.  Closes the last documented vmap-around-
--
packages/core/legoesm/grids/gaussian.py-804-
packages/core/legoesm/grids/gaussian.py-805-    In spectral space:
packages/core/legoesm/grids/gaussian.py-806-        d/d(theta) -> multiply by Hnm (derivative Legendre)
packages/core/legoesm/grids/gaussian.py:807:        d/d(lon) -> multiply by im (applied during synthesis)
packages/core/legoesm/grids/gaussian.py-808-
packages/core/legoesm/grids/gaussian.py-809-    Parameters
packages/core/legoesm/grids/gaussian.py-810-    ----------
--
packages/core/legoesm/grids/dgrid_halo.py-43-  per-timestep hot path (codex review P3).
packages/core/legoesm/grids/dgrid_halo.py-44-- ``pad_halo_dgrid_vector_4d_mpi(u_d, v_d, topology)``
packages/core/legoesm/grids/dgrid_halo.py-45-  (iter-1083) — MPI-aware variant via batched-per-peer
packages/core/legoesm/grids/dgrid_halo.py:46:  ``mpi4jax.sendrecv``.  Rank-local input ``(n_local, ...)``.
packages/core/legoesm/grids/dgrid_halo.py-47-- ``pad_halo_dgrid_vector_4d_replicated_mpi(u_d, v_d, topology)``
packages/core/legoesm/grids/dgrid_halo.py-48-  (iter-1083) — wrapper for full ``(6, ...)`` replicated state
packages/core/legoesm/grids/dgrid_halo.py-49-  (canonical cubed-sphere MPI mode).
--
packages/core/legoesm/grids/dgrid_halo.py-704-
packages/core/legoesm/grids/dgrid_halo.py-705-
packages/core/legoesm/grids/dgrid_halo.py-706-# =====================================================================
packages/core/legoesm/grids/dgrid_halo.py:707:# iter-1083: MPI-aware DGRID vector halo (batched-per-peer sendrecv)
packages/core/legoesm/grids/dgrid_halo.py-708-# =====================================================================
packages/core/legoesm/grids/dgrid_halo.py-709-
packages/core/legoesm/grids/dgrid_halo.py-710-
--
packages/core/legoesm/grids/dgrid_halo.py-753-
packages/core/legoesm/grids/dgrid_halo.py-754-
packages/core/legoesm/grids/dgrid_halo.py-755-def pad_halo_dgrid_vector_4d_mpi(u_d, v_d, topology):
packages/core/legoesm/grids/dgrid_halo.py:756:    """MPI-aware DGRID vector halo with batched-per-peer sendrecv.
packages/core/legoesm/grids/dgrid_halo.py-757-
packages/core/legoesm/grids/dgrid_halo.py-758-    Mirrors the deadlock-free pattern from
packages/core/legoesm/grids/dgrid_halo.py-759-    ``_pad_halo_mpi_face_only_4d`` (sorted peer iteration, one
packages/core/legoesm/grids/dgrid_halo.py:760:    sendrecv per peer with packed multi-edge buffer).  Same-axis +
packages/core/legoesm/grids/dgrid_halo.py-761-    axis-swap edges are encoded into the strip pack/unpack and the
packages/core/legoesm/grids/dgrid_halo.py-762-    iter-1078 component-swap signs.
packages/core/legoesm/grids/dgrid_halo.py-763-
--
packages/core/legoesm/grids/dgrid_halo.py-842-    if not remote_edges:
packages/core/legoesm/grids/dgrid_halo.py-843-        return u_padded, v_padded
packages/core/legoesm/grids/dgrid_halo.py-844-
packages/core/legoesm/grids/dgrid_halo.py:845:    # ---- Remote edges: batched-per-peer sendrecv ----
packages/core/legoesm/grids/dgrid_halo.py-846-    from collections import defaultdict
packages/core/legoesm/grids/dgrid_halo.py-847-    comm = MPI.COMM_WORLD
packages/core/legoesm/grids/dgrid_halo.py-848-    rank = topology.rank
--
packages/core/legoesm/grids/dgrid_halo.py-871-            send_parts.append(v_strip.reshape(-1))
packages/core/legoesm/grids/dgrid_halo.py-872-        send_buf = jnp.concatenate(send_parts)
packages/core/legoesm/grids/dgrid_halo.py-873-
packages/core/legoesm/grids/dgrid_halo.py:874:        # Single sendrecv per neighbor rank.  Use the AD-safe
packages/core/legoesm/grids/dgrid_halo.py:875:        # ``_sendrecv_vjp`` wrapper (custom_vjp) so jax.grad flows
packages/core/legoesm/grids/dgrid_halo.py:876:        # through MPI sendrecv (raw mpi4jax.sendrecv chokes on the
packages/core/legoesm/grids/dgrid_halo.py-877-        # symbolic Zero cotangent JAX emits during backward).
packages/core/legoesm/grids/dgrid_halo.py:878:        from legoesm.parallel.halo_exchange import get_sendrecv_vjp
packages/core/legoesm/grids/dgrid_halo.py:879:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/grids/dgrid_halo.py:880:        recv_buf = sendrecv(
packages/core/legoesm/grids/dgrid_halo.py-881-            send_buf, jnp.zeros_like(send_buf),
packages/core/legoesm/grids/dgrid_halo.py-882-            nbr_rank, nbr_rank,
packages/core/legoesm/grids/dgrid_halo.py-883-            rank, nbr_rank, comm,
--
packages/core/legoesm/grids/dgrid_halo.py-937-    (the canonical cubed-sphere MPI mode per
packages/core/legoesm/grids/dgrid_halo.py-938-    ``driver/model_driver.py:1206``).  Returns full ``(6, n+2, n+3,
packages/core/legoesm/grids/dgrid_halo.py-939-    nlev)`` / ``(6, n+3, n+2, nlev)`` with this rank's OWNED face
packages/core/legoesm/grids/dgrid_halo.py:940:    halos correctly filled via cross-rank sendrecv; non-owned face
packages/core/legoesm/grids/dgrid_halo.py-941-    halos stay at edge-replicate (consumers should only use their
packages/core/legoesm/grids/dgrid_halo.py-942-    owned-face slices).
packages/core/legoesm/grids/dgrid_halo.py-943-
--
packages/core/legoesm/grids/fv3_native_metrics.py-78-def _spherical_angle(e1: np.ndarray, e2: np.ndarray, e3: np.ndarray) -> np.ndarray:
packages/core/legoesm/grids/fv3_native_metrics.py-79-    """Interior angle at e1 between great circles to e2 and e3.
packages/core/legoesm/grids/fv3_native_metrics.py-80-
packages/core/legoesm/grids/fv3_native_metrics.py:81:    Computed in extended precision (np.longdouble — 80-bit on x86), mirroring
packages/core/legoesm/grids/fv3_native_metrics.py-82-    FV3's quad-precision ``f_p`` intermediates: the spherical-excess area is
packages/core/legoesm/grids/fv3_native_metrics.py-83-    a tiny difference of angles near pi/2, and plain double precision leaves
packages/core/legoesm/grids/fv3_native_metrics.py-84-    ~1e-12 relative cancellation noise in C36 cell areas (measured 7.6e-13
--
packages/core/legoesm/grids/cubed_sphere.py-866-    (corner-derived), not construct-at-cell-centre-θ.
packages/core/legoesm/grids/cubed_sphere.py-867-    """
packages/core/legoesm/grids/cubed_sphere.py-868-    lon_c, lat_c = make_fv3_native_grid(n, grid_type=0)  # (6, n+1, n+1) corners
packages/core/legoesm/grids/cubed_sphere.py:869:    lon_c, lat_c = gnomonic_ed_remap_to_create(lon_c, lat_c)  # → create numbering
packages/core/legoesm/grids/cubed_sphere.py-870-    return cell_center2(
packages/core/legoesm/grids/cubed_sphere.py-871-        lon_c[:, :-1, :-1], lat_c[:, :-1, :-1],   # SW
packages/core/legoesm/grids/cubed_sphere.py-872-        lon_c[:, 1:, :-1], lat_c[:, 1:, :-1],     # SE
--
packages/core/legoesm/grids/halo_latlon.py-1-"""Halo exchange for the latitude-longitude grid.
packages/core/legoesm/grids/halo_latlon.py-2-
packages/core/legoesm/grids/halo_latlon.py-3-- Longitude: periodic wrap (every rank holds full lon — no decomposition)
packages/core/legoesm/grids/halo_latlon.py:4:- Latitude: pole-folding at boundary ranks; inter-rank sendrecv at
packages/core/legoesm/grids/halo_latlon.py-5-  interior partition cuts (MPI backend only)
packages/core/legoesm/grids/halo_latlon.py-6-
packages/core/legoesm/grids/halo_latlon.py-7-Backend dispatch
--
packages/core/legoesm/grids/halo_latlon.py-13-* ``"local"`` — single-rank / single-process: lon-wrap + pole-fold
packages/core/legoesm/grids/halo_latlon.py-14-  at both lat ends (the historical behaviour, kept unchanged).
packages/core/legoesm/grids/halo_latlon.py-15-* ``"mpi"`` — multi-rank lat-lon band: lon-wrap + pole-fold at
packages/core/legoesm/grids/halo_latlon.py:16:  pole-touching ranks, MPI sendrecv at interior partition cuts.
packages/core/legoesm/grids/halo_latlon.py-17-  The implementation lives in
packages/core/legoesm/grids/halo_latlon.py-18-  :mod:`legoesm.parallel.latlon_mpi`; this module dispatches
packages/core/legoesm/grids/halo_latlon.py-19-  there when the active topology is a ``LatLonBandLayout`` and
--
packages/core/legoesm/grids/halo_latlon.py-150-
packages/core/legoesm/grids/halo_latlon.py-151-    A 2-D ``("lat", "lon")`` mesh (M3a native 2-D tiling) routes to
packages/core/legoesm/grids/halo_latlon.py-152-    :func:`legoesm.parallel.latlon_spmd.make_latlon_2d_pad_body` — lat
packages/core/legoesm/grids/halo_latlon.py:153:    ppermute + periodic lon ring ppermute + the exact serial 180-deg pole
packages/core/legoesm/grids/halo_latlon.py:154:    fold (all_gather'd over the lon ring at the pole tiles).  A degenerate
packages/core/legoesm/grids/halo_latlon.py-155-    ``p_lon == 1`` lon axis takes the body's static local-wrap branch,
packages/core/legoesm/grids/halo_latlon.py-156-    bit-identical to the 1-D band body."""
packages/core/legoesm/grids/halo_latlon.py-157-    from legoesm.grids.halo import get_halo_backend, get_spmd_mesh
--
packages/core/legoesm/grids/halo_latlon.py-182-
packages/core/legoesm/grids/halo_latlon.py-183-
packages/core/legoesm/grids/halo_latlon.py-184-def _dispatch_latlon_2d_fold(data, topology, halo, *, is_vector_v):
packages/core/legoesm/grids/halo_latlon.py:185:    """Fold-family ``pad_halo_latlon*`` dispatch for a ``LatLon2DLayout``.
packages/core/legoesm/grids/halo_latlon.py-186-
packages/core/legoesm/grids/halo_latlon.py-187-    ``proc_lon == 1`` is a pure latitude band (every rank owns the full lon
packages/core/legoesm/grids/halo_latlon.py-188-    circle), so the 180-deg pole fold is LOCAL — reuse the validated band
--
packages/core/legoesm/grids/halo_latlon.py-227-    :class:`~legoesm.parallel.latlon_mpi.LatLonBandLayout` topology)
packages/core/legoesm/grids/halo_latlon.py-228-    routes through
packages/core/legoesm/grids/halo_latlon.py-229-    :func:`legoesm.parallel.latlon_mpi.pad_halo_latlon_mpi`, which
packages/core/legoesm/grids/halo_latlon.py:230:    does lon-wrap + pole-fold at boundary ranks + MPI sendrecv at
packages/core/legoesm/grids/halo_latlon.py-231-    interior partition cuts.  Callers stay backend-oblivious; same
packages/core/legoesm/grids/halo_latlon.py-232-    pattern as cubed-sphere ``pad_halo``.
packages/core/legoesm/grids/halo_latlon.py-233-
--
packages/core/legoesm/grids/halo_latlon.py-257-        # back to the local serial path rather than crashing in the
packages/core/legoesm/grids/halo_latlon.py-258-        # MPI dispatch with an opaque error.
packages/core/legoesm/grids/halo_latlon.py-259-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:260:            LatLon2DLayout, LatLonBandLayout, pad_halo_latlon_mpi,
packages/core/legoesm/grids/halo_latlon.py-261-        )
packages/core/legoesm/grids/halo_latlon.py-262-        if isinstance(topology, LatLonBandLayout):
packages/core/legoesm/grids/halo_latlon.py-263-            return pad_halo_latlon_mpi(
packages/core/legoesm/grids/halo_latlon.py-264-                data, topology, halo=halo, is_vector_v=False,
packages/core/legoesm/grids/halo_latlon.py-265-            )
packages/core/legoesm/grids/halo_latlon.py:266:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/halo_latlon.py-267-            return _dispatch_latlon_2d_fold(
packages/core/legoesm/grids/halo_latlon.py-268-                data, topology, halo, is_vector_v=False,
packages/core/legoesm/grids/halo_latlon.py-269-            )
--
packages/core/legoesm/grids/halo_latlon.py-311-    if get_halo_backend() == "mpi":
packages/core/legoesm/grids/halo_latlon.py-312-        topology = get_mpi_topology()
packages/core/legoesm/grids/halo_latlon.py-313-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:314:            LatLon2DLayout, LatLonBandLayout, pad_halo_latlon_mpi,
packages/core/legoesm/grids/halo_latlon.py-315-        )
packages/core/legoesm/grids/halo_latlon.py-316-        if isinstance(topology, LatLonBandLayout):
packages/core/legoesm/grids/halo_latlon.py-317-            return pad_halo_latlon_mpi(
packages/core/legoesm/grids/halo_latlon.py-318-                data, topology, halo=halo, is_vector_v=True,
packages/core/legoesm/grids/halo_latlon.py-319-            )
packages/core/legoesm/grids/halo_latlon.py:320:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/halo_latlon.py-321-            return _dispatch_latlon_2d_fold(
packages/core/legoesm/grids/halo_latlon.py-322-                data, topology, halo, is_vector_v=True,
packages/core/legoesm/grids/halo_latlon.py-323-            )
--
packages/core/legoesm/grids/halo_latlon.py-387-    if get_halo_backend() == "mpi":
packages/core/legoesm/grids/halo_latlon.py-388-        topology = get_mpi_topology()
packages/core/legoesm/grids/halo_latlon.py-389-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:390:            LatLon2DLayout, LatLonBandLayout, pad_halo_latlon_mpi,
packages/core/legoesm/grids/halo_latlon.py-391-        )
packages/core/legoesm/grids/halo_latlon.py-392-        if isinstance(topology, LatLonBandLayout):
packages/core/legoesm/grids/halo_latlon.py-393-            return pad_halo_latlon_mpi(
packages/core/legoesm/grids/halo_latlon.py-394-                data, topology, halo=halo, is_vector_v=False,
packages/core/legoesm/grids/halo_latlon.py-395-            )
packages/core/legoesm/grids/halo_latlon.py:396:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/halo_latlon.py-397-            return _dispatch_latlon_2d_fold(
packages/core/legoesm/grids/halo_latlon.py-398-                data, topology, halo, is_vector_v=False,
packages/core/legoesm/grids/halo_latlon.py-399-            )
--
packages/core/legoesm/grids/halo_latlon.py-420-    if get_halo_backend() == "mpi":
packages/core/legoesm/grids/halo_latlon.py-421-        topology = get_mpi_topology()
packages/core/legoesm/grids/halo_latlon.py-422-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:423:            LatLon2DLayout, LatLonBandLayout, pad_halo_latlon_mpi,
packages/core/legoesm/grids/halo_latlon.py-424-        )
packages/core/legoesm/grids/halo_latlon.py-425-        if isinstance(topology, LatLonBandLayout):
packages/core/legoesm/grids/halo_latlon.py-426-            return pad_halo_latlon_mpi(
packages/core/legoesm/grids/halo_latlon.py-427-                data, topology, halo=halo, is_vector_v=True,
packages/core/legoesm/grids/halo_latlon.py-428-            )
packages/core/legoesm/grids/halo_latlon.py:429:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/halo_latlon.py-430-            return _dispatch_latlon_2d_fold(
packages/core/legoesm/grids/halo_latlon.py-431-                data, topology, halo, is_vector_v=True,
packages/core/legoesm/grids/halo_latlon.py-432-            )
--
packages/core/legoesm/grids/halo_latlon.py-469-# execution the constant pad IS the correct boundary condition.
packages/core/legoesm/grids/halo_latlon.py-470-# Under latitude-band MPI, the same convention applies only at the
packages/core/legoesm/grids/halo_latlon.py-471-# pole-touching ranks; interior partition cuts must instead receive
packages/core/legoesm/grids/halo_latlon.py:472:# the neighbour rank's value via MPI sendrecv.
packages/core/legoesm/grids/halo_latlon.py-473-#
packages/core/legoesm/grids/halo_latlon.py-474-# We expose one helper, ``pad_with_pole_bc_lat``, and route every
packages/core/legoesm/grids/halo_latlon.py-475-# pole-BC-style pad through it (``pad_ns_zero`` and friends in
--
packages/core/legoesm/grids/halo_latlon.py-495-        (``layout.south_rank is None``); zeros index ``-1`` only if
packages/core/legoesm/grids/halo_latlon.py-496-        this rank touches the north pole.  Interior partition cuts
packages/core/legoesm/grids/halo_latlon.py-497-        are left intact so the cross-partition gradient computed
packages/core/legoesm/grids/halo_latlon.py:498:        from neighbour-sendrecv'd halo values is preserved.
packages/core/legoesm/grids/halo_latlon.py-499-
packages/core/legoesm/grids/halo_latlon.py-500-    Parameters
packages/core/legoesm/grids/halo_latlon.py-501-    ----------
--
packages/core/legoesm/grids/halo_latlon.py-524-    if get_halo_backend() == "mpi":
packages/core/legoesm/grids/halo_latlon.py-525-        topology = get_mpi_topology()
packages/core/legoesm/grids/halo_latlon.py-526-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:527:            LatLon2DLayout, LatLonBandLayout,
packages/core/legoesm/grids/halo_latlon.py-528-        )
packages/core/legoesm/grids/halo_latlon.py-529-        # Band AND 2-D pencil share the pole-touch test: a rank zeros its
packages/core/legoesm/grids/halo_latlon.py-530-        # south end only if it OWNS the south pole (``south_rank is None``)
packages/core/legoesm/grids/halo_latlon.py-531-        # and its north end only if it owns the north pole.  In the 2-D
packages/core/legoesm/grids/halo_latlon.py-532-        # pencil this is the proc_row-0 / proc_row-last test (the riskiest
packages/core/legoesm/grids/halo_latlon.py-533-        # 2-D bug — an interior proc row must NOT wall its lat cut, or the
packages/core/legoesm/grids/halo_latlon.py:534:        # cross-cut gradient sendrecv'd from the neighbour is destroyed).
packages/core/legoesm/grids/halo_latlon.py:535:        if isinstance(topology, (LatLonBandLayout, LatLon2DLayout)):
packages/core/legoesm/grids/halo_latlon.py-536-            out = field
packages/core/legoesm/grids/halo_latlon.py-537-            if topology.south_rank is None:
packages/core/legoesm/grids/halo_latlon.py-538-                out = out.at[0].set(jnp.zeros_like(out[0]))
--
packages/core/legoesm/grids/halo_latlon.py-568-    -----------
packages/core/legoesm/grids/halo_latlon.py-569-    Pole-touching south rank → pad with ``south_value``.
packages/core/legoesm/grids/halo_latlon.py-570-    Pole-touching north rank → pad with ``north_value``.
packages/core/legoesm/grids/halo_latlon.py:571:    Interior partition cuts → MPI sendrecv with the neighbour rank,
packages/core/legoesm/grids/halo_latlon.py-572-    reusing the AD-safe
packages/core/legoesm/grids/halo_latlon.py-573-    :func:`legoesm.parallel.latlon_mpi.exchange_halo_latlon` path.
packages/core/legoesm/grids/halo_latlon.py-574-    The ``is_vector_v`` flag is forwarded to the exchange so the
--
packages/core/legoesm/grids/halo_latlon.py-664-    # if the active backend is MPI but for a different grid, fall
packages/core/legoesm/grids/halo_latlon.py-665-    # back to the local serial pad.
packages/core/legoesm/grids/halo_latlon.py-666-    from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:667:        LatLon2DLayout,
packages/core/legoesm/grids/halo_latlon.py-668-        LatLonBandLayout,
packages/core/legoesm/grids/halo_latlon.py:669:        pad_with_pole_bc_lat_2d,
packages/core/legoesm/grids/halo_latlon.py-670-        pad_with_pole_bc_lat_mpi,
packages/core/legoesm/grids/halo_latlon.py-671-    )
packages/core/legoesm/grids/halo_latlon.py:672:    if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/halo_latlon.py:673:        # 2-D pencil: lat-axis-ONLY wall pad (interior lat cut sendrecv +
packages/core/legoesm/grids/halo_latlon.py-674-        # pole wall constant).  Longitude is left untouched — the band path
packages/core/legoesm/grids/halo_latlon.py-675-        # never split lon, so its wall-BC pad is lat-only; the 2-D path
packages/core/legoesm/grids/halo_latlon.py-676-        # keeps that contract and the operator adds lon ghosts through its
--
packages/core/legoesm/grids/halo_latlon.py-683-                "is_vector_u need the lat-pencil transpose (wall-pole 2-D "
packages/core/legoesm/grids/halo_latlon.py-684-                f"only). north_fold={north_fold!r} is_vector_u={is_vector_u!r}"
packages/core/legoesm/grids/halo_latlon.py-685-            )
packages/core/legoesm/grids/halo_latlon.py:686:        return pad_with_pole_bc_lat_2d(
packages/core/legoesm/grids/halo_latlon.py-687-            interior, topology, halo=halo,
packages/core/legoesm/grids/halo_latlon.py-688-            south_value=south_value, north_value=north_value,
packages/core/legoesm/grids/halo_latlon.py-689-        )
--
packages/core/legoesm/grids/halo_latlon.py-704-    )
packages/core/legoesm/grids/halo_latlon.py-705-
packages/core/legoesm/grids/halo_latlon.py-706-
packages/core/legoesm/grids/halo_latlon.py:707:def pad_with_pole_bc_lat_multi(
packages/core/legoesm/grids/halo_latlon.py-708-    fields,
packages/core/legoesm/grids/halo_latlon.py-709-    halo: int = 1,
packages/core/legoesm/grids/halo_latlon.py-710-    south_values=None,
--
packages/core/legoesm/grids/halo_latlon.py-716-    boundary values, backend-dispatched.  Value-identical to calling
packages/core/legoesm/grids/halo_latlon.py-717-    :func:`pad_with_pole_bc_lat` once per field — but under the MPI
packages/core/legoesm/grids/halo_latlon.py-718-    lat-lon band backend the interior partition cuts are exchanged in
packages/core/legoesm/grids/halo_latlon.py:719:    ONE fused sendrecv pair per cut per dtype group instead of one pair
packages/core/legoesm/grids/halo_latlon.py:720:    per field.  mpi4jax sendrecvs are token-serialized (no overlap), so
packages/core/legoesm/grids/halo_latlon.py:721:    each fused cluster of N pads saves ``(N-1) x 2`` sendrecv latencies
packages/core/legoesm/grids/halo_latlon.py-722-    per step — the measured rank-growing term of the ocean baroclinic
packages/core/legoesm/grids/halo_latlon.py-723-    phase (scaling campaign audit lever O4).
packages/core/legoesm/grids/halo_latlon.py-724-
--
packages/core/legoesm/grids/halo_latlon.py-757-    north_values = tuple(north_values)
packages/core/legoesm/grids/halo_latlon.py-758-    if len(south_values) != n or len(north_values) != n:
packages/core/legoesm/grids/halo_latlon.py-759-        raise ValueError(
packages/core/legoesm/grids/halo_latlon.py:760:            "pad_with_pole_bc_lat_multi: south_values/north_values must "
packages/core/legoesm/grids/halo_latlon.py-761-            f"match len(fields)={n}; got {len(south_values)}/"
packages/core/legoesm/grids/halo_latlon.py-762-            f"{len(north_values)}."
packages/core/legoesm/grids/halo_latlon.py-763-        )
--
packages/core/legoesm/grids/halo_latlon.py-788-    fused = os.environ.get("LEGOESM_LATLON_FUSED_HALO", "1") != "0"
packages/core/legoesm/grids/halo_latlon.py-789-    if get_halo_backend() == "mpi" and fused:
packages/core/legoesm/grids/halo_latlon.py-790-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:791:            LatLon2DLayout,
packages/core/legoesm/grids/halo_latlon.py-792-            LatLonBandLayout,
packages/core/legoesm/grids/halo_latlon.py:793:            pad_with_pole_bc_lat_multi_2d,
packages/core/legoesm/grids/halo_latlon.py:794:            pad_with_pole_bc_lat_multi_mpi,
packages/core/legoesm/grids/halo_latlon.py-795-        )
packages/core/legoesm/grids/halo_latlon.py-796-        topology = get_mpi_topology()
packages/core/legoesm/grids/halo_latlon.py-797-        if isinstance(topology, LatLonBandLayout):
packages/core/legoesm/grids/halo_latlon.py:798:            return pad_with_pole_bc_lat_multi_mpi(
packages/core/legoesm/grids/halo_latlon.py-799-                fields, topology, halo=halo,
packages/core/legoesm/grids/halo_latlon.py-800-                south_values=south_values, north_values=north_values,
packages/core/legoesm/grids/halo_latlon.py-801-            )
packages/core/legoesm/grids/halo_latlon.py:802:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/halo_latlon.py-803-            # 2-D pencil twin (codex consult #3, 2026-08-04): same
packages/core/legoesm/grids/halo_latlon.py-804-            # dtype-group fusion on the lat axis; lon stays a separate
packages/core/legoesm/grids/halo_latlon.py-805-            # dispatched exchange, as in the single-field 2-D path.
packages/core/legoesm/grids/halo_latlon.py:806:            return pad_with_pole_bc_lat_multi_2d(
packages/core/legoesm/grids/halo_latlon.py-807-                fields, topology, halo=halo,
packages/core/legoesm/grids/halo_latlon.py-808-                south_values=south_values, north_values=north_values,
packages/core/legoesm/grids/halo_latlon.py-809-            )
packages/core/legoesm/grids/halo_latlon.py-810-    # Local backend / non-latlon topology / fused-off: per-field pads
packages/core/legoesm/grids/halo_latlon.py-811-    # (bit-identical semantics; under band MPI this is the legacy
packages/core/legoesm/grids/halo_latlon.py:812:    # one-sendrecv-pair-per-field schedule).  The 2-D pencil now has its
packages/core/legoesm/grids/halo_latlon.py-813-    # own fused branch above (2026-08-04); it lands here only when the
packages/core/legoesm/grids/halo_latlon.py-814-    # fused gate is off.
packages/core/legoesm/grids/halo_latlon.py-815-    return tuple(
--
packages/core/legoesm/grids/halo_latlon.py-848-    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
packages/core/legoesm/grids/halo_latlon.py-849-    if get_halo_backend() == "mpi":
packages/core/legoesm/grids/halo_latlon.py-850-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:851:            LatLon2DLayout,
packages/core/legoesm/grids/halo_latlon.py-852-            LatLonBandLayout,
packages/core/legoesm/grids/halo_latlon.py-853-        )
packages/core/legoesm/grids/halo_latlon.py-854-        topology = get_mpi_topology()
packages/core/legoesm/grids/halo_latlon.py:855:        if isinstance(topology, (LatLonBandLayout, LatLon2DLayout)):
packages/core/legoesm/grids/halo_latlon.py-856-            return topology.south_rank is None, topology.north_rank is None
packages/core/legoesm/grids/halo_latlon.py-857-    return True, True
packages/core/legoesm/grids/halo_latlon.py-858-
--
packages/core/legoesm/grids/halo_latlon.py-888-    """Widen cell-row (leading dim ``n_lat``) fields by ``halo`` rows/side.
packages/core/legoesm/grids/halo_latlon.py-889-
packages/core/legoesm/grids/halo_latlon.py-890-    ONE fused exchange under the MPI band backend
packages/core/legoesm/grids/halo_latlon.py:891:    (:func:`pad_with_pole_bc_lat_multi`); wall-zero at physical poles,
packages/core/legoesm/grids/halo_latlon.py-892-    optionally clamped to the nearest physical row (metric arrays — see
packages/core/legoesm/grids/halo_latlon.py-893-    :func:`_clamp_pole_pad_rows`).  Applies to T-point AND u-point fields
packages/core/legoesm/grids/halo_latlon.py-894-    (both carry one row per cell).
packages/core/legoesm/grids/halo_latlon.py-895-    """
packages/core/legoesm/grids/halo_latlon.py:896:    padded = pad_with_pole_bc_lat_multi(fields, halo=halo)
packages/core/legoesm/grids/halo_latlon.py-897-    if not clamp_poles:
packages/core/legoesm/grids/halo_latlon.py-898-        return padded
packages/core/legoesm/grids/halo_latlon.py-899-    south_is_pole, north_is_pole = band_pole_flags()
--
packages/core/legoesm/grids/halo_latlon.py-926-    uniform under SPMD.
packages/core/legoesm/grids/halo_latlon.py-927-    """
packages/core/legoesm/grids/halo_latlon.py-928-    interiors = tuple(f[:-1] for f in fields)
packages/core/legoesm/grids/halo_latlon.py:929:    padded = pad_with_pole_bc_lat_multi(interiors, halo=halo + 1)
packages/core/legoesm/grids/halo_latlon.py-930-    out = tuple(
packages/core/legoesm/grids/halo_latlon.py-931-        p[1:].at[halo + f.shape[0] - 1].set(f[-1])
packages/core/legoesm/grids/halo_latlon.py-932-        for p, f in zip(padded, fields)
--
packages/core/legoesm/grids/fv3_native_gridstruct.py-1355-
packages/core/legoesm/grids/fv3_native_gridstruct.py-1356-def k2e_remap_halo_rings(f6: list, stag: str, n: int, ng: int,
packages/core/legoesm/grids/fv3_native_gridstruct.py-1357-                         k2e_nord: int = 4):
packages/core/legoesm/grids/fv3_native_gridstruct.py:1358:    """Kinked-to-extended along-edge Lagrange remap of the halo rings
packages/core/legoesm/grids/fv3_native_gridstruct.py-1359-    (cube_rmp semantics) for one stagger family, applied AFTER the
packages/core/legoesm/grids/fv3_native_gridstruct.py-1360-    index-copy exchange: each halo-ring value becomes the certified
packages/core/legoesm/grids/fv3_native_gridstruct.py-1361-    k2e interpolation of the copied (neighbour-line) ring at the
--
packages/core/legoesm/grids/nesting.py-186-
packages/core/legoesm/grids/nesting.py-187-def _interior_mask(n_lat: int, n_lon: int, n_halo: int) -> jax.Array:
packages/core/legoesm/grids/nesting.py-188-    m = jnp.zeros((n_lat, n_lon))
packages/core/legoesm/grids/nesting.py:189:    m = m.at[n_halo : n_lat - n_halo, n_halo : n_lon - n_halo].set(1.0)
packages/core/legoesm/grids/nesting.py-190-    return m
packages/core/legoesm/grids/nesting.py-191-
packages/core/legoesm/grids/nesting.py-192-
--
packages/core/legoesm/grids/nesting.py-594-    )
packages/core/legoesm/grids/nesting.py-595-
packages/core/legoesm/grids/nesting.py-596-    n_lat_c, n_lon_c = child.n_lat, child.n_lon
packages/core/legoesm/grids/nesting.py:597:    if 2 * (n_halo + n_relax) >= min(n_lat_c, n_lon_c):
packages/core/legoesm/grids/nesting.py-598-        raise ValueError(
packages/core/legoesm/grids/nesting.py-599-            f"n_halo={n_halo} + n_relax={n_relax} too large for child "
packages/core/legoesm/grids/nesting.py-600-            f"{n_lat_c}x{n_lon_c}: the prescribed band + relaxation zone would "
--
packages/core/legoesm/grids/polar_filter.py-28-# lon SLICE, so the filter must first gather the full lon axis, rFFT/mask/irFFT
packages/core/legoesm/grids/polar_filter.py-29-# on it, then scatter its block back.  The 2-D MPI step
packages/core/legoesm/grids/polar_filter.py-30-# (``make_latlon_2d_mpi_step``) injects an AD-safe gather/scatter pair here
packages/core/legoesm/grids/polar_filter.py:31:# (bound to its ``LatLon2DLayout`` + longitude row sub-communicator) BEFORE it
packages/core/legoesm/grids/polar_filter.py-32-# builds the jitted step; the filter reads this process-global at trace time,
packages/core/legoesm/grids/polar_filter.py-33-# exactly like the halo backend (``grids.halo.get_halo_backend``).  This keeps
packages/core/legoesm/grids/polar_filter.py-34-# ``polar_filter`` free of a ``parallel.latlon_mpi`` import (no cycle) and the
--
packages/core/legoesm/core/conservation.py-471-            # 2-D ("lat", "lon") tile mesh (M3a): every TILE holds a partial
packages/core/legoesm/core/conservation.py-472-            # sum — reduce across BOTH axes, or the "global" mass integral
packages/core/legoesm/core/conservation.py-473-            # would silently remain a per-lon-sector partial.  The 1-D band
packages/core/legoesm/core/conservation.py:474:            # mesh — and the degenerate (N, 1) tile mesh, whose lon rings
packages/core/legoesm/core/conservation.py-475-            # have one member — keep the bare "lat" psum (byte-unchanged /
packages/core/legoesm/core/conservation.py-476-            # structurally identical to the band program for the (N, 1)
packages/core/legoesm/core/conservation.py-477-            # bit-identity gate).
--
packages/core/legoesm/grids/latlon.py-401-    dtype=None,
packages/core/legoesm/grids/latlon.py-402-    periodic_x: bool = False,
packages/core/legoesm/grids/latlon.py-403-) -> tuple[LatLonGrid, jax.Array]:
packages/core/legoesm/grids/latlon.py:404:    """Create a regional lat-lon grid covering a limited domain.
packages/core/legoesm/grids/latlon.py-405-
packages/core/legoesm/grids/latlon.py-406-    The grid spans the specified lat/lon bounding box with *n_lat* x
packages/core/legoesm/grids/latlon.py-407-    *n_lon* interior cells.  By default, a 1-cell wall (land mask = 0)
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-31-
packages/core/legoesm/grids/operators_latlon_cgrid.py-32-    MPI backend (lat-lon band layout)
packages/core/legoesm/grids/operators_latlon_cgrid.py-33-        Pole-touching ranks pad their pole side with zero (wall BC at
packages/core/legoesm/grids/operators_latlon_cgrid.py:34:        the actual pole); interior partition cuts MPI-sendrecv with
packages/core/legoesm/grids/operators_latlon_cgrid.py-35-        the neighbour rank so the gradient / divergence stencil sees
packages/core/legoesm/grids/operators_latlon_cgrid.py-36-        continuous data across the cut.  Routes through
packages/core/legoesm/grids/operators_latlon_cgrid.py-37-        :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat` which
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-60-
packages/core/legoesm/grids/operators_latlon_cgrid.py-61-    Value-identical to ``tuple(pad_ns_zero(f) for f in fields)``; under
packages/core/legoesm/grids/operators_latlon_cgrid.py-62-    the MPI lat-lon band backend the interior partition cuts are
packages/core/legoesm/grids/operators_latlon_cgrid.py:63:    exchanged in ONE fused sendrecv pair per cut per dtype group instead
packages/core/legoesm/grids/operators_latlon_cgrid.py-64-    of one pair per field (see
packages/core/legoesm/grids/operators_latlon_cgrid.py:65:    :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat_multi`).
packages/core/legoesm/grids/operators_latlon_cgrid.py-66-    Use for clusters of pads at the same dataflow level (no data
packages/core/legoesm/grids/operators_latlon_cgrid.py-67-    dependency between the fields).
packages/core/legoesm/grids/operators_latlon_cgrid.py-68-    """
packages/core/legoesm/grids/operators_latlon_cgrid.py:69:    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat_multi
packages/core/legoesm/grids/operators_latlon_cgrid.py:70:    return pad_with_pole_bc_lat_multi(fields, halo=1)
packages/core/legoesm/grids/operators_latlon_cgrid.py-71-
packages/core/legoesm/grids/operators_latlon_cgrid.py-72-
packages/core/legoesm/grids/operators_latlon_cgrid.py-73-def _vface_cos_lat_core(grid) -> jnp.ndarray:
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-77-    ``divergence_cgrid`` and ``gradient_curl_to_v``: pad ``grid.lat`` with the
packages/core/legoesm/grids/operators_latlon_cgrid.py-78-    halo-aware pole/cut BC (``pad_with_pole_bc_lat`` — at an interior MPI band cut
packages/core/legoesm/grids/operators_latlon_cgrid.py-79-    the ghost row is the neighbour rank's true edge latitude via the AD-safe
packages/core/legoesm/grids/operators_latlon_cgrid.py:80:    sendrecv), take the v-face midpoint latitude, return its cosine.  Callers apply
packages/core/legoesm/grids/operators_latlon_cgrid.py-81-    their OWN pole step (``zero_polar_lat_ends`` vs a ``1e-30`` floor) and the
packages/core/legoesm/grids/operators_latlon_cgrid.py-82-    ``R*dlon`` scaling.  Kept on the STORED ``grid.lat`` dtype (NOT
packages/core/legoesm/grids/operators_latlon_cgrid.py-83-    ``result_type``-cast — unlike the ocean ``vface_zonal_cos_lat``) so the core
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-170-    On tripolar (fold.is_active): south = zero, north = fold-reflected.
packages/core/legoesm/grids/operators_latlon_cgrid.py-171-
packages/core/legoesm/grids/operators_latlon_cgrid.py-172-    Under MPI, all ranks call ``pad_ns_zero`` first (ensuring consistent
packages/core/legoesm/grids/operators_latlon_cgrid.py:173:    MPI sendrecv call counts), then the northernmost rank replaces the
packages/core/legoesm/grids/operators_latlon_cgrid.py-174-    north ghost row with fold-permuted data via :func:`fold_is_local`.
packages/core/legoesm/grids/operators_latlon_cgrid.py-175-
packages/core/legoesm/grids/operators_latlon_cgrid.py-176-    Parameters
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-276-
packages/core/legoesm/grids/operators_latlon_cgrid.py-277-    * local / band / SPMD-lat / non-2-D MPI — every rank owns the full
packages/core/legoesm/grids/operators_latlon_cgrid.py-278-      longitude circle, so the wrap is LOCAL: ``jnp.pad(mode="wrap")``.
packages/core/legoesm/grids/operators_latlon_cgrid.py:279:    * 2-D pencil (``LatLon2DLayout``) — longitude is split, so the wrap
packages/core/legoesm/grids/operators_latlon_cgrid.py-280-      becomes an MPI ring exchange with the W/E neighbour
packages/core/legoesm/grids/operators_latlon_cgrid.py-281-      (:func:`legoesm.parallel.latlon_mpi.exchange_halo_lon`).
packages/core/legoesm/grids/operators_latlon_cgrid.py-282-    * 2-D SPMD ``("lat", "lon")`` mesh (M3a) — the wrap becomes the cyclic
packages/core/legoesm/grids/operators_latlon_cgrid.py-283-      ring ``ppermute`` over the ``"lon"`` mesh axis
packages/core/legoesm/grids/operators_latlon_cgrid.py:284:      (:func:`legoesm.parallel.latlon_spmd.lon_ring_ghosts_spmd`); a
packages/core/legoesm/grids/operators_latlon_cgrid.py-285-      degenerate ``p_lon == 1`` axis takes that helper's STATIC local-wrap
packages/core/legoesm/grids/operators_latlon_cgrid.py-286-      branch (no collective — bit-identical to the band path).
packages/core/legoesm/grids/operators_latlon_cgrid.py-287-
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-290-    pad-then-stencil through this helper stay byte-for-byte unchanged on the
packages/core/legoesm/grids/operators_latlon_cgrid.py-291-    serial / band / SPMD paths and only gain the true neighbour columns under
packages/core/legoesm/grids/operators_latlon_cgrid.py-292-    a genuine longitude split.  AD-safe (the exchange uses the shared
packages/core/legoesm/grids/operators_latlon_cgrid.py:293:    sendrecv VJP; ``ppermute`` is self-transposing).  ``f`` may be 2-D
packages/core/legoesm/grids/operators_latlon_cgrid.py-294-    ``(n_lat, n_lon[_local], ...)`` or 3-D; the lon axis is axis 1 and must
packages/core/legoesm/grids/operators_latlon_cgrid.py-295-    be CELL-ALIGNED (never an ``n_lon+1`` u-face field).
packages/core/legoesm/grids/operators_latlon_cgrid.py-296-    """
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-303-    if backend == "mpi":
packages/core/legoesm/grids/operators_latlon_cgrid.py-304-        topology = get_mpi_topology()
packages/core/legoesm/grids/operators_latlon_cgrid.py-305-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/operators_latlon_cgrid.py:306:            LatLon2DLayout, exchange_halo_lon,
packages/core/legoesm/grids/operators_latlon_cgrid.py-307-        )
packages/core/legoesm/grids/operators_latlon_cgrid.py:308:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/operators_latlon_cgrid.py-309-            return exchange_halo_lon(
packages/core/legoesm/grids/operators_latlon_cgrid.py-310-                f, topology.west_rank, topology.east_rank,
packages/core/legoesm/grids/operators_latlon_cgrid.py-311-                topology.rank, halo=halo,
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-314-        mesh = get_spmd_mesh()
packages/core/legoesm/grids/operators_latlon_cgrid.py-315-        if mesh is not None and "lon" in tuple(
packages/core/legoesm/grids/operators_latlon_cgrid.py-316-                getattr(mesh, "axis_names", ())):
packages/core/legoesm/grids/operators_latlon_cgrid.py:317:            from legoesm.parallel.latlon_spmd import lon_ring_ghosts_spmd
packages/core/legoesm/grids/operators_latlon_cgrid.py:318:            return lon_ring_ghosts_spmd(f, mesh, halo=halo)
packages/core/legoesm/grids/operators_latlon_cgrid.py-319-    # Local periodic wrap (lon = axis 1); single Pad HLO.
packages/core/legoesm/grids/operators_latlon_cgrid.py-320-    pad = [(0, 0)] * f.ndim
packages/core/legoesm/grids/operators_latlon_cgrid.py-321-    pad[1] = (halo, halo)
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-336-    -------
packages/core/legoesm/grids/operators_latlon_cgrid.py-337-    f_u : (n_lat, n_lon+1, ...) at u-faces.
packages/core/legoesm/grids/operators_latlon_cgrid.py-338-    """
packages/core/legoesm/grids/operators_latlon_cgrid.py:339:    # Pad-then-average: a lon halo (local wrap, or the 2-D ring exchange) +
packages/core/legoesm/grids/operators_latlon_cgrid.py-340-    # the 2-pt face average over the padded cells.  Bit-identical to the
packages/core/legoesm/grids/operators_latlon_cgrid.py-341-    # former ``0.5*(roll(f,1)+f)`` + wrap-column concat at proc_lon==1, and
packages/core/legoesm/grids/operators_latlon_cgrid.py-342-    # spans lon partition cuts under a 2-D split.  Output face j = mean of
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-399-        return None
packages/core/legoesm/grids/operators_latlon_cgrid.py-400-    topology = get_mpi_topology()
packages/core/legoesm/grids/operators_latlon_cgrid.py-401-    from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/operators_latlon_cgrid.py:402:        LatLon2DLayout, LatLonBandLayout,
packages/core/legoesm/grids/operators_latlon_cgrid.py-403-    )
packages/core/legoesm/grids/operators_latlon_cgrid.py-404-    # Band AND 2-D pencil expose the same pole-terminated lat-LINE
packages/core/legoesm/grids/operators_latlon_cgrid.py-405-    # semantics (``south_rank``/``north_rank is None`` at the physical
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-408-    # ``interp_cell_to_vface_halo`` correct on a ``proc_lat>1`` pencil rank
packages/core/legoesm/grids/operators_latlon_cgrid.py-409-    # — without it an interior lat-cut rank would clamp its band edge to a
packages/core/legoesm/grids/operators_latlon_cgrid.py-410-    # physical pole (e.g. curl_vertex's sin clamp), corrupting metrics.
packages/core/legoesm/grids/operators_latlon_cgrid.py:411:    if isinstance(topology, (LatLonBandLayout, LatLon2DLayout)) and (
packages/core/legoesm/grids/operators_latlon_cgrid.py-412-        topology.south_rank is not None
packages/core/legoesm/grids/operators_latlon_cgrid.py-413-        or topology.north_rank is not None
packages/core/legoesm/grids/operators_latlon_cgrid.py-414-    ):
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-460-      latitude row through the backend-dispatched
packages/core/legoesm/grids/operators_latlon_cgrid.py-461-      :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat` (interior
packages/core/legoesm/grids/operators_latlon_cgrid.py-462-      cuts receive the neighbour rank's row via the AD-safe
packages/core/legoesm/grids/operators_latlon_cgrid.py:463:      ``_sendrecv_vjp`` sendrecv; a pole-touching end receives a
packages/core/legoesm/grids/operators_latlon_cgrid.py-464-      constant ghost that never reaches the output, see below),
packages/core/legoesm/grids/operators_latlon_cgrid.py-465-      computes every face with the interior average on the padded
packages/core/legoesm/grids/operators_latlon_cgrid.py-466-      array, then restores the legacy edge copy at any pole-touching
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-490-    if band is not None or spmd_pm is not None:
packages/core/legoesm/grids/operators_latlon_cgrid.py-491-        from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
packages/core/legoesm/grids/operators_latlon_cgrid.py-492-        # Interior cuts: neighbour row via the backend-aware pad (AD-safe MPI
packages/core/legoesm/grids/operators_latlon_cgrid.py:493:        # sendrecv / SPMD lat-band ppermute).  A pole-touching end gets a
packages/core/legoesm/grids/operators_latlon_cgrid.py-494-        # constant-0 ghost row here that is immediately overridden by the
packages/core/legoesm/grids/operators_latlon_cgrid.py-495-        # legacy edge copy below, so the constant never reaches the output.
packages/core/legoesm/grids/operators_latlon_cgrid.py-496-        # ``f_pad`` may be supplied pre-padded by a caller that fused this
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-529-    Cell-pad-first (PR357 Bug-2 pattern): ``u`` is padded by one
packages/core/legoesm/grids/operators_latlon_cgrid.py-530-    latitude row through the backend-dispatched
packages/core/legoesm/grids/operators_latlon_cgrid.py-531-    :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat` (local
packages/core/legoesm/grids/operators_latlon_cgrid.py:532:    ``jnp.pad`` / AD-safe MPI sendrecv at interior band cuts), so ALL
packages/core/legoesm/grids/operators_latlon_cgrid.py-533-    ``n_lat+1`` local v-faces — including partition-cut faces — average
packages/core/legoesm/grids/operators_latlon_cgrid.py-534-    the true u rows.  The historical pattern (interior faces then
packages/core/legoesm/grids/operators_latlon_cgrid.py-535-    ``pad_ns_vector_u`` refill) delivered the neighbour's ADJACENT face
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-628-    # Face j sits between cell (j-1) mod n_lon (west) and cell j (east),
packages/core/legoesm/grids/operators_latlon_cgrid.py-629-    # matching the divergence convention (cell j: west=face j, east=face j+1).
packages/core/legoesm/grids/operators_latlon_cgrid.py-630-    # Gradient at face j: (f[j] - f[(j-1) mod n_lon]) / dx
packages/core/legoesm/grids/operators_latlon_cgrid.py:631:    # Pad-then-diff: a lon halo (local wrap, or the 2-D ring exchange) then
packages/core/legoesm/grids/operators_latlon_cgrid.py-632-    # the compact zonal difference over the padded cells.  df_full[j] =
packages/core/legoesm/grids/operators_latlon_cgrid.py-633-    # f[j] - f[j-1] at u-face j; n_lon+1 faces (the last the periodic
packages/core/legoesm/grids/operators_latlon_cgrid.py-634-    # closure).  ndim-agnostic (lon = axis 1).  Bit-identical to the former
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-679-    # compact-stencil-then-pad that zeroed the south v-face and read only
packages/core/legoesm/grids/operators_latlon_cgrid.py-680-    # local ``f``, so under MPI it dropped the neighbour band's row at a
packages/core/legoesm/grids/operators_latlon_cgrid.py-681-    # partition cut (#356 follow-up, PR358).  Pre-padding ``f`` via the
packages/core/legoesm/grids/operators_latlon_cgrid.py:682:    # backend-dispatched halo (local pole-fold OR MPI sendrecv + boundary
packages/core/legoesm/grids/operators_latlon_cgrid.py-683-    # pole-fold) makes the compact stencil span partition cuts on every
packages/core/legoesm/grids/operators_latlon_cgrid.py-684-    # grid; ``zero_polar_lat_ends`` restores the wall BC at the physical
packages/core/legoesm/grids/operators_latlon_cgrid.py-685-    # poles only, and the north fold-face gradient is replaced with the
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-928-        # pad the CELL-CENTRE latitudes by one row through the
packages/core/legoesm/grids/operators_latlon_cgrid.py-929-        # backend-dispatched pad (at an interior MPI band cut the ghost
packages/core/legoesm/grids/operators_latlon_cgrid.py-930-        # row is the neighbour rank's true edge cell latitude via the
packages/core/legoesm/grids/operators_latlon_cgrid.py:931:        # AD-safe sendrecv), then take midpoints.  The previous code
packages/core/legoesm/grids/operators_latlon_cgrid.py-932-        # padded the INTERIOR-FACE cos array instead, so a cut face
packages/core/legoesm/grids/operators_latlon_cgrid.py-933-        # received the neighbour's ADJACENT face metric — one face row
packages/core/legoesm/grids/operators_latlon_cgrid.py-934-        # off — breaking both np>=2 parity and cross-cut flux
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-1081-        lat = grid.lat
packages/core/legoesm/grids/operators_latlon_cgrid.py-1082-        cos_lat = grid.cos_lat
packages/core/legoesm/grids/operators_latlon_cgrid.py-1083-        # Wall-BC pad of sin_lat: sin(south_pole)=-1, sin(north_pole)=+1.
packages/core/legoesm/grids/operators_latlon_cgrid.py:1084:        # Backend-aware so MPI interior ranks sendrecv from neighbour
packages/core/legoesm/grids/operators_latlon_cgrid.py-1085-        # rather than apply pole BC at the wrong location.
packages/core/legoesm/grids/operators_latlon_cgrid.py-1086-        # Pad the CONCRETE ``lat``, THEN take sin: ``jnp.sin(lat)``
packages/core/legoesm/grids/operators_latlon_cgrid.py-1087-        # inside jit stages to a Tracer, which kept this pad on the
packages/core/legoesm/grids/operators_latlon_cgrid.py:1088:        # traced per-step sendrecv path (census 8459326); the
packages/core/legoesm/grids/operators_latlon_cgrid.py-1089-        # concrete-lat pad constant-folds at trace time instead, and
packages/core/legoesm/grids/operators_latlon_cgrid.py-1090-        # ``jnp.sin`` of that constant folds at XLA compile time.
packages/core/legoesm/grids/operators_latlon_cgrid.py-1091-        # Interior/cut rows are bit-identical (same jnp.sin applied to
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-1165-        dy_edge_interior = 0.5 * (dy_h[1:] + dy_h[:-1])    # (n_lat-1,)
packages/core/legoesm/grids/operators_latlon_cgrid.py-1166-        dy_edge = jnp.pad(dy_edge_interior, (1, 1), mode='edge')  # (n_lat+1,)
packages/core/legoesm/grids/operators_latlon_cgrid.py-1167-        bcast_lat = (slice(None),) + (jnp.newaxis,) * (v.ndim - 1)
packages/core/legoesm/grids/operators_latlon_cgrid.py:1168:        # Pad-then-diff: a lon halo (local wrap / 2-D ring exchange) then the
packages/core/legoesm/grids/operators_latlon_cgrid.py-1169-        # compact vertex difference (v[j]-v[j-1]) over padded v => n_lon+1
packages/core/legoesm/grids/operators_latlon_cgrid.py-1170-        # vertex columns directly.  Bit-identical to ``roll(v,1)`` + wrap-
packages/core/legoesm/grids/operators_latlon_cgrid.py-1171-        # column concat at proc_lon==1; spans lon cuts under a 2-D split.
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-1175-    # u contribution: u[i-1, j]*dx[i-1] - u[i, j]*dx[i].
packages/core/legoesm/grids/operators_latlon_cgrid.py-1176-    # Pad with zeros at poles along the lat axis (axis 0).  Wall BC
packages/core/legoesm/grids/operators_latlon_cgrid.py-1177-    # at the pole; under MPI on an interior rank the same call
packages/core/legoesm/grids/operators_latlon_cgrid.py:1178:    # sendrecv's the neighbour's u row instead (the pole pad fires
packages/core/legoesm/grids/operators_latlon_cgrid.py-1179-    # only at boundary ranks).
packages/core/legoesm/grids/operators_latlon_cgrid.py-1180-    # ``u_ext`` may be supplied pre-padded by the caller (the atm
packages/core/legoesm/grids/operators_latlon_cgrid.py-1181-    # tendency fuses this pad with its other entry-level lat pads and
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-1189-        )
packages/core/legoesm/grids/operators_latlon_cgrid.py-1190-    if _tripolar_curl:
packages/core/legoesm/grids/operators_latlon_cgrid.py-1191-        # dx_cell is 2D (n_lat, n_lon); pad lat axis, append wrap column.
packages/core/legoesm/grids/operators_latlon_cgrid.py:1192:        # Use the backend-aware pad so an interior MPI rank sendrecv's the
packages/core/legoesm/grids/operators_latlon_cgrid.py-1193-        # neighbour band's dx_T row at a partition cut, matching the u halo
packages/core/legoesm/grids/operators_latlon_cgrid.py-1194-        # above.  A plain jnp.pad would force the metric to zero at the cut
packages/core/legoesm/grids/operators_latlon_cgrid.py-1195-        # and corrupt the first/last local vertex circulation (#357 review).
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-1209-    else:
packages/core/legoesm/grids/operators_latlon_cgrid.py-1210-        # Wall-BC pad of dx_cell at poles (zero contribution beyond
packages/core/legoesm/grids/operators_latlon_cgrid.py-1211-        # the pole); under MPI interior ranks pad with neighbour's
packages/core/legoesm/grids/operators_latlon_cgrid.py:1212:        # dx via sendrecv instead.
packages/core/legoesm/grids/operators_latlon_cgrid.py-1213-        # Pad the CONCRETE ``cos_lat`` first (static-folds at trace
packages/core/legoesm/grids/operators_latlon_cgrid.py-1214-        # time — same recipe as the lat/sin pad above), then form
packages/core/legoesm/grids/operators_latlon_cgrid.py-1215-        # ``R * cos * dlon`` on the padded array: row-wise identical
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-1346-        # Regular / Mercator: 1D metric, cell-pad-first (same recipe as
packages/core/legoesm/grids/operators_latlon_cgrid.py-1347-        # divergence_cgrid): at an interior MPI band cut the ghost row
packages/core/legoesm/grids/operators_latlon_cgrid.py-1348-        # is the neighbour's true edge cell latitude via the AD-safe
packages/core/legoesm/grids/operators_latlon_cgrid.py:1349:        # sendrecv, so the end faces divide by the exact serial metric.
packages/core/legoesm/grids/operators_latlon_cgrid.py-1350-        R = grid.radius
packages/core/legoesm/grids/operators_latlon_cgrid.py-1351-        dlon = grid.dlon
packages/core/legoesm/grids/operators_latlon_cgrid.py-1352-        cos_lat_v = _vface_cos_lat_core(grid)
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-1511-    # the four surrounding cells.  Builds the full n_lon+1 vertex columns
packages/core/legoesm/grids/operators_latlon_cgrid.py-1512-    # directly (no wrap-column append) — bit-identical to ``roll(m_pad,1)`` +
packages/core/legoesm/grids/operators_latlon_cgrid.py-1513-    # wrap concat at proc_lon==1; spans lon partition cuts under a 2-D split.
packages/core/legoesm/grids/operators_latlon_cgrid.py:1514:    m_pad_lon = pad_lon_cgrid(m_pad, halo=1)  # (n_lat+2, n_lon_local+2)
packages/core/legoesm/grids/operators_latlon_cgrid.py-1515-    east = m_pad_lon[:, 1:]                    # cell j   at each vertex column
packages/core/legoesm/grids/operators_latlon_cgrid.py-1516-    west = m_pad_lon[:, :-1]                   # cell j-1 at each vertex column
packages/core/legoesm/grids/operators_latlon_cgrid.py-1517-    full = east[:-1] * east[1:] * west[:-1] * west[1:]  # (n_lat+1, n_lon+1)
--
packages/core/legoesm/core/fv_tp_2d.py-1266-
packages/core/legoesm/core/fv_tp_2d.py-1267-    The per-level ``jax.vmap(transport_step)`` used by the flux-form moisture
packages/core/legoesm/core/fv_tp_2d.py-1268-    substep makes ``fv_tp_2d``'s internal ``pad_halo`` a *vmapped* cross-face
packages/core/legoesm/core/fv_tp_2d.py:1269:    ``sendrecv`` — a ``batch_axes`` failure under MPI face-scatter (CLAUDE.md:
packages/core/legoesm/core/fv_tp_2d.py-1270-    never ``vmap(pad_halo)``).  This routes the TWO field halos through
packages/core/legoesm/core/fv_tp_2d.py-1271-    ``pad_halo_4d`` (ONE message for all levels each) and ``vmap``s only the
packages/core/legoesm/core/fv_tp_2d.py-1272-    PURE-LOCAL PPM sweeps (``_fv_tp_2d_sweep1``/``_fv_tp_2d_sweep2`` — the SAME
--
packages/core/legoesm/core/fv3_sw_core.py-1711-    ``d2a2c_vect`` over levels (through its ``global_fields=`` fast path, so the
packages/core/legoesm/core/fv3_sw_core.py-1712-    A→C numerics are shared bit-for-bit — no duplication).  This replaces the
packages/core/legoesm/core/fv3_sw_core.py-1713-    per-level ``vmap(d2a2c_vect)`` whose internal ``pad_halo_vector`` was a
packages/core/legoesm/core/fv3_sw_core.py:1714:    vmapped cross-face ``sendrecv`` (``batch_axes`` failure under MPI
packages/core/legoesm/core/fv3_sw_core.py-1715-    face-scatter).  BIT-IDENTICAL to ``jax.vmap(d2a2c_vect)`` on single-rank.
packages/core/legoesm/core/fv3_sw_core.py-1716-
packages/core/legoesm/core/fv3_sw_core.py-1717-    NON-duogrid only (the duogrid ``_d2a2c_vect_duogrid`` path is not 4D-ified;
--
packages/core/legoesm/core/operators_fv_latlon.py-177-    # --- Longitude flux ---
packages/core/legoesm/core/operators_fv_latlon.py-178-    q_L_lon, q_R_lon = _ppm_reconstruct_lon(q_pad, limiter)  # (n_lat, n_lon+1)
packages/core/legoesm/core/operators_fv_latlon.py-179-
packages/core/legoesm/core/operators_fv_latlon.py:180:    # u at longitude interfaces (average neighboring cells)
packages/core/legoesm/core/operators_fv_latlon.py-181-    u_strip = u_pad[2:-2, :]  # (n_lat, n_lon+4)
packages/core/legoesm/core/operators_fv_latlon.py-182-    u_iface = 0.5 * (u_strip[:, 1:-2] + u_strip[:, 2:-1])  # (n_lat, n_lon+1)
packages/core/legoesm/core/operators_fv_latlon.py-183-
--
packages/core/legoesm/core/operators_fv_latlon.py-319-    q_3d : jax.Array, shape (n_lat, n_lon, nlev)
packages/core/legoesm/core/operators_fv_latlon.py-320-    grid : LatLonGrid
packages/core/legoesm/core/operators_fv_latlon.py-321-    padded : jax.Array, optional
packages/core/legoesm/core/operators_fv_latlon.py:322:        Pre-padded field with halo=2, shape ``(n_lat+4, n_lon+4, nlev)``.
packages/core/legoesm/core/operators_fv_latlon.py-323-        When supplied, the internal ``pad_halo_latlon_3d`` call is
packages/core/legoesm/core/operators_fv_latlon.py-324-        skipped — useful for paired (∂/∂lon, ∂/∂lat) calls on the
packages/core/legoesm/core/operators_fv_latlon.py-325-        same input where the halo pad can be shared.
--
packages/core/legoesm/core/operators_fv_latlon.py-330-    """
packages/core/legoesm/core/operators_fv_latlon.py-331-    if padded is None:
packages/core/legoesm/core/operators_fv_latlon.py-332-        # Pad once for all levels.
packages/core/legoesm/core/operators_fv_latlon.py:333:        padded = pad_halo_latlon_3d(q_3d, halo=2)        # (n_lat+4, n_lon+4, nlev)
packages/core/legoesm/core/operators_fv_latlon.py-334-    # Strip latitude halo; longitude axis is now axis -2 (the axis
packages/core/legoesm/core/operators_fv_latlon.py-335-    # ``ppm_edge_values`` operates on).
packages/core/legoesm/core/operators_fv_latlon.py-336-    q_strip = padded[2:-2, :, :]                         # (n_lat, n_lon+4, nlev)
--
packages/core/legoesm/core/operators_fv_latlon.py-360-    jax.Array, shape (n_lat, n_lon, nlev)
packages/core/legoesm/core/operators_fv_latlon.py-361-    """
packages/core/legoesm/core/operators_fv_latlon.py-362-    if padded is None:
packages/core/legoesm/core/operators_fv_latlon.py:363:        padded = pad_halo_latlon_3d(q_3d, halo=2)        # (n_lat+4, n_lon+4, nlev)
packages/core/legoesm/core/operators_fv_latlon.py-364-    # Strip longitude halo.  ``ppm_edge_values`` operates on axis -2,
packages/core/legoesm/core/operators_fv_latlon.py-365-    # so swap the lat/lon axes so latitude lives there.
packages/core/legoesm/core/operators_fv_latlon.py-366-    q_strip = padded[:, 2:-2, :]                         # (n_lat+4, n_lon, nlev)
--
packages/core/legoesm/core/_fv3_divergence_corner.py-155-    FV3_3D iter-1044: ndim-polymorphic.  Accepts both 3D inputs
packages/core/legoesm/core/_fv3_divergence_corner.py-156-    ``(6, n+1, n+1)`` and 4D inputs ``(6, n+1, n+1, nlev)``.  Under
packages/core/legoesm/core/_fv3_divergence_corner.py-157-    MPI the 4D path issues exactly ONE ``pad_halo_4d`` call for
packages/core/legoesm/core/_fv3_divergence_corner.py:158:    ``ua`` and one for ``va`` (one ``mpi4jax.sendrecv`` per call,
packages/core/legoesm/core/_fv3_divergence_corner.py-159-    batched over levels via the trailing axis), instead of ``nlev``
packages/core/legoesm/core/_fv3_divergence_corner.py:160:    separate sendrecvs.  Static metric pads still happen once
packages/core/legoesm/core/_fv3_divergence_corner.py-161-    regardless of ``nlev``.  This closes the codex iter-1044 review
packages/core/legoesm/core/_fv3_divergence_corner.py:162:    blocker on the per-level Python loop's O(nlev) sendrecv count.
packages/core/legoesm/core/_fv3_divergence_corner.py-163-
packages/core/legoesm/core/_fv3_divergence_corner.py-164-    Direct port of the non-bounded-domain, ``grid_type < 3`` branch
packages/core/legoesm/core/_fv3_divergence_corner.py-165-    (lines 2181-2225 in the Fortran source).  Computes B-grid corner
--
packages/core/legoesm/core/_fv3_divergence_corner.py-208-    # Step 2: pad ua, va with halo=1 so the cosa cross-correction at
packages/core/legoesm/core/_fv3_divergence_corner.py-209-    # j-1 / i-1 reads neighbour-panel cells (cross-face values).
packages/core/legoesm/core/_fv3_divergence_corner.py-210-    # FV3_3D iter-1044: dispatch pad_halo vs pad_halo_4d by ndim so
packages/core/legoesm/core/_fv3_divergence_corner.py:211:    # the 4D path issues one batched sendrecv per array (not nlev).
packages/core/legoesm/core/_fv3_divergence_corner.py-212-    if _is_4d:
packages/core/legoesm/core/_fv3_divergence_corner.py-213-        from legoesm.grids.halo import pad_halo_4d
packages/core/legoesm/core/_fv3_divergence_corner.py-214-        ua_pad = pad_halo_4d(ua)
--
packages/core/legoesm/core/_fv3_divergence_corner.py-572-    FV3_3D iter-1044: ``fv3_divergence_corner_2d`` is shape-polymorphic
packages/core/legoesm/core/_fv3_divergence_corner.py-573-    (3D and 4D), so the 3D wrapper is now a single direct call rather
packages/core/legoesm/core/_fv3_divergence_corner.py-574-    than a per-level ``jax.vmap`` or Python loop.  Under MPI the dynamic
packages/core/legoesm/core/_fv3_divergence_corner.py:575:    ``ua``/``va`` go through one ``pad_halo_4d`` (one batched sendrecv
packages/core/legoesm/core/_fv3_divergence_corner.py-576-    per array, not ``nlev`` of them); static metric pads happen once.
packages/core/legoesm/core/_fv3_divergence_corner.py-577-
packages/core/legoesm/core/_fv3_divergence_corner.py-578-    Parameters
--
packages/core/legoesm/core/_fv3_divergence_corner.py-704-    # FV3_3D iter-1044: dispatch by ndim so NH callers can pass the
packages/core/legoesm/core/_fv3_divergence_corner.py-705-    # full 4D ``(6, n+1, n+1, nlev)`` corner-staggered ``delpc`` array
packages/core/legoesm/core/_fv3_divergence_corner.py-706-    # directly and avoid wrapping this function in ``jax.vmap`` (which
packages/core/legoesm/core/_fv3_divergence_corner.py:707:    # under MPI puts ``mpi4jax.sendrecv`` inside a vmap and trips
packages/core/legoesm/core/_fv3_divergence_corner.py-708-    # mpi4jax's batch-axis assertion).  Padded index 0 = west halo
packages/core/legoesm/core/_fv3_divergence_corner.py-709-    # (i = -1 shifted), padded index n+2 = east halo (i = n+1).
packages/core/legoesm/core/_fv3_divergence_corner.py-710-    # Cube-vertex halo follows ``set_corner_fill_mode`` (iter-7 "avg").

exec
/usr/bin/bash -lc "git status --short && nl -ba packages/core/legoesm/parallel/latlon_mpi.py | sed -n '1,420p' && nl -ba packages/core/legoesm/parallel/latlon_mpi.py | sed -n '1450,1870p' && nl -ba packages/core/legoesm/grids/halo_latlon.py | sed -n '680,850p' && if [ -f tests/distributed/test_latlon_2d_fused_pad.py ]; then nl -ba tests/distributed/test_latlon_2d_fused_pad.py | sed -n '1,460p'; fi" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 443ms:
 M packages/core/legoesm/grids/halo_latlon.py
 M packages/core/legoesm/parallel/latlon_mpi.py
?? .physics-validator/scaling_campaign/codex_2dfused_r28_2026-08-04.md
?? .physics-validator/scaling_campaign/codex_consult_improvements_2026-08-04.md
?? scripts/cluster/scaling_levante/mpas_s10_192b.sbatch
?? tests/distributed/test_latlon_2d_fused_pad.py
     1	"""MPI domain decomposition for the lat-lon C-grid atmospheric dycore.
     2	
     3	Latitude-band decomposition: each MPI rank owns a contiguous band of
     4	latitude rows; every rank owns all longitudes (no decomposition in the
     5	periodic direction).
     6	
     7	This file provides the halo-exchange and padded-grid machinery
     8	together with the per-step driver ``make_latlon_mpi_step``, which is
     9	fully implemented: it delegates to the existing serial C-grid step in
    10	:mod:`legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid` via the
    11	backend-aware MPI halo path.
    12	
    13	Conventions
    14	-----------
    15	- Halo width ``halo``: number of ghost lat rows added on each side
    16	  (south + north).  Operators with compact 1-cell stencils (gradient,
    17	  divergence, curl) need ``halo=1``; operators with PPM
    18	  reconstruction or biharmonic stencils need ``halo=2``.
    19	- Pole fold matches :func:`legoesm.grids.halo_latlon.pad_halo_latlon`
    20	  exactly: mirror-reverse the first/last ``halo`` interior rows, shift
    21	  by 180° in longitude, and (optionally) negate for vector ``v``
    22	  components.  This single source of truth lives in
    23	  :mod:`legoesm.grids.halo_latlon`; we reuse it here so MPI and
    24	  single-rank paths can never disagree at pole-touching ranks.
    25	- C-grid layout:
    26	
    27	      u : zonal velocity at LON faces.  Global ``n_lon+1`` faces (the
    28	                              trailing one the periodic closure
    29	                              ``face n_lon == face 0``).  The 1-D band keeps
    30	                              the full ``n_lon+1`` (lon is not split); the 2-D
    31	                              pencil keeps ``n_lon_local+1`` faces, so W/E
    32	                              neighbours SHARE the boundary east face — the
    33	                              lon twin of v's shared row (see
    34	                              ``scatter_state_latlon_2d`` / the ``is_u_face``
    35	                              gather mode).
    36	      v : (n_lat_local+1,  n_lon,   nlev)  — meridional velocity at lat
    37	                              interfaces; N/S neighbouring ranks share the
    38	                              ``v`` row at the partition boundary
    39	      T, p_s, phis, tracers : (n_lat_local, n_lon, ...) cell-centered
    40	
    41	  For ``v`` we keep the duplicated boundary row across neighbours; the
    42	  scatter and halo-exchange helpers handle this explicitly.
    43	
    44	- AD safety: halo sendrecv uses
    45	  :func:`legoesm.parallel.halo_exchange.get_sendrecv_vjp` (the AD-safe
    46	  wrapper around ``mpi4jax.sendrecv``).  Backward swaps source/dest as
    47	  required by the reverse-mode rule.
    48	
    49	Status
    50	------
    51	Halo-exchange and padded-grid machinery (DONE):
    52	    LatLonBandLayout, make_latlon_band_layout, exchange_halo_latlon,
    53	    scatter_state_latlon, gather_state_latlon, pad_state_halos,
    54	    strip_halos, build_padded_grid.
    55	
    56	Per-step driver (DONE):
    57	    make_latlon_mpi_step — wraps
    58	    ``cgrid_latlon_hydrostatic_tendencies`` and the RK driver inside a
    59	    pad → step → strip cycle.  The pole-wall BC (``v=0`` at the global
    60	    lat boundaries) is rank-aware so interior cuts are NOT zeroed.
    61	"""
    62	
    63	from __future__ import annotations
    64	
    65	import functools
    66	
    67	from typing import TYPE_CHECKING, Callable, NamedTuple
    68	
    69	import jax
    70	import jax.numpy as jnp
    71	import numpy as np
    72	
    73	if TYPE_CHECKING:
    74	    from legoesm.grids.latlon import FoldDescriptor
    75	    # NOTE: ocean.state.LatLonCGridOceanState is referenced only in docstrings
    76	    # (:class: cross-refs), so it is intentionally NOT imported here — importing
    77	    # it would make the low-level parallel layer depend UP on the ocean
    78	    # component, which blocks component independence (see import-linter contracts).
    79	
    80	from legoesm.grids.halo_latlon import (
    81	    fold_pole_rows,
    82	    fold_pole_rows_3d,
    83	)
    84	from legoesm.parallel.halo_exchange import get_sendrecv_vjp
    85	
    86	
    87	# ============================================================================
    88	# Layout
    89	# ============================================================================
    90	
    91	
    92	class LatLonBandLayout(NamedTuple):
    93	    """Latitude-band decomposition layout for MPI.
    94	
    95	    Attributes
    96	    ----------
    97	    rank, n_ranks : int
    98	        This process's MPI rank and the world size.
    99	    n_lat_global, n_lon_global : int
   100	        Total latitude rows / longitude columns (global problem size).
   101	    n_lat_local : int
   102	        Interior latitude rows owned by this rank (no halos).
   103	    lat_start, lat_end : int
   104	        Global indices of the owned band: rows ``[lat_start, lat_end)``.
   105	    south_rank, north_rank : int or None
   106	        MPI ranks of the southern / northern neighbour, or ``None`` if
   107	        this rank touches the south / north pole.
   108	    fold : FoldDescriptor or None
   109	        Tripolar north-fold descriptor (issue #353).  When non-``None``
   110	        and ``fold.is_active`` is True, the **northernmost** rank
   111	        (``north_rank is None``) applies the permutation-based tripolar
   112	        fold (``perm_T``/``perm_v`` + ``vector_sign_u``/``vector_sign_v``)
   113	        at the north boundary instead of the geographic 180°-roll
   114	        pole-fold.  Interior partition cuts and the south boundary are
   115	        unaffected.  ``None`` (the default) ⇒ regular lat-lon /
   116	        atmospheric pole-fold everywhere — fully backward compatible.
   117	    """
   118	
   119	    rank: int
   120	    n_ranks: int
   121	    n_lat_global: int
   122	    n_lon_global: int
   123	    n_lat_local: int
   124	    lat_start: int
   125	    lat_end: int
   126	    south_rank: int | None
   127	    north_rank: int | None
   128	    # Trailing field with a default so all existing constructor call
   129	    # sites (which never passed ``fold``) keep working unchanged.
   130	    fold: "FoldDescriptor | None" = None
   131	
   132	
   133	def wet_band_boundaries(
   134	    wet_rows,
   135	    n_ranks: int,
   136	    *,
   137	    min_rows: int = 1,
   138	) -> tuple[int, ...]:
   139	    """Contiguous latitude-band boundaries equalizing WET (ocean) cells.
   140	
   141	    The default row-count split (``make_latlon_band_layout`` with no
   142	    ``boundaries``) gives land-heavy bands the same row budget as open-ocean
   143	    bands, so ranks whose band is mostly land idle at every collective (the
   144	    lat-lon analogue of the Voronoi path's METIS weighting). This computes
   145	    band boundaries from the per-ROW wet-cell counts instead: each band gets
   146	    as close to ``total_wet / n_ranks`` wet cells as a contiguous row split
   147	    allows.
   148	
   149	    Pure host-side numpy — deterministic, no MPI; every rank computes the
   150	    identical result from the identical global mask. Feed the result to
   151	    ``make_latlon_band_layout(..., boundaries=...)`` (or
   152	    ``initialize_distributed_latlon(band_boundaries=...)``).
   153	
   154	    Parameters
   155	    ----------
   156	    wet_rows : (n_lat,) array-like
   157	        Wet-cell count per latitude row, e.g. ``land_mask.sum(axis=1)``
   158	        (any non-negative row weight works).
   159	    n_ranks : int
   160	        Number of contiguous bands.
   161	    min_rows : int
   162	        Minimum rows per band (halo floor — the MPI pads raise when
   163	        ``halo > n_lat_local``; pass 2 for the halo=2 PPM/biharmonic paths).
   164	
   165	    Returns
   166	    -------
   167	    boundaries : tuple of ``n_ranks + 1`` ints, ``boundaries[0] == 0``,
   168	        ``boundaries[-1] == n_lat``, strictly increasing; band ``r`` owns rows
   169	        ``[boundaries[r], boundaries[r+1])``.
   170	
   171	    Algorithm: walk the cumulative wet count and cut band ``r`` at the first
   172	    row where the running total reaches ``r / n_ranks`` of the global wet
   173	    total, clamped so every band (including all remaining ones) keeps
   174	    ``min_rows`` rows. All-land stretches (zero weight) attach to whichever
   175	    band the running quantile puts them in — they cost compute-idle rows but
   176	    carry no wet work, which is exactly the imbalance being minimized.
   177	    """
   178	    w = np.asarray(wet_rows, dtype=np.float64)
   179	    if w.ndim != 1:
   180	        raise ValueError(f"wet_rows must be 1-D (n_lat,), got shape {w.shape}")
   181	    n_lat = int(w.shape[0])
   182	    if n_ranks < 1:
   183	        raise ValueError(f"n_ranks must be >=1, got {n_ranks}")
   184	    if min_rows < 1:
   185	        raise ValueError(f"min_rows must be >=1, got {min_rows}")
   186	    if np.any(w < 0) or not np.all(np.isfinite(w)):
   187	        raise ValueError("wet_rows must be finite and non-negative")
   188	    if n_lat < n_ranks * min_rows:
   189	        raise ValueError(
   190	            f"Cannot decompose {n_lat} lat rows across {n_ranks} ranks with "
   191	            f"min_rows={min_rows} (need at least {n_ranks * min_rows} rows)."
   192	        )
   193	    total = float(w.sum())
   194	    if total <= 0.0:
   195	        # Degenerate all-land domain: fall back to the even row split (there
   196	        # is no wet work to balance).
   197	        base, rem = divmod(n_lat, n_ranks)
   198	        sizes = [base + 1 if r < rem else base for r in range(n_ranks)]
   199	        return tuple(int(x) for x in np.concatenate([[0], np.cumsum(sizes)]))
   200	
   201	    cum = np.cumsum(w)
   202	    boundaries = [0]
   203	    for r in range(1, n_ranks):
   204	        target = total * r / n_ranks
   205	        # First row whose cumulative weight reaches the target; the cut is
   206	        # AFTER that row (searchsorted on the running total).
   207	        cut = int(np.searchsorted(cum, target, side="left")) + 1
   208	        lo = boundaries[-1] + min_rows                 # this band keeps min_rows
   209	        hi = n_lat - (n_ranks - r) * min_rows          # remaining bands too
   210	        boundaries.append(int(np.clip(cut, lo, hi)))
   211	    boundaries.append(n_lat)
   212	    return tuple(boundaries)
   213	
   214	
   215	def validate_band_boundaries(boundaries, n_ranks: int, n_lat: int,
   216	                             ) -> tuple[int, ...]:
   217	    """Validate explicit band boundaries; return them as a tuple of ints.
   218	
   219	    Shared by :func:`make_latlon_band_layout` and the
   220	    ``initialize_distributed_latlon`` re-init span comparison (which must
   221	    reject invalid boundaries BEFORE deciding a stale layout may be reused —
   222	    an invalid request must never be silently served; codex round 2).
   223	    Checks: integral values (no silent float truncation), ``n_ranks + 1``
   224	    entries, ``[0, n_lat]`` span, strictly increasing.
   225	    """
   226	    if any(int(x) != x for x in boundaries):
   227	        raise ValueError(
   228	            f"boundaries must be integers (a float like 1.9 would be "
   229	            f"silently truncated), got {tuple(boundaries)}")
   230	    b = tuple(int(x) for x in boundaries)
   231	    if len(b) != n_ranks + 1:
   232	        raise ValueError(
   233	            f"boundaries must have n_ranks+1 = {n_ranks + 1} entries, "
   234	            f"got {len(b)}")
   235	    if b[0] != 0 or b[-1] != n_lat:
   236	        raise ValueError(
   237	            f"boundaries must span [0, n_lat={n_lat}], got "
   238	            f"[{b[0]}, {b[-1]}]")
   239	    if any(b[i + 1] <= b[i] for i in range(n_ranks)):
   240	        raise ValueError(
   241	            f"boundaries must be strictly increasing (every band >= 1 "
   242	            f"row), got {b}")
   243	    return b
   244	
   245	
   246	def make_latlon_band_layout(
   247	    rank: int,
   248	    n_ranks: int,
   249	    n_lat: int,
   250	    n_lon: int,
   251	    fold: "FoldDescriptor | None" = None,
   252	    *,
   253	    boundaries: tuple[int, ...] | None = None,
   254	) -> LatLonBandLayout:
   255	    """Build a latitude-band decomposition layout.
   256	
   257	    Divides ``n_lat`` rows as evenly as possible across ``n_ranks``;
   258	    the first ``n_lat % n_ranks`` ranks get one extra row.
   259	
   260	    Parameters
   261	    ----------
   262	    fold : FoldDescriptor or None
   263	        Tripolar north-fold descriptor (issue #353).  Carried on the
   264	        layout so the MPI halo path can apply the permutation-based
   265	        fold at the northernmost rank.  ``None`` ⇒ regular lat-lon.
   266	    boundaries : tuple of ints or None
   267	        Optional explicit band boundaries (``n_ranks + 1`` strictly
   268	        increasing ints spanning ``[0, n_lat]``): band ``r`` owns rows
   269	        ``[boundaries[r], boundaries[r+1])``.  Use
   270	        :func:`wet_band_boundaries` to balance WET cells instead of row
   271	        counts (land-heavy bands otherwise idle).  ``None`` (default)
   272	        keeps the even row split byte-identically.  Every rank must pass
   273	        the IDENTICAL boundaries (a deterministic host computation from
   274	        global data).  Non-uniform bands are an MPI-band-path feature:
   275	        the SPMD steps require uniform ``n_lat % n_devices == 0`` bands,
   276	        and the banded-multigrid preconditioner needs even-aligned
   277	        boundaries to coarsen (it degrades gracefully to fewer levels
   278	        otherwise).
   279	    """
   280	    if n_ranks < 1:
   281	        raise ValueError(f"n_ranks must be >=1, got {n_ranks}")
   282	    if rank < 0 or rank >= n_ranks:
   283	        raise ValueError(f"rank {rank} out of range [0, {n_ranks})")
   284	    if n_lat < n_ranks:
   285	        raise ValueError(
   286	            f"Cannot decompose {n_lat} lat rows across {n_ranks} ranks "
   287	            "(at least one row per rank is required for a 1-cell halo "
   288	            "neighbour exchange)."
   289	        )
   290	
   291	    if boundaries is not None:
   292	        b = validate_band_boundaries(boundaries, n_ranks, n_lat)
   293	        lat_start = b[rank]
   294	        lat_end = b[rank + 1]
   295	        n_local = lat_end - lat_start
   296	    else:
   297	        base = n_lat // n_ranks
   298	        remainder = n_lat % n_ranks
   299	        if rank < remainder:
   300	            n_local = base + 1
   301	            lat_start = rank * (base + 1)
   302	        else:
   303	            n_local = base
   304	            lat_start = remainder * (base + 1) + (rank - remainder) * base
   305	        lat_end = lat_start + n_local
   306	
   307	    return LatLonBandLayout(
   308	        rank=rank,
   309	        n_ranks=n_ranks,
   310	        n_lat_global=n_lat,
   311	        n_lon_global=n_lon,
   312	        n_lat_local=n_local,
   313	        lat_start=lat_start,
   314	        lat_end=lat_end,
   315	        south_rank=rank - 1 if rank > 0 else None,
   316	        north_rank=rank + 1 if rank < n_ranks - 1 else None,
   317	        fold=fold,
   318	    )
   319	
   320	
   321	class LatLon2DLayout(NamedTuple):
   322	    """2-D pencil (lat × lon) decomposition layout for MPI.
   323	
   324	    Increment 2 of the lat-lon 2-D decomposition
   325	    (``docs/performance/scaling/latlon_2d_decomposition_design.md``).  Generalises
   326	    :class:`LatLonBandLayout` from a 1-D latitude band to a 2-D
   327	    ``(proc_lat, proc_lon)`` process grid (row-major rank =
   328	    ``proc_row * proc_lon + proc_col``).  Latitude is a LINE (poles
   329	    terminate it ⇒ ``south_rank``/``north_rank`` are ``None`` at the
   330	    grid's top/bottom rows); longitude is a periodic RING (``west_rank``/
   331	    ``east_rank`` are ALWAYS defined, wrapping ``proc_col`` modulo
   332	    ``proc_lon``).  ``proc_lon == 1`` reproduces the 1-D band exactly
   333	    (west==east==rank ⇒ the :func:`exchange_halo_lon` local-wrap path).
   334	
   335	    The N/S neighbours feed :func:`exchange_halo_latlon`; the W/E
   336	    neighbours feed :func:`exchange_halo_lon` (increment 1).
   337	    """
   338	    rank: int
   339	    n_ranks: int
   340	    proc_lat: int
   341	    proc_lon: int
   342	    proc_row: int
   343	    proc_col: int
   344	    n_lat_global: int
   345	    n_lon_global: int
   346	    n_lat_local: int
   347	    n_lon_local: int
   348	    lat_start: int
   349	    lat_end: int
   350	    lon_start: int
   351	    lon_end: int
   352	    south_rank: int | None
   353	    north_rank: int | None
   354	    west_rank: int
   355	    east_rank: int
   356	    fold: "FoldDescriptor | None" = None
   357	
   358	
   359	def _even_split(n: int, parts: int, idx: int) -> tuple[int, int]:
   360	    """Block ``idx`` of an even-as-possible split of ``n`` into ``parts``
   361	    (first ``n % parts`` blocks get one extra).  Returns ``(start, len)``."""
   362	    base, rem = n // parts, n % parts
   363	    if idx < rem:
   364	        return idx * (base + 1), base + 1
   365	    return rem * (base + 1) + (idx - rem) * base, base
   366	
   367	
   368	def make_latlon_2d_layout(
   369	    rank: int,
   370	    proc_lat: int,
   371	    proc_lon: int,
   372	    n_lat: int,
   373	    n_lon: int,
   374	    fold: "FoldDescriptor | None" = None,
   375	) -> LatLon2DLayout:
   376	    """Build a 2-D pencil decomposition layout for ``rank``.
   377	
   378	    ``rank = proc_row * proc_lon + proc_col`` (row-major).  Latitude is
   379	    split over ``proc_lat`` (line, pole-terminated), longitude over
   380	    ``proc_lon`` (periodic ring).  ``proc_lat * proc_lon`` must equal the
   381	    world size; ``proc_lon == 1`` gives the 1-D-band-equivalent layout.
   382	    """
   383	    if proc_lat < 1 or proc_lon < 1:
   384	        raise ValueError(
   385	            f"proc_lat and proc_lon must be >=1, got "
   386	            f"proc_lat={proc_lat}, proc_lon={proc_lon}")
   387	    n_ranks = proc_lat * proc_lon
   388	    if rank < 0 or rank >= n_ranks:
   389	        raise ValueError(f"rank {rank} out of range [0, {n_ranks})")
   390	    if n_lat < proc_lat:
   391	        raise ValueError(
   392	            f"cannot split {n_lat} lat rows over proc_lat={proc_lat} "
   393	            "(>=1 row/block required for a 1-cell halo)."
   394	        )
   395	    if n_lon < proc_lon:
   396	        raise ValueError(
   397	            f"cannot split {n_lon} lon cols over proc_lon={proc_lon} "
   398	            "(>=1 col/block required for a 1-cell halo)."
   399	        )
   400	
   401	    proc_row, proc_col = divmod(rank, proc_lon)
   402	    lat_start, n_lat_local = _even_split(n_lat, proc_lat, proc_row)
   403	    lon_start, n_lon_local = _even_split(n_lon, proc_lon, proc_col)
   404	
   405	    def _rank_at(r, c):
   406	        return r * proc_lon + c
   407	
   408	    # Latitude = pole-terminated line: None at the grid's top/bottom row.
   409	    south_rank = _rank_at(proc_row - 1, proc_col) if proc_row > 0 else None
   410	    north_rank = (_rank_at(proc_row + 1, proc_col)
   411	                  if proc_row < proc_lat - 1 else None)
   412	    # Longitude = periodic ring: always defined (wrap modulo proc_lon).
   413	    west_rank = _rank_at(proc_row, (proc_col - 1) % proc_lon)
   414	    east_rank = _rank_at(proc_row, (proc_col + 1) % proc_lon)
   415	
   416	    return LatLon2DLayout(
   417	        rank=rank, n_ranks=n_ranks, proc_lat=proc_lat, proc_lon=proc_lon,
   418	        proc_row=proc_row, proc_col=proc_col,
   419	        n_lat_global=n_lat, n_lon_global=n_lon,
   420	        n_lat_local=n_lat_local, n_lon_local=n_lon_local,
  1450	    if halo > interior.shape[0]:
  1451	        raise ValueError(
  1452	            f"_pad_with_pole_bc_lat_mpi_1d: halo={halo} exceeds "
  1453	            f"n_lat_local={interior.shape[0]} on rank {layout.rank}"
  1454	        )
  1455	
  1456	    # South side
  1457	    if layout.south_rank is None:
  1458	        south_slab = jnp.full(
  1459	            (halo,), jnp.asarray(south_value, dtype=interior_dtype),
  1460	        )
  1461	    else:
  1462	        try:
  1463	            import mpi4jax
  1464	            from mpi4py import MPI
  1465	        except ImportError as exc:
  1466	            raise ImportError(
  1467	                "Lat-lon MPI 1-D pad (interior partition cut) requires "
  1468	                "mpi4jax and mpi4py."
  1469	            ) from exc
  1470	        comm = MPI.COMM_WORLD
  1471	        sendrecv = get_sendrecv_vjp(mpi4jax)
  1472	        send_bot = interior[:halo]
  1473	        recv_template = jnp.zeros_like(send_bot)
  1474	        south_slab = sendrecv(
  1475	            send_bot, recv_template,
  1476	            layout.south_rank, layout.south_rank,
  1477	            layout.rank, layout.south_rank, comm,
  1478	        )
  1479	
  1480	    # North side
  1481	    if layout.north_rank is None:
  1482	        north_slab = jnp.full(
  1483	            (halo,), jnp.asarray(north_value, dtype=interior_dtype),
  1484	        )
  1485	    else:
  1486	        try:
  1487	            import mpi4jax
  1488	            from mpi4py import MPI
  1489	        except ImportError as exc:
  1490	            raise ImportError(
  1491	                "Lat-lon MPI 1-D pad (interior partition cut) requires "
  1492	                "mpi4jax and mpi4py."
  1493	            ) from exc
  1494	        comm = MPI.COMM_WORLD
  1495	        sendrecv = get_sendrecv_vjp(mpi4jax)
  1496	        send_top = interior[-halo:]
  1497	        recv_template = jnp.zeros_like(send_top)
  1498	        north_slab = sendrecv(
  1499	            send_top, recv_template,
  1500	            layout.north_rank, layout.north_rank,
  1501	            layout.rank, layout.north_rank, comm,
  1502	        )
  1503	
  1504	    return jnp.concatenate([south_slab, interior, north_slab], axis=0)
  1505	
  1506	
  1507	def _pad_static_wall_bc_mpi(
  1508	    interior, layout: LatLonBandLayout, halo: int,
  1509	    south_value: float, north_value: float,
  1510	):
  1511	    """Trace-time host-MPI pad of a CONCRETE (non-Tracer) field.
  1512	
  1513	    Grid metrics (``grid.lat``, ``sin_lat``, tripolar ``dx_T`` rows, …)
  1514	    are closed-over constants inside the jitted step, so their wall-BC
  1515	    pads are static — yet the traced ``sendrecv`` op cannot be
  1516	    constant-folded by XLA, so every operator call re-exchanged the
  1517	    same bytes every step (census job 8459289: 14 metric pads/step, two
  1518	    of them inside the barotropic PCG ``fori_loop`` body = 120 executed
  1519	    sendrecv pairs/step at M=60).  This helper performs the exchange
  1520	    ONCE at trace time with host mpi4py; the result is a compile-time
  1521	    constant and the per-step collective disappears.
  1522	
  1523	    Deadlock safety: tracing is SPMD-synchronous — every rank traces
  1524	    the same Python (same shapes/dtypes ⇒ same jit cache hits/misses;
  1525	    the persistent XLA cache caches compilation, not tracing), so all
  1526	    ranks execute the same eager ``Sendrecv`` schedule in the same
  1527	    order.  Blocking host ``Sendrecv`` pairs match neighbour-to-
  1528	    neighbour exactly like the traced path.
  1529	
  1530	    Wall-BC semantics only (constants at pole-touching boundaries —
  1531	    callers with ``north_fold=True`` keep the traced path).
  1532	
  1533	    ``mpi4py`` is imported only inside the neighbor branches (codex
  1534	    round-2 MAJOR): a single-rank armed-"mpi" backend (both neighbors
  1535	    ``None`` — ``make_latlon_mpi_step`` arms even at n_ranks=1) must
  1536	    keep working without the optional MPI stack, exactly like the
  1537	    traced 1-D path.
  1538	    """
  1539	    arr = np.asarray(interior)
  1540	    trailing = arr.shape[1:]
  1541	
  1542	    if layout.south_rank is None:
  1543	        south = np.full((halo,) + trailing, south_value, dtype=arr.dtype)
  1544	    else:
  1545	        from mpi4py import MPI
  1546	
  1547	        south = np.empty((halo,) + trailing, dtype=arr.dtype)
  1548	        MPI.COMM_WORLD.Sendrecv(
  1549	            np.ascontiguousarray(arr[:halo]), dest=layout.south_rank,
  1550	            sendtag=layout.rank,
  1551	            recvbuf=south, source=layout.south_rank,
  1552	            recvtag=layout.south_rank,
  1553	        )
  1554	    if layout.north_rank is None:
  1555	        north = np.full((halo,) + trailing, north_value, dtype=arr.dtype)
  1556	    else:
  1557	        from mpi4py import MPI
  1558	
  1559	        north = np.empty((halo,) + trailing, dtype=arr.dtype)
  1560	        MPI.COMM_WORLD.Sendrecv(
  1561	            np.ascontiguousarray(arr[-halo:]), dest=layout.north_rank,
  1562	            sendtag=layout.rank,
  1563	            recvbuf=north, source=layout.north_rank,
  1564	            recvtag=layout.north_rank,
  1565	        )
  1566	    return jnp.concatenate(
  1567	        [jnp.asarray(south), jnp.asarray(interior), jnp.asarray(north)],
  1568	        axis=0,
  1569	    )
  1570	
  1571	
  1572	def pad_with_pole_bc_lat_mpi(
  1573	    interior, layout: LatLonBandLayout, halo: int = 1,
  1574	    south_value: float = 0.0, north_value: float = 0.0,
  1575	    is_vector_v: bool = False, is_vector_u: bool = False,
  1576	    north_fold: bool = False,
  1577	):
  1578	    """MPI variant of :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat`.
  1579	
  1580	    Pole-touching south rank → ``south_value`` constant pad.
  1581	    Pole-touching north rank → ``north_value`` constant pad.
  1582	    Interior partition cuts → MPI sendrecv with neighbour.
  1583	
  1584	    Reuses :func:`exchange_halo_latlon` for the inter-rank exchange;
  1585	    overrides the pole-fold result with the requested constants on
  1586	    the boundary ranks afterwards (so the wall-BC semantics is
  1587	    preserved instead of pole-folding).  This is the SIMPLEST way
  1588	    to reuse the existing AD-safe sendrecv machinery — we let
  1589	    ``exchange_halo_latlon`` do the heavy lifting and then patch
  1590	    the boundary halo slabs.
  1591	
  1592	    Tripolar grids (issue #353): the north fold is OPT-IN via
  1593	    ``north_fold``.  The default (``north_fold=False``) keeps the
  1594	    historical WALL semantics at the north — even on a tripolar layout —
  1595	    so existing wall-BC callers (``pad_ns_zero`` and the like) are
  1596	    unchanged.  Only when ``north_fold=True`` AND ``layout.fold`` is
  1597	    active AND this rank owns the north boundary is the north slab left
  1598	    as the permutation-folded data from ``exchange_halo_latlon`` instead
  1599	    of being overwritten with ``north_value``.  ``is_vector_u`` /
  1600	    ``is_vector_v`` then select ``vector_sign_u`` / ``vector_sign_v`` at
  1601	    the fold.  The south boundary is always a wall (an ORCA grid's south
  1602	    is a closed edge).
  1603	    """
  1604	    if halo <= 0:
  1605	        return interior
  1606	
  1607	    # Static-metric constant folding (census job 8459289): a concrete
  1608	    # (non-Tracer) 1-D input is a closed-over compile-time constant —
  1609	    # its wall-BC pad is exchanged ONCE at trace time via host MPI
  1610	    # instead of a per-step traced sendrecv that XLA cannot fold.
  1611	    # Codex-hardened gate (review 2026-06-10):
  1612	    #   * ndim == 1 only — covers every measured static pad (the
  1613	    #     1-D lat metrics; census 8459289) while keeping the rarely-
  1614	    #     trodden 2-D tripolar-metric pads on the traced path;
  1615	    #   * boundary constants must ALSO be non-Tracers (a traced
  1616	    #     south/north value must not be constant-folded);
  1617	    #   * same halo <= n_lat_local guard as the traced path (a silent
  1618	    #     short send would truncate/hang instead of raising).
  1619	    # INVARIANT (same class as every traced mpi4jax collective in this
  1620	    # module): tracing is SPMD-symmetric — every rank traces the same
  1621	    # jitted functions in the same order.  Rank-subset tracing would
  1622	    # block in the trace-time Sendrecv exactly like rank-subset
  1623	    # EXECUTION blocks the traced sendrecv.  Set
  1624	    # ``LEGOESM_LATLON_STATIC_METRIC_PAD=0`` to restore the traced
  1625	    # exchange (A/B + kill-switch lever).
  1626	    import os
  1627	    if (
  1628	        not north_fold
  1629	        and interior.ndim == 1
  1630	        and not isinstance(interior, jax.core.Tracer)
  1631	        and not isinstance(south_value, jax.core.Tracer)
  1632	        and not isinstance(north_value, jax.core.Tracer)
  1633	        and os.environ.get("LEGOESM_LATLON_STATIC_METRIC_PAD", "1") != "0"
  1634	    ):
  1635	        if halo > interior.shape[0]:
  1636	            raise ValueError(
  1637	                f"pad_with_pole_bc_lat_mpi: halo={halo} exceeds "
  1638	                f"n_lat_local={interior.shape[0]} on rank {layout.rank}"
  1639	            )
  1640	        return _pad_static_wall_bc_mpi(
  1641	            interior, layout, halo, south_value, north_value,
  1642	        )
  1643	
  1644	    # 1D lat-axis metrics (sin_lat, cos_lat_v_interior, dx_cell, …)
  1645	    # don't have a lon axis to pole-fold over.  Build the result
  1646	    # directly: constant pad at pole-touching boundaries, sendrecv
  1647	    # at interior cuts — no call to ``exchange_halo_latlon`` (which
  1648	    # assumes axis 1 = lon and would fail on ndim=1).
  1649	    if interior.ndim == 1:
  1650	        return _pad_with_pole_bc_lat_mpi_1d(
  1651	            interior, layout, halo=halo,
  1652	            south_value=south_value, north_value=north_value,
  1653	        )
  1654	
  1655	    # Step 1: run the standard halo exchange.  Boundary ranks will
  1656	    # get pole-folded values; interior cuts will get sendrecv'd
  1657	    # neighbour values (which is what we want — those are NOT to
  1658	    # be overwritten).
  1659	    padded = exchange_halo_latlon(
  1660	        interior, layout, halo=halo,
  1661	        is_vector_v=is_vector_v, is_vector_u=is_vector_u,
  1662	    )
  1663	    # Step 2: where this rank touches a pole, replace the
  1664	    # pole-folded slab with the wall-BC constant.  Slab shapes
  1665	    # match by construction.  We use the bcast-tuple pattern so
  1666	    # this works for 1D (sin_lat), 2D (face metrics) and 3D
  1667	    # (u-on-face-with-levels) fields uniformly.
  1668	    keep_north_fold = north_fold and _is_tripolar_layout(layout)
  1669	    if layout.south_rank is None:
  1670	        south_const = jnp.full(
  1671	            (halo,) + interior.shape[1:],
  1672	            jnp.asarray(south_value, dtype=interior.dtype),
  1673	        )
  1674	        padded = jnp.concatenate([south_const, padded[halo:]], axis=0)
  1675	    if layout.north_rank is None and not keep_north_fold:
  1676	        # Regular north pole / wall-BC caller → ``north_value`` constant.
  1677	        # Only an explicit ``north_fold=True`` request on an active
  1678	        # tripolar layout keeps the permutation-folded north slab that
  1679	        # ``exchange_halo_latlon`` produced above.
  1680	        north_const = jnp.full(
  1681	            (halo,) + interior.shape[1:],
  1682	            jnp.asarray(north_value, dtype=interior.dtype),
  1683	        )
  1684	        padded = jnp.concatenate([padded[:-halo], north_const], axis=0)
  1685	    return padded
  1686	
  1687	
  1688	# ============================================================================
  1689	# Fused multi-field halo exchange (scaling campaign, audit lever O4)
  1690	# ============================================================================
  1691	#
  1692	# The lat-lon band MPI step issues O(30) independent wall-BC cell pads per
  1693	# step, each paying its own token-serialized sendrecv pair (~200-300 us
  1694	# latency on Ginsburg CPU nodes — the measured rank-growing term of the
  1695	# baroclinic phase, jobs 8458934/8458989).  mpi4jax sendrecvs do NOT
  1696	# overlap (token chain), so N independent pads cost N x latency.  Fusing
  1697	# independent same-dataflow-level pads into ONE concatenated sendrecv per
  1698	# cut per dtype group cuts that latency term by the cluster size while
  1699	# exchanging bit-identical bytes (concat -> sendrecv -> split is value-
  1700	# identical to per-field sendrecvs; no arithmetic).
  1701	#
  1702	# Scope: scalar wall-BC fields ONLY (``pad_ns_zero`` /
  1703	# ``pad_with_pole_bc_lat`` with constant boundary values, no
  1704	# ``north_fold``, no ``is_vector_*``) — boundary slabs are constant fills,
  1705	# so per-field flags reduce to per-field constants and the interior cut
  1706	# exchange is flag-independent.  Pole-fold / tripolar-fold callers keep
  1707	# the single-field path.
  1708	
  1709	
  1710	def pad_with_pole_bc_lat_multi_2d(
  1711	    fields,
  1712	    layout: "LatLon2DLayout",
  1713	    halo: int = 1,
  1714	    south_values=None,
  1715	    north_values=None,
  1716	):
  1717	    """Fused lat-axis wall pad for a 2-D PENCIL layout (codex consult #3).
  1718	
  1719	    The 2-D twin of :func:`pad_with_pole_bc_lat_multi_mpi`: N independent
  1720	    wall-BC scalars share ONE lat sendrecv pair per cut per dtype group
  1721	    instead of one pair per field, exactly as the band lane does. Only
  1722	    the lat axis is touched (the pencil's lon halo is a separate
  1723	    dispatched exchange) — the same contract as the single-field
  1724	    :func:`pad_with_pole_bc_lat_2d`, which this is value-identical to
  1725	    (both fill pole rows with the constants and sendrecv interior cuts;
  1726	    concatenate/slice carry native VJPs and the exchange is the shared
  1727	    AD-safe :func:`get_sendrecv_vjp`).
  1728	
  1729	    Guard: the halo must fit the SMALLEST local lat block, the same
  1730	    condition :func:`_pad_lat_wall_2d` enforces — a neighbour owning
  1731	    fewer rows would send a mismatched slab and hang.
  1732	    """
  1733	    fields = tuple(fields)
  1734	    n = len(fields)
  1735	    if n == 0:
  1736	        return ()
  1737	    if halo <= 0:
  1738	        return fields
  1739	    if south_values is None:
  1740	        south_values = (0.0,) * n
  1741	    if north_values is None:
  1742	        north_values = (0.0,) * n
  1743	    south_values = tuple(south_values)
  1744	    north_values = tuple(north_values)
  1745	    if len(south_values) != n or len(north_values) != n:
  1746	        raise ValueError(
  1747	            "pad_with_pole_bc_lat_multi_2d: south_values/north_values must "
  1748	            f"match len(fields)={n}; got {len(south_values)}/"
  1749	            f"{len(north_values)}.")
  1750	    n_lat_local = fields[0].shape[0]
  1751	    for i, f in enumerate(fields):
  1752	        if f.shape[0] != n_lat_local:
  1753	            raise ValueError(
  1754	                "pad_with_pole_bc_lat_multi_2d: all fields must share "
  1755	                f"n_lat_local (axis 0); field 0 has {n_lat_local}, field "
  1756	                f"{i} has {f.shape[0]}.")
  1757	    min_lat_block = layout.n_lat_global // layout.proc_lat
  1758	    if halo > min_lat_block:
  1759	        raise ValueError(
  1760	            f"pad_with_pole_bc_lat_multi_2d: halo={halo} exceeds the "
  1761	            f"smallest local lat block ({min_lat_block}); a neighbour "
  1762	            f"would send/recv a mismatched halo and the exchange would "
  1763	            f"abort/hang.")
  1764	
  1765	    south_slabs: list = [None] * n
  1766	    north_slabs: list = [None] * n
  1767	    if layout.south_rank is None:
  1768	        for i, f in enumerate(fields):
  1769	            south_slabs[i] = jnp.full(
  1770	                (halo,) + f.shape[1:],
  1771	                jnp.asarray(south_values[i], dtype=f.dtype))
  1772	    if layout.north_rank is None:
  1773	        for i, f in enumerate(fields):
  1774	            north_slabs[i] = jnp.full(
  1775	                (halo,) + f.shape[1:],
  1776	                jnp.asarray(north_values[i], dtype=f.dtype))
  1777	
  1778	    if layout.south_rank is not None or layout.north_rank is not None:
  1779	        import mpi4jax
  1780	        from mpi4py import MPI
  1781	
  1782	        comm = MPI.COMM_WORLD
  1783	        sendrecv = get_sendrecv_vjp(mpi4jax)
  1784	        # dtype groups in first-appearance order — trace-deterministic, so
  1785	        # every rank issues the same fused schedule (pairing depends on it).
  1786	        groups: dict = {}
  1787	        for i, f in enumerate(fields):
  1788	            groups.setdefault(jnp.dtype(f.dtype), []).append(i)
  1789	        for idxs in groups.values():
  1790	            sizes = [
  1791	                halo * int(np.prod(fields[i].shape[1:], dtype=np.int64))
  1792	                for i in idxs
  1793	            ]
  1794	            offsets = np.concatenate([[0], np.cumsum(sizes)])
  1795	            if layout.south_rank is not None:
  1796	                send_bot = jnp.concatenate(
  1797	                    [fields[i][:halo].reshape(-1) for i in idxs])
  1798	                recv_south = sendrecv(
  1799	                    send_bot, jnp.zeros_like(send_bot),
  1800	                    layout.south_rank, layout.south_rank,
  1801	                    layout.rank, layout.south_rank, comm)
  1802	                for k, i in enumerate(idxs):
  1803	                    south_slabs[i] = recv_south[
  1804	                        offsets[k]:offsets[k + 1]
  1805	                    ].reshape((halo,) + fields[i].shape[1:])
  1806	            if layout.north_rank is not None:
  1807	                send_top = jnp.concatenate(
  1808	                    [fields[i][-halo:].reshape(-1) for i in idxs])
  1809	                recv_north = sendrecv(
  1810	                    send_top, jnp.zeros_like(send_top),
  1811	                    layout.north_rank, layout.north_rank,
  1812	                    layout.rank, layout.north_rank, comm)
  1813	                for k, i in enumerate(idxs):
  1814	                    north_slabs[i] = recv_north[
  1815	                        offsets[k]:offsets[k + 1]
  1816	                    ].reshape((halo,) + fields[i].shape[1:])
  1817	
  1818	    return tuple(
  1819	        jnp.concatenate([south_slabs[i], fields[i], north_slabs[i]], axis=0)
  1820	        for i in range(n)
  1821	    )
  1822	
  1823	
  1824	def pad_with_pole_bc_lat_multi_mpi(
  1825	    fields,
  1826	    layout: LatLonBandLayout,
  1827	    halo: int = 1,
  1828	    south_values=None,
  1829	    north_values=None,
  1830	):
  1831	    """Fused MPI variant of N independent ``pad_with_pole_bc_lat`` calls.
  1832	
  1833	    Pads every field in ``fields`` along the lat axis (axis 0) with
  1834	    ``halo`` rows per side: constant ``south_values[i]`` /
  1835	    ``north_values[i]`` at pole-touching boundaries, MPI-sendrecv'd
  1836	    neighbour rows at interior partition cuts.  All fields must share
  1837	    ``n_lat_local`` (axis 0); trailing shapes and dtypes may differ
  1838	    (fields are flattened and concatenated per dtype group — ONE
  1839	    sendrecv pair per cut per dtype group instead of one per field).
  1840	
  1841	    Value-identical to ``tuple(pad_with_pole_bc_lat_mpi(f, layout,
  1842	    halo, sv, nv) for ...)`` for wall-BC scalars: the single-field path
  1843	    pole-folds at boundary ranks and then overwrites the boundary slabs
  1844	    with the constants, so skipping the fold and filling constants
  1845	    directly produces the same result with less local compute.
  1846	
  1847	    AD-safe: the fused buffer goes through the same
  1848	    :func:`get_sendrecv_vjp` custom-vjp as the single-field path;
  1849	    ``concatenate``/``slice`` carry native JAX VJPs.
  1850	
  1851	    Returns a tuple of padded arrays, in input order.
  1852	    """
  1853	    fields = tuple(fields)
  1854	    n = len(fields)
  1855	    if n == 0:
  1856	        return ()
  1857	    if south_values is None:
  1858	        south_values = (0.0,) * n
  1859	    if north_values is None:
  1860	        north_values = (0.0,) * n
  1861	    south_values = tuple(south_values)
  1862	    north_values = tuple(north_values)
  1863	    if len(south_values) != n or len(north_values) != n:
  1864	        raise ValueError(
  1865	            "pad_with_pole_bc_lat_multi_mpi: south_values/north_values "
  1866	            f"must match len(fields)={n}; got {len(south_values)}/"
  1867	            f"{len(north_values)}."
  1868	        )
  1869	    if halo <= 0:
  1870	        return fields
   680	        if north_fold or is_vector_u:
   681	            raise NotImplementedError(
   682	                "pad_with_pole_bc_lat 2-D pencil: tripolar north_fold / "
   683	                "is_vector_u need the lat-pencil transpose (wall-pole 2-D "
   684	                f"only). north_fold={north_fold!r} is_vector_u={is_vector_u!r}"
   685	            )
   686	        return pad_with_pole_bc_lat_2d(
   687	            interior, topology, halo=halo,
   688	            south_value=south_value, north_value=north_value,
   689	        )
   690	    if not isinstance(topology, LatLonBandLayout):
   691	        return jnp.pad(
   692	            interior, pad_widths,
   693	            constant_values=((south_value, north_value),)
   694	            + ((0, 0),) * (interior.ndim - 1),
   695	        )
   696	    return pad_with_pole_bc_lat_mpi(
   697	        interior, topology,
   698	        halo=halo,
   699	        south_value=south_value,
   700	        north_value=north_value,
   701	        is_vector_v=is_vector_v,
   702	        is_vector_u=is_vector_u,
   703	        north_fold=north_fold,
   704	    )
   705	
   706	
   707	def pad_with_pole_bc_lat_multi(
   708	    fields,
   709	    halo: int = 1,
   710	    south_values=None,
   711	    north_values=None,
   712	) -> tuple:
   713	    """Batched :func:`pad_with_pole_bc_lat` for independent wall-BC scalars.
   714	
   715	    Pads every field in ``fields`` along the lat axis with constant
   716	    boundary values, backend-dispatched.  Value-identical to calling
   717	    :func:`pad_with_pole_bc_lat` once per field — but under the MPI
   718	    lat-lon band backend the interior partition cuts are exchanged in
   719	    ONE fused sendrecv pair per cut per dtype group instead of one pair
   720	    per field.  mpi4jax sendrecvs are token-serialized (no overlap), so
   721	    each fused cluster of N pads saves ``(N-1) x 2`` sendrecv latencies
   722	    per step — the measured rank-growing term of the ocean baroclinic
   723	    phase (scaling campaign audit lever O4).
   724	
   725	    Scalar wall-BC fields only: no ``is_vector_*`` / ``north_fold``
   726	    support (those callers keep the single-field path; their boundary
   727	    handling is field-specific, while the interior-cut exchange this
   728	    helper fuses is flag-independent).
   729	
   730	    Set ``LEGOESM_LATLON_FUSED_HALO=0`` to force the per-field
   731	    single-exchange fallback (A/B lever; trace-time Python switch, same
   732	    pattern as the other feature gates).
   733	
   734	    Parameters
   735	    ----------
   736	    fields : sequence of jax.Array
   737	        Fields to pad along axis 0.  Must share ``n_lat`` (axis 0);
   738	        trailing shapes / dtypes may differ.
   739	    halo : int
   740	    south_values, north_values : sequence of float, optional
   741	        Per-field boundary constants (default all-zero, i.e.
   742	        ``pad_ns_zero`` semantics).
   743	
   744	    Returns
   745	    -------
   746	    tuple of jax.Array, in input order.
   747	    """
   748	    fields = tuple(fields)
   749	    n = len(fields)
   750	    if n == 0:
   751	        return ()
   752	    if south_values is None:
   753	        south_values = (0.0,) * n
   754	    if north_values is None:
   755	        north_values = (0.0,) * n
   756	    south_values = tuple(south_values)
   757	    north_values = tuple(north_values)
   758	    if len(south_values) != n or len(north_values) != n:
   759	        raise ValueError(
   760	            "pad_with_pole_bc_lat_multi: south_values/north_values must "
   761	            f"match len(fields)={n}; got {len(south_values)}/"
   762	            f"{len(north_values)}."
   763	        )
   764	
   765	    import os
   766	
   767	    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
   768	
   769	    # SPMD leg of the message-aggregation lever (audit item 7): ONE
   770	    # ppermute pair per direction per dtype group instead of one per
   771	    # field.  OPT-IN (default off — flip per deck only with a measured
   772	    # GPU A/B receipt, per the audit item's contract).  Value-identical
   773	    # to the per-field pads (the exchange is a bit-copy).
   774	    _spmd_fused = os.environ.get(
   775	        "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
   776	    if _spmd_fused:
   777	        mesh = _spmd_lat_mesh()
   778	        if mesh is not None:
   779	            from legoesm.parallel.latlon_spmd import (
   780	                make_latlon_band_wall_multi_pad_body,
   781	            )
   782	            body = make_latlon_band_wall_multi_pad_body(
   783	                mesh, halo=halo,
   784	                south_values=south_values, north_values=north_values,
   785	                n_fields=n)
   786	            return body(*fields)
   787	
   788	    fused = os.environ.get("LEGOESM_LATLON_FUSED_HALO", "1") != "0"
   789	    if get_halo_backend() == "mpi" and fused:
   790	        from legoesm.parallel.latlon_mpi import (
   791	            LatLon2DLayout,
   792	            LatLonBandLayout,
   793	            pad_with_pole_bc_lat_multi_2d,
   794	            pad_with_pole_bc_lat_multi_mpi,
   795	        )
   796	        topology = get_mpi_topology()
   797	        if isinstance(topology, LatLonBandLayout):
   798	            return pad_with_pole_bc_lat_multi_mpi(
   799	                fields, topology, halo=halo,
   800	                south_values=south_values, north_values=north_values,
   801	            )
   802	        if isinstance(topology, LatLon2DLayout):
   803	            # 2-D pencil twin (codex consult #3, 2026-08-04): same
   804	            # dtype-group fusion on the lat axis; lon stays a separate
   805	            # dispatched exchange, as in the single-field 2-D path.
   806	            return pad_with_pole_bc_lat_multi_2d(
   807	                fields, topology, halo=halo,
   808	                south_values=south_values, north_values=north_values,
   809	            )
   810	    # Local backend / non-latlon topology / fused-off: per-field pads
   811	    # (bit-identical semantics; under band MPI this is the legacy
   812	    # one-sendrecv-pair-per-field schedule).  The 2-D pencil now has its
   813	    # own fused branch above (2026-08-04); it lands here only when the
   814	    # fused gate is off.
   815	    return tuple(
   816	        pad_with_pole_bc_lat(
   817	            f, halo=halo,
   818	            south_value=south_values[i], north_value=north_values[i],
   819	        )
   820	        for i, f in enumerate(fields)
   821	    )
   822	
   823	
   824	# ============================================================================
   825	# Wide-halo band widening (opt-in wide-halo split-explicit barotropic)
   826	# ============================================================================
   827	
   828	def band_pole_flags():
   829	    """(south_is_pole, north_is_pole) for the ACTIVE lat-band backend.
   830	
   831	    Booleans may be Python bools (local / MPI band — static at trace time)
   832	    or traced scalars (SPMD: derived from ``lax.axis_index`` inside the
   833	    shard_map body, where a rank-static branch is impossible because the
   834	    body is one uniform program).  Callers must therefore consume them with
   835	    ``jnp.where``-style selects, never Python ``if``.
   836	
   837	    Local backend: a single band owns both physical poles → (True, True).
   838	    MPI band layout: pole ownership is ``south_rank/north_rank is None``.
   839	    2-D pencil: pole ownership of the proc-row (lat) axis.
   840	    """
   841	    mesh = _spmd_lat_mesh()
   842	    if mesh is not None:
   843	        import jax
   844	
   845	        idx = jax.lax.axis_index("lat")
   846	        n_bands = mesh.shape["lat"]
   847	        return idx == 0, idx == n_bands - 1
   848	    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
   849	    if get_halo_backend() == "mpi":
   850	        from legoesm.parallel.latlon_mpi import (
     1	"""2-rank MPI equivalence for the fused 2-D pencil wall pad.
     2	
     3	``pad_with_pole_bc_lat_multi_2d`` (codex consult #3) must be
     4	value-identical to N single-field ``pad_with_pole_bc_lat_2d`` calls
     5	while issuing ONE sendrecv pair per cut per dtype group.
     6	
     7	Run: mpirun -np 2 python -m pytest tests/distributed/test_latlon_2d_fused_pad.py
     8	"""
     9	import numpy as np
    10	import pytest
    11	
    12	jax = pytest.importorskip("jax")
    13	jnp = jax.numpy
    14	pytest.importorskip("mpi4py")
    15	pytest.importorskip("mpi4jax")
    16	
    17	from legoesm.parallel.latlon_mpi import (  # noqa: E402
    18	    make_latlon_2d_layout,
    19	    pad_with_pole_bc_lat_2d,
    20	    pad_with_pole_bc_lat_multi_2d,
    21	)
    22	
    23	
    24	def _layout(rank, size, n_lat=8, n_lon=6):
    25	    # proc_lat=size, proc_lon=1: lat is split (the axis this pad touches).
    26	    return make_latlon_2d_layout(rank, size, 1, n_lat, n_lon)
    27	
    28	
    29	def test_fused_2d_pad_matches_per_field():
    30	    from mpi4py import MPI
    31	
    32	    comm = MPI.COMM_WORLD
    33	    rank, size = comm.Get_rank(), comm.Get_size()
    34	    if size < 2:
    35	        pytest.skip("needs >=2 ranks")
    36	    lay = _layout(rank, size)
    37	    nl = lay.n_lat_local
    38	    rng = np.random.default_rng(rank)
    39	    f32 = jnp.asarray(rng.normal(size=(nl, 6, 3)), dtype=jnp.float32)
    40	    f32b = jnp.asarray(rng.normal(size=(nl, 6)), dtype=jnp.float32)
    41	    f64 = jnp.asarray(rng.normal(size=(nl, 6, 2)), dtype=jnp.float64)
    42	    fields = (f32, f32b, f64)
    43	    sv, nv = (0.0, 1.5, -2.0), (0.5, 0.0, 3.0)
    44	
    45	    fused = pad_with_pole_bc_lat_multi_2d(
    46	        fields, lay, halo=1, south_values=sv, north_values=nv)
    47	    ref = tuple(
    48	        pad_with_pole_bc_lat_2d(f, lay, halo=1,
    49	                                south_value=sv[i], north_value=nv[i])
    50	        for i, f in enumerate(fields))
    51	    for i, (a, b) in enumerate(zip(fused, ref)):
    52	        assert a.shape == b.shape, (i, a.shape, b.shape)
    53	        assert a.dtype == b.dtype
    54	        np.testing.assert_array_equal(
    55	            np.asarray(a), np.asarray(b),
    56	            err_msg=f"fused 2-D pad differs from per-field on field {i}")
    57	
    58	
    59	def test_fused_2d_pad_is_ad_safe():
    60	    from mpi4py import MPI
    61	
    62	    comm = MPI.COMM_WORLD
    63	    rank, size = comm.Get_rank(), comm.Get_size()
    64	    if size < 2:
    65	        pytest.skip("needs >=2 ranks")
    66	    lay = _layout(rank, size)
    67	    nl = lay.n_lat_local
    68	    x = jnp.asarray(np.random.default_rng(0).normal(size=(nl, 4)),
    69	                    dtype=jnp.float64)
    70	
    71	    def loss(a):
    72	        out = pad_with_pole_bc_lat_multi_2d((a, a * 2.0), lay, halo=1)
    73	        return sum(jnp.sum(o ** 2) for o in out)
    74	
    75	    g = jax.grad(loss)(x)
    76	    assert np.all(np.isfinite(np.asarray(g)))

exec
/usr/bin/bash -lc "rg -n -C 8 \"def _pad_lat_wall_2d|def pad_with_pole_bc_lat_2d|def exchange_halo_lon|def pad_lon|def _sendrecv|tag=|sendtag|recvtag|sendrecv\\(\" packages/core/legoesm/parallel/latlon_mpi.py packages/core/legoesm/grids/halo_latlon.py && nl -ba packages/core/legoesm/parallel/latlon_mpi.py | sed -n '400,760p' && nl -ba packages/core/legoesm/parallel/latlon_mpi.py | sed -n '1824,2025p' && rg -n \"test_latlon_2d|distributed\" pyproject.toml pytest.ini setup.cfg tox.ini .github 2>/dev/null || true && rg -n \"pad_with_pole_bc_lat_multi\\(\" packages/core/legoesm tests | head -200" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 1081ms:
packages/core/legoesm/parallel/latlon_mpi.py-929-    # ---- South halo ----
packages/core/legoesm/parallel/latlon_mpi.py-930-    if layout.south_rank is not None:
packages/core/legoesm/parallel/latlon_mpi.py-931-        # Receive from southern neighbour's TOP `halo` rows (their
packages/core/legoesm/parallel/latlon_mpi.py-932-        # ``field[-halo:]``), placed as our ``[0:halo)``.  We send our
packages/core/legoesm/parallel/latlon_mpi.py-933-        # bottom `halo` rows in return (the south neighbour's north
packages/core/legoesm/parallel/latlon_mpi.py-934-        # halo).
packages/core/legoesm/parallel/latlon_mpi.py-935-        send_bot = field[:halo].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py-936-        recv_template = jnp.zeros_like(send_bot)
packages/core/legoesm/parallel/latlon_mpi.py:937:        recv_south = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-938-            send_bot, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-939-            layout.south_rank,         # source
packages/core/legoesm/parallel/latlon_mpi.py-940-            layout.south_rank,         # dest
packages/core/legoesm/parallel/latlon_mpi.py:941:            layout.rank,               # sendtag = sender's rank
packages/core/legoesm/parallel/latlon_mpi.py:942:            layout.south_rank,         # recvtag = source's rank
packages/core/legoesm/parallel/latlon_mpi.py-943-            comm,
packages/core/legoesm/parallel/latlon_mpi.py-944-        )
packages/core/legoesm/parallel/latlon_mpi.py-945-        south_halo = recv_south.reshape((halo,) + trailing)
packages/core/legoesm/parallel/latlon_mpi.py-946-    else:
packages/core/legoesm/parallel/latlon_mpi.py-947-        south_halo = _pole_fold_south(field, halo, negate=is_vector_v)
packages/core/legoesm/parallel/latlon_mpi.py-948-
packages/core/legoesm/parallel/latlon_mpi.py-949-    # ---- North halo ----
packages/core/legoesm/parallel/latlon_mpi.py-950-    if layout.north_rank is not None:
packages/core/legoesm/parallel/latlon_mpi.py-951-        send_top = field[-halo:].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py-952-        recv_template = jnp.zeros_like(send_top)
packages/core/legoesm/parallel/latlon_mpi.py:953:        recv_north = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-954-            send_top, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-955-            layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-956-            layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-957-            layout.rank,
packages/core/legoesm/parallel/latlon_mpi.py-958-            layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-959-            comm,
packages/core/legoesm/parallel/latlon_mpi.py-960-        )
packages/core/legoesm/parallel/latlon_mpi.py-961-        north_halo = recv_north.reshape((halo,) + trailing)
packages/core/legoesm/parallel/latlon_mpi.py-962-    else:
packages/core/legoesm/parallel/latlon_mpi.py-963-        north_halo = _north_halo_boundary(
packages/core/legoesm/parallel/latlon_mpi.py-964-            field, halo, layout, is_vector_u, is_vector_v,
packages/core/legoesm/parallel/latlon_mpi.py-965-        )
packages/core/legoesm/parallel/latlon_mpi.py-966-
packages/core/legoesm/parallel/latlon_mpi.py-967-    return jnp.concatenate([south_halo, field, north_halo], axis=0)
packages/core/legoesm/parallel/latlon_mpi.py-968-
packages/core/legoesm/parallel/latlon_mpi.py-969-
packages/core/legoesm/parallel/latlon_mpi.py:970:def exchange_halo_lon(
packages/core/legoesm/parallel/latlon_mpi.py-971-    field: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py-972-    west_rank: int,
packages/core/legoesm/parallel/latlon_mpi.py-973-    east_rank: int,
packages/core/legoesm/parallel/latlon_mpi.py-974-    rank: int,
packages/core/legoesm/parallel/latlon_mpi.py-975-    halo: int = 1,
packages/core/legoesm/parallel/latlon_mpi.py-976-) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-977-    """Exchange ``halo`` ghost LONGITUDE columns on each side (W/E).
packages/core/legoesm/parallel/latlon_mpi.py-978-
--
packages/core/legoesm/parallel/latlon_mpi.py-1036-            "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1037-        ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1038-
packages/core/legoesm/parallel/latlon_mpi.py-1039-    comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py-1040-    sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1041-    trailing_lat = field.shape[0]
packages/core/legoesm/parallel/latlon_mpi.py-1042-    other = field.shape[2:]  # (nlev,) or ()
packages/core/legoesm/parallel/latlon_mpi.py-1043-
packages/core/legoesm/parallel/latlon_mpi.py:1044:    # PHASE-CONSTANT tags with sendtag == recvtag (one per shift
packages/core/legoesm/parallel/latlon_mpi.py-1045-    # direction, +1 for the opposite) — the AD-safe convention from
packages/core/legoesm/parallel/latlon_mpi.py-1046-    # plane_mpi._TAG_EW.  The shared sendrecv VJP's backward swaps
packages/core/legoesm/parallel/latlon_mpi.py-1047-    # source<->dest but KEEPS the tags, so a rank-as-tag scheme
packages/core/legoesm/parallel/latlon_mpi.py:1048:    # (sendtag=rank, recvtag=source) mismatches on the reverse ring at
packages/core/legoesm/parallel/latlon_mpi.py-1049-    # proc_lon>=3 (codex review MAJOR 2026-06-13: gradients would hang).
packages/core/legoesm/parallel/latlon_mpi.py-1050-    # With a single tag per phase, forward AND backward messages match
packages/core/legoesm/parallel/latlon_mpi.py-1051-    # for any ring size.  (Numerically distinct from plane_mpi's
packages/core/legoesm/parallel/latlon_mpi.py-1052-    # 1000/2000; not strictly distinct from the N/S rank-tags, but a
packages/core/legoesm/parallel/latlon_mpi.py-1053-    # cross-match would also need the same comm + same source + same
packages/core/legoesm/parallel/latlon_mpi.py-1054-    # dest + concurrent outstanding recvs — E/W and N/S neighbour pairs
packages/core/legoesm/parallel/latlon_mpi.py-1055-    # are disjoint at proc_lon>1, and proc_lon==1 uses the local-wrap
packages/core/legoesm/parallel/latlon_mpi.py-1056-    # fast path, so no cross-match path exists.)
--
packages/core/legoesm/parallel/latlon_mpi.py-1065-    # the neighbour's phase-2 → circular deadlock (observed: job
packages/core/legoesm/parallel/latlon_mpi.py-1066-    # 8476475 timed out in mpi_sendrecv).  Instead each phase is a
packages/core/legoesm/parallel/latlon_mpi.py-1067-    # UNIFORM shift where every send is matched by a recv IN THE SAME
packages/core/legoesm/parallel/latlon_mpi.py-1068-    # phase (a permutation), so no cross-phase ring dependency exists.
packages/core/legoesm/parallel/latlon_mpi.py-1069-
packages/core/legoesm/parallel/latlon_mpi.py-1070-    # Phase 1 — EASTWARD shift: send our EAST edge to the east neighbour,
packages/core/legoesm/parallel/latlon_mpi.py-1071-    # receive the west neighbour's east edge into our WEST halo.
packages/core/legoesm/parallel/latlon_mpi.py-1072-    send_e = field[:, -halo:].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py:1073:    recv_w = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1074-        send_e, jnp.zeros_like(send_e),
packages/core/legoesm/parallel/latlon_mpi.py-1075-        west_rank,   # source (recv from west)
packages/core/legoesm/parallel/latlon_mpi.py-1076-        east_rank,   # dest   (send to east)
packages/core/legoesm/parallel/latlon_mpi.py:1077:        _TAG_LON,    # sendtag == recvtag (phase-constant, AD-safe)
packages/core/legoesm/parallel/latlon_mpi.py-1078-        _TAG_LON,
packages/core/legoesm/parallel/latlon_mpi.py-1079-        comm,
packages/core/legoesm/parallel/latlon_mpi.py-1080-    )
packages/core/legoesm/parallel/latlon_mpi.py-1081-    west_halo = recv_w.reshape((trailing_lat, halo) + other)
packages/core/legoesm/parallel/latlon_mpi.py-1082-
packages/core/legoesm/parallel/latlon_mpi.py-1083-    # Phase 2 — WESTWARD shift: send our WEST edge to the west neighbour,
packages/core/legoesm/parallel/latlon_mpi.py-1084-    # receive the east neighbour's west edge into our EAST halo.
packages/core/legoesm/parallel/latlon_mpi.py-1085-    send_w = field[:, :halo].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py:1086:    recv_e = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1087-        send_w, jnp.zeros_like(send_w),
packages/core/legoesm/parallel/latlon_mpi.py-1088-        east_rank,     # source (recv from east)
packages/core/legoesm/parallel/latlon_mpi.py-1089-        west_rank,     # dest   (send to west)
packages/core/legoesm/parallel/latlon_mpi.py:1090:        _TAG_LON + 1,  # phase-2 tag (sendtag == recvtag)
packages/core/legoesm/parallel/latlon_mpi.py-1091-        _TAG_LON + 1,
packages/core/legoesm/parallel/latlon_mpi.py-1092-        comm,
packages/core/legoesm/parallel/latlon_mpi.py-1093-    )
packages/core/legoesm/parallel/latlon_mpi.py-1094-    east_halo = recv_e.reshape((trailing_lat, halo) + other)
packages/core/legoesm/parallel/latlon_mpi.py-1095-
packages/core/legoesm/parallel/latlon_mpi.py-1096-    return jnp.concatenate([west_halo, field, east_halo], axis=1)
packages/core/legoesm/parallel/latlon_mpi.py-1097-
packages/core/legoesm/parallel/latlon_mpi.py-1098-
packages/core/legoesm/parallel/latlon_mpi.py:1099:def _pad_lat_wall_2d(
packages/core/legoesm/parallel/latlon_mpi.py-1100-    field: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py-1101-    layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-1102-    halo: int,
packages/core/legoesm/parallel/latlon_mpi.py-1103-    south_value: float,
packages/core/legoesm/parallel/latlon_mpi.py-1104-    north_value: float,
packages/core/legoesm/parallel/latlon_mpi.py-1105-) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-1106-    """Lat-axis (N/S) wall pad for a 2-D pencil row — the shared core of
packages/core/legoesm/parallel/latlon_mpi.py-1107-    :func:`pad_halo_latlon_2d` and :func:`pad_with_pole_bc_lat_2d`.
--
packages/core/legoesm/parallel/latlon_mpi.py-1145-
packages/core/legoesm/parallel/latlon_mpi.py-1146-    comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py-1147-    sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1148-
packages/core/legoesm/parallel/latlon_mpi.py-1149-    if south_is_pole:
packages/core/legoesm/parallel/latlon_mpi.py-1150-        south = jnp.full((halo,) + trailing, south_value, field.dtype)
packages/core/legoesm/parallel/latlon_mpi.py-1151-    else:
packages/core/legoesm/parallel/latlon_mpi.py-1152-        send_s = field[:halo].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py:1153:        recv_s = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1154-            send_s, jnp.zeros_like(send_s),
packages/core/legoesm/parallel/latlon_mpi.py-1155-            layout.south_rank, layout.south_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1156-            layout.rank, layout.south_rank, comm)
packages/core/legoesm/parallel/latlon_mpi.py-1157-        south = recv_s.reshape((halo,) + trailing)
packages/core/legoesm/parallel/latlon_mpi.py-1158-
packages/core/legoesm/parallel/latlon_mpi.py-1159-    if north_is_pole:
packages/core/legoesm/parallel/latlon_mpi.py-1160-        north = jnp.full((halo,) + trailing, north_value, field.dtype)
packages/core/legoesm/parallel/latlon_mpi.py-1161-    else:
packages/core/legoesm/parallel/latlon_mpi.py-1162-        send_n = field[-halo:].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py:1163:        recv_n = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1164-            send_n, jnp.zeros_like(send_n),
packages/core/legoesm/parallel/latlon_mpi.py-1165-            layout.north_rank, layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1166-            layout.rank, layout.north_rank, comm)
packages/core/legoesm/parallel/latlon_mpi.py-1167-        north = recv_n.reshape((halo,) + trailing)
packages/core/legoesm/parallel/latlon_mpi.py-1168-
packages/core/legoesm/parallel/latlon_mpi.py-1169-    return jnp.concatenate([south, field, north], axis=0)
packages/core/legoesm/parallel/latlon_mpi.py-1170-
packages/core/legoesm/parallel/latlon_mpi.py-1171-
packages/core/legoesm/parallel/latlon_mpi.py:1172:def pad_with_pole_bc_lat_2d(
packages/core/legoesm/parallel/latlon_mpi.py-1173-    interior: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py-1174-    layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-1175-    halo: int = 1,
packages/core/legoesm/parallel/latlon_mpi.py-1176-    south_value: float = 0.0,
packages/core/legoesm/parallel/latlon_mpi.py-1177-    north_value: float = 0.0,
packages/core/legoesm/parallel/latlon_mpi.py-1178-) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-1179-    """Lat-axis-ONLY wall pad for a 2-D pencil layout (NO longitude halo).
packages/core/legoesm/parallel/latlon_mpi.py-1180-
--
packages/core/legoesm/parallel/latlon_mpi.py-1366-            raise ImportError(
packages/core/legoesm/parallel/latlon_mpi.py-1367-                "pad_halo_latlon_mpi: multi-rank lat halo requires "
packages/core/legoesm/parallel/latlon_mpi.py-1368-                "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1369-            ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1370-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py-1371-        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1372-        send_bot = lon_padded[:halo].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py-1373-        recv_template = jnp.zeros_like(send_bot)
packages/core/legoesm/parallel/latlon_mpi.py:1374:        recv_south = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1375-            send_bot, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-1376-            layout.south_rank, layout.south_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1377-            layout.rank, layout.south_rank, comm,
packages/core/legoesm/parallel/latlon_mpi.py-1378-        )
packages/core/legoesm/parallel/latlon_mpi.py-1379-        # Inter-rank exchange: no sign flip (v is continuous across a
packages/core/legoesm/parallel/latlon_mpi.py-1380-        # partition cut, only flips across the actual pole).
packages/core/legoesm/parallel/latlon_mpi.py-1381-        south_halo = recv_south.reshape((halo,) + lon_padded.shape[1:])
packages/core/legoesm/parallel/latlon_mpi.py-1382-
--
packages/core/legoesm/parallel/latlon_mpi.py-1409-            raise ImportError(
packages/core/legoesm/parallel/latlon_mpi.py-1410-                "pad_halo_latlon_mpi: multi-rank lat halo requires "
packages/core/legoesm/parallel/latlon_mpi.py-1411-                "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1412-            ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1413-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py-1414-        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1415-        send_top = lon_padded[-halo:].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py-1416-        recv_template = jnp.zeros_like(send_top)
packages/core/legoesm/parallel/latlon_mpi.py:1417:        recv_north = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1418-            send_top, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-1419-            layout.north_rank, layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1420-            layout.rank, layout.north_rank, comm,
packages/core/legoesm/parallel/latlon_mpi.py-1421-        )
packages/core/legoesm/parallel/latlon_mpi.py-1422-        north_halo = recv_north.reshape((halo,) + lon_padded.shape[1:])
packages/core/legoesm/parallel/latlon_mpi.py-1423-
packages/core/legoesm/parallel/latlon_mpi.py-1424-    return jnp.concatenate([south_halo, lon_padded, north_halo], axis=0)
packages/core/legoesm/parallel/latlon_mpi.py-1425-
--
packages/core/legoesm/parallel/latlon_mpi.py-1466-            raise ImportError(
packages/core/legoesm/parallel/latlon_mpi.py-1467-                "Lat-lon MPI 1-D pad (interior partition cut) requires "
packages/core/legoesm/parallel/latlon_mpi.py-1468-                "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1469-            ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1470-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py-1471-        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1472-        send_bot = interior[:halo]
packages/core/legoesm/parallel/latlon_mpi.py-1473-        recv_template = jnp.zeros_like(send_bot)
packages/core/legoesm/parallel/latlon_mpi.py:1474:        south_slab = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1475-            send_bot, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-1476-            layout.south_rank, layout.south_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1477-            layout.rank, layout.south_rank, comm,
packages/core/legoesm/parallel/latlon_mpi.py-1478-        )
packages/core/legoesm/parallel/latlon_mpi.py-1479-
packages/core/legoesm/parallel/latlon_mpi.py-1480-    # North side
packages/core/legoesm/parallel/latlon_mpi.py-1481-    if layout.north_rank is None:
packages/core/legoesm/parallel/latlon_mpi.py-1482-        north_slab = jnp.full(
--
packages/core/legoesm/parallel/latlon_mpi.py-1490-            raise ImportError(
packages/core/legoesm/parallel/latlon_mpi.py-1491-                "Lat-lon MPI 1-D pad (interior partition cut) requires "
packages/core/legoesm/parallel/latlon_mpi.py-1492-                "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1493-            ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1494-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py-1495-        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1496-        send_top = interior[-halo:]
packages/core/legoesm/parallel/latlon_mpi.py-1497-        recv_template = jnp.zeros_like(send_top)
packages/core/legoesm/parallel/latlon_mpi.py:1498:        north_slab = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1499-            send_top, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-1500-            layout.north_rank, layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1501-            layout.rank, layout.north_rank, comm,
packages/core/legoesm/parallel/latlon_mpi.py-1502-        )
packages/core/legoesm/parallel/latlon_mpi.py-1503-
packages/core/legoesm/parallel/latlon_mpi.py-1504-    return jnp.concatenate([south_slab, interior, north_slab], axis=0)
packages/core/legoesm/parallel/latlon_mpi.py-1505-
packages/core/legoesm/parallel/latlon_mpi.py-1506-
--
packages/core/legoesm/parallel/latlon_mpi.py-1542-    if layout.south_rank is None:
packages/core/legoesm/parallel/latlon_mpi.py-1543-        south = np.full((halo,) + trailing, south_value, dtype=arr.dtype)
packages/core/legoesm/parallel/latlon_mpi.py-1544-    else:
packages/core/legoesm/parallel/latlon_mpi.py-1545-        from mpi4py import MPI
packages/core/legoesm/parallel/latlon_mpi.py-1546-
packages/core/legoesm/parallel/latlon_mpi.py-1547-        south = np.empty((halo,) + trailing, dtype=arr.dtype)
packages/core/legoesm/parallel/latlon_mpi.py-1548-        MPI.COMM_WORLD.Sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1549-            np.ascontiguousarray(arr[:halo]), dest=layout.south_rank,
packages/core/legoesm/parallel/latlon_mpi.py:1550:            sendtag=layout.rank,
packages/core/legoesm/parallel/latlon_mpi.py-1551-            recvbuf=south, source=layout.south_rank,
packages/core/legoesm/parallel/latlon_mpi.py:1552:            recvtag=layout.south_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1553-        )
packages/core/legoesm/parallel/latlon_mpi.py-1554-    if layout.north_rank is None:
packages/core/legoesm/parallel/latlon_mpi.py-1555-        north = np.full((halo,) + trailing, north_value, dtype=arr.dtype)
packages/core/legoesm/parallel/latlon_mpi.py-1556-    else:
packages/core/legoesm/parallel/latlon_mpi.py-1557-        from mpi4py import MPI
packages/core/legoesm/parallel/latlon_mpi.py-1558-
packages/core/legoesm/parallel/latlon_mpi.py-1559-        north = np.empty((halo,) + trailing, dtype=arr.dtype)
packages/core/legoesm/parallel/latlon_mpi.py-1560-        MPI.COMM_WORLD.Sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1561-            np.ascontiguousarray(arr[-halo:]), dest=layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py:1562:            sendtag=layout.rank,
packages/core/legoesm/parallel/latlon_mpi.py-1563-            recvbuf=north, source=layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py:1564:            recvtag=layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1565-        )
packages/core/legoesm/parallel/latlon_mpi.py-1566-    return jnp.concatenate(
packages/core/legoesm/parallel/latlon_mpi.py-1567-        [jnp.asarray(south), jnp.asarray(interior), jnp.asarray(north)],
packages/core/legoesm/parallel/latlon_mpi.py-1568-        axis=0,
packages/core/legoesm/parallel/latlon_mpi.py-1569-    )
packages/core/legoesm/parallel/latlon_mpi.py-1570-
packages/core/legoesm/parallel/latlon_mpi.py-1571-
packages/core/legoesm/parallel/latlon_mpi.py-1572-def pad_with_pole_bc_lat_mpi(
--
packages/core/legoesm/parallel/latlon_mpi.py-1790-            sizes = [
packages/core/legoesm/parallel/latlon_mpi.py-1791-                halo * int(np.prod(fields[i].shape[1:], dtype=np.int64))
packages/core/legoesm/parallel/latlon_mpi.py-1792-                for i in idxs
packages/core/legoesm/parallel/latlon_mpi.py-1793-            ]
packages/core/legoesm/parallel/latlon_mpi.py-1794-            offsets = np.concatenate([[0], np.cumsum(sizes)])
packages/core/legoesm/parallel/latlon_mpi.py-1795-            if layout.south_rank is not None:
packages/core/legoesm/parallel/latlon_mpi.py-1796-                send_bot = jnp.concatenate(
packages/core/legoesm/parallel/latlon_mpi.py-1797-                    [fields[i][:halo].reshape(-1) for i in idxs])
packages/core/legoesm/parallel/latlon_mpi.py:1798:                recv_south = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1799-                    send_bot, jnp.zeros_like(send_bot),
packages/core/legoesm/parallel/latlon_mpi.py-1800-                    layout.south_rank, layout.south_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1801-                    layout.rank, layout.south_rank, comm)
packages/core/legoesm/parallel/latlon_mpi.py-1802-                for k, i in enumerate(idxs):
packages/core/legoesm/parallel/latlon_mpi.py-1803-                    south_slabs[i] = recv_south[
packages/core/legoesm/parallel/latlon_mpi.py-1804-                        offsets[k]:offsets[k + 1]
packages/core/legoesm/parallel/latlon_mpi.py-1805-                    ].reshape((halo,) + fields[i].shape[1:])
packages/core/legoesm/parallel/latlon_mpi.py-1806-            if layout.north_rank is not None:
packages/core/legoesm/parallel/latlon_mpi.py-1807-                send_top = jnp.concatenate(
packages/core/legoesm/parallel/latlon_mpi.py-1808-                    [fields[i][-halo:].reshape(-1) for i in idxs])
packages/core/legoesm/parallel/latlon_mpi.py:1809:                recv_north = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1810-                    send_top, jnp.zeros_like(send_top),
packages/core/legoesm/parallel/latlon_mpi.py-1811-                    layout.north_rank, layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1812-                    layout.rank, layout.north_rank, comm)
packages/core/legoesm/parallel/latlon_mpi.py-1813-                for k, i in enumerate(idxs):
packages/core/legoesm/parallel/latlon_mpi.py-1814-                    north_slabs[i] = recv_north[
packages/core/legoesm/parallel/latlon_mpi.py-1815-                        offsets[k]:offsets[k + 1]
packages/core/legoesm/parallel/latlon_mpi.py-1816-                    ].reshape((halo,) + fields[i].shape[1:])
packages/core/legoesm/parallel/latlon_mpi.py-1817-
--
packages/core/legoesm/parallel/latlon_mpi.py-1926-                for i in idxs
packages/core/legoesm/parallel/latlon_mpi.py-1927-            ]
packages/core/legoesm/parallel/latlon_mpi.py-1928-            offsets = np.concatenate([[0], np.cumsum(sizes)])
packages/core/legoesm/parallel/latlon_mpi.py-1929-
packages/core/legoesm/parallel/latlon_mpi.py-1930-            if layout.south_rank is not None:
packages/core/legoesm/parallel/latlon_mpi.py-1931-                send_bot = jnp.concatenate(
packages/core/legoesm/parallel/latlon_mpi.py-1932-                    [fields[i][:halo].reshape(-1) for i in idxs]
packages/core/legoesm/parallel/latlon_mpi.py-1933-                )
packages/core/legoesm/parallel/latlon_mpi.py:1934:                recv_south = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1935-                    send_bot, jnp.zeros_like(send_bot),
packages/core/legoesm/parallel/latlon_mpi.py-1936-                    layout.south_rank,      # source
packages/core/legoesm/parallel/latlon_mpi.py-1937-                    layout.south_rank,      # dest
packages/core/legoesm/parallel/latlon_mpi.py:1938:                    layout.rank,            # sendtag = sender's rank
packages/core/legoesm/parallel/latlon_mpi.py:1939:                    layout.south_rank,      # recvtag = source's rank
packages/core/legoesm/parallel/latlon_mpi.py-1940-                    comm,
packages/core/legoesm/parallel/latlon_mpi.py-1941-                )
packages/core/legoesm/parallel/latlon_mpi.py-1942-                for k, i in enumerate(idxs):
packages/core/legoesm/parallel/latlon_mpi.py-1943-                    south_slabs[i] = recv_south[
packages/core/legoesm/parallel/latlon_mpi.py-1944-                        offsets[k]:offsets[k + 1]
packages/core/legoesm/parallel/latlon_mpi.py-1945-                    ].reshape((halo,) + fields[i].shape[1:])
packages/core/legoesm/parallel/latlon_mpi.py-1946-
packages/core/legoesm/parallel/latlon_mpi.py-1947-            if layout.north_rank is not None:
packages/core/legoesm/parallel/latlon_mpi.py-1948-                send_top = jnp.concatenate(
packages/core/legoesm/parallel/latlon_mpi.py-1949-                    [fields[i][-halo:].reshape(-1) for i in idxs]
packages/core/legoesm/parallel/latlon_mpi.py-1950-                )
packages/core/legoesm/parallel/latlon_mpi.py:1951:                recv_north = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1952-                    send_top, jnp.zeros_like(send_top),
packages/core/legoesm/parallel/latlon_mpi.py-1953-                    layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1954-                    layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1955-                    layout.rank,
packages/core/legoesm/parallel/latlon_mpi.py-1956-                    layout.north_rank,
packages/core/legoesm/parallel/latlon_mpi.py-1957-                    comm,
packages/core/legoesm/parallel/latlon_mpi.py-1958-                )
packages/core/legoesm/parallel/latlon_mpi.py-1959-                for k, i in enumerate(idxs):
   400	
   401	    proc_row, proc_col = divmod(rank, proc_lon)
   402	    lat_start, n_lat_local = _even_split(n_lat, proc_lat, proc_row)
   403	    lon_start, n_lon_local = _even_split(n_lon, proc_lon, proc_col)
   404	
   405	    def _rank_at(r, c):
   406	        return r * proc_lon + c
   407	
   408	    # Latitude = pole-terminated line: None at the grid's top/bottom row.
   409	    south_rank = _rank_at(proc_row - 1, proc_col) if proc_row > 0 else None
   410	    north_rank = (_rank_at(proc_row + 1, proc_col)
   411	                  if proc_row < proc_lat - 1 else None)
   412	    # Longitude = periodic ring: always defined (wrap modulo proc_lon).
   413	    west_rank = _rank_at(proc_row, (proc_col - 1) % proc_lon)
   414	    east_rank = _rank_at(proc_row, (proc_col + 1) % proc_lon)
   415	
   416	    return LatLon2DLayout(
   417	        rank=rank, n_ranks=n_ranks, proc_lat=proc_lat, proc_lon=proc_lon,
   418	        proc_row=proc_row, proc_col=proc_col,
   419	        n_lat_global=n_lat, n_lon_global=n_lon,
   420	        n_lat_local=n_lat_local, n_lon_local=n_lon_local,
   421	        lat_start=lat_start, lat_end=lat_start + n_lat_local,
   422	        lon_start=lon_start, lon_end=lon_start + n_lon_local,
   423	        south_rank=south_rank, north_rank=north_rank,
   424	        west_rank=west_rank, east_rank=east_rank, fold=fold,
   425	    )
   426	
   427	
   428	def scatter_field_latlon_2d(
   429	    global_field: jax.Array, layout: LatLon2DLayout,
   430	) -> jax.Array:
   431	    """Slice the rank's 2-D block ``[lat_start:lat_end, lon_start:lon_end]``
   432	    from a global (n_lat, n_lon[, ...]) field.  Deterministic slice (every
   433	    rank derives the identical global field) — no MPI."""
   434	    return global_field[
   435	        layout.lat_start:layout.lat_end,
   436	        layout.lon_start:layout.lon_end,
   437	    ]
   438	
   439	
   440	def gather_field_latlon_2d(
   441	    local_field, layout: LatLon2DLayout,
   442	    *, is_u_face: bool = False, is_v_face: bool = False,
   443	):
   444	    """Reassemble the global field from every rank's 2-D block (I/O only).
   445	
   446	    mpi4py ``allgather`` of host blocks, placed by (proc_row, proc_col).
   447	    Returns a numpy array on every rank; single-rank returns the input
   448	    as numpy.
   449	
   450	    Staggering (mirror of :func:`scatter_state_latlon_2d`): a cell-centred
   451	    field tiles ``(n_lat_global, n_lon_global)`` exactly.  A FACE field has a
   452	    shared boundary face duplicated on the neighbour, so its block is one
   453	    wider/taller and the assembled global array gains the matching face:
   454	
   455	    * ``is_u_face`` — lon-face ``u`` (block ``n_lon_local+1``): assemble to
   456	      ``n_lon_global+1`` (faces ``0..n_lon``, the trailing the periodic
   457	      closure).  Interior shared east faces overwrite with the identical
   458	      neighbour value; the last block fills the closure column.
   459	    * ``is_v_face`` — lat-face ``v`` (block ``n_lat_local+1``): assemble to
   460	      ``n_lat_global+1``.
   461	
   462	    Passing both flags is an error (a field is one stagger or the other).
   463	    """
   464	    import numpy as _np
   465	
   466	    if is_u_face and is_v_face:
   467	        raise ValueError(
   468	            "gather_field_latlon_2d: a field is u-face OR v-face, not both."
   469	        )
   470	
   471	    local_np = _np.asarray(local_field)
   472	    if layout.n_ranks == 1:
   473	        return local_np
   474	
   475	    from mpi4py import MPI
   476	
   477	    comm = MPI.COMM_WORLD
   478	    if comm.Get_size() != layout.n_ranks:
   479	        # The constructor takes proc_lat/proc_lon explicitly; this guard
   480	        # catches a layout built for a different world size than the one
   481	        # actually running (would silently mis-assemble the global field;
   482	        # codex review 2026-06-13).
   483	        raise ValueError(
   484	            f"gather_field_latlon_2d: layout n_ranks={layout.n_ranks} "
   485	            f"(proc_lat={layout.proc_lat} x proc_lon={layout.proc_lon}) "
   486	            f"!= MPI world size {comm.Get_size()}."
   487	        )
   488	    blocks = comm.allgather(
   489	        (layout.lat_start, layout.lon_start, local_np))
   490	    trailing = local_np.shape[2:]
   491	    # Face fields carry the shared boundary face, so the global array gains
   492	    # the matching extra row/column (else an ``n_lon_local+1`` u block would
   493	    # overrun an ``n_lon_global``-wide buffer — codex round-2).
   494	    lat_g = layout.n_lat_global + (1 if is_v_face else 0)
   495	    lon_g = layout.n_lon_global + (1 if is_u_face else 0)
   496	    out = _np.zeros((lat_g, lon_g) + trailing, dtype=local_np.dtype)
   497	    for ls, los, blk in blocks:
   498	        out[ls:ls + blk.shape[0], los:los + blk.shape[1]] = blk
   499	    return out
   500	
   501	
   502	def make_lon_row_comm(layout: LatLon2DLayout):
   503	    """Create the longitude-ring sub-communicator for this rank's
   504	    ``proc_row`` (the ``proc_lon`` ranks that share a latitude band).
   505	
   506	    COLLECTIVE over ``COMM_WORLD`` (every rank must call it).  ``key=
   507	    proc_col`` orders the sub-comm ranks by longitude block, so an
   508	    ``allgather`` over it returns blocks in west->east order.  Build
   509	    this ONCE at layout/setup time and reuse — ``Split`` is not free.
   510	    Returns the row sub-communicator.
   511	    """
   512	    from mpi4py import MPI
   513	
   514	    return MPI.COMM_WORLD.Split(color=layout.proc_row, key=layout.proc_col)
   515	
   516	
   517	# custom_vjp on SCALAR lon metadata only (NOT the whole layout): the
   518	# layout's ``fold`` field can carry jax.Array permutations, which must
   519	# not become custom-VJP static args (codex MAJOR 2026-06-13).  The public
   520	# wrapper below extracts the scalar fields.  nondiff args = lon_start,
   521	# lon_end, n_lon_global, proc_lon, row_comm (indices 1..5).
   522	@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
   523	def _lon_gather_full_p(local_block, lon_start, lon_end, n_lon_global,
   524	                       proc_lon, row_comm):
   525	    from legoesm.parallel.reductions import mpi4jax_array_result, require_mpi_stack
   526	
   527	    # Checked accessor (not a bare ``import mpi4jax``): runs the GPU-transport
   528	    # preflight so this device-array allgather can't slip a GPU-direct
   529	    # misconfiguration past the fail-closed check.
   530	    mpi4jax, _ = require_mpi_stack()
   531	
   532	    if n_lon_global % proc_lon != 0:
   533	        raise ValueError(
   534	            f"lon_gather_full needs an equal lon split (n_lon="
   535	            f"{n_lon_global} % proc_lon={proc_lon} != 0) for the "
   536	            "allgather; non-uniform splits need alltoallv (not yet "
   537	            "implemented)."
   538	        )
   539	    # (proc_lon, n_lat_local, n_lon_local[, nlev]); key=proc_col ordered
   540	    # the sub-comm ranks west->east, so axis 0 IS lon-block order.
   541	    stacked = mpi4jax_array_result(mpi4jax.allgather(local_block, comm=row_comm))
   542	    return jnp.concatenate(
   543	        [stacked[i] for i in range(stacked.shape[0])], axis=1)
   544	
   545	
   546	def _lon_gather_full_p_fwd(local_block, lon_start, lon_end, n_lon_global,
   547	                           proc_lon, row_comm):
   548	    # No residual: the adjoint depends only on the (static) scalar args.
   549	    return _lon_gather_full_p(
   550	        local_block, lon_start, lon_end, n_lon_global, proc_lon, row_comm), None
   551	
   552	
   553	def _lon_gather_full_p_bwd(lon_start, lon_end, n_lon_global, proc_lon,
   554	                           row_comm, _res, g_full):
   555	    """Adjoint of the lat-pencil gather.
   556	
   557	    Forward ``G`` maps each rank's block ``x_s`` into the full-lon array
   558	    that is REPLICATED across the whole row ring (``y_r = concat_s x_s``
   559	    for every ring rank ``r``).  Hence ``x̄_s = Σ_r ȳ_r[:, block_s]`` —
   560	    sum the cotangent over the ring (``allreduce(SUM)``, the only AD-safe
   561	    collective), then slice this rank's lon block.  (``allgather``'s true
   562	    adjoint is a reduce-scatter; ``allreduce(SUM)`` + slice computes the
   563	    same value without a reduce-scatter primitive.)
   564	    """
   565	    from legoesm.parallel.reductions import mpi4jax_array_result, require_mpi_stack
   566	
   567	    # Checked accessor (see the forward primal): GPU-transport preflight before
   568	    # this device-array allreduce.
   569	    mpi4jax, MPI = require_mpi_stack()
   570	
   571	    summed = mpi4jax_array_result(
   572	        mpi4jax.allreduce(g_full, op=MPI.SUM, comm=row_comm))
   573	    return (summed[:, lon_start:lon_end],)
   574	
   575	
   576	_lon_gather_full_p.defvjp(_lon_gather_full_p_fwd, _lon_gather_full_p_bwd)
   577	
   578	
   579	def lon_gather_full(local_block: jax.Array, layout: LatLon2DLayout,
   580	                    row_comm) -> jax.Array:
   581	    """In-trace lat-pencil transpose: assemble the FULL longitude axis
   582	    for this rank's latitude band from the ``proc_lon`` lon-ring blocks.
   583	
   584	    The crux primitive for any operation that needs all longitudes on a
   585	    lon-split grid — the N/S pole-fold (180° lon shift / tripole perm)
   586	    and the polar filter (per-row rfft) — design doc §3b/§4.  Uses
   587	    ``mpi4jax.allgather`` over the ``proc_row`` sub-comm
   588	    (:func:`make_lon_row_comm`), so each rank ends with
   589	    ``(n_lat_local, n_lon_global[, nlev])``; apply the fold/filter on
   590	    that, then :func:`lon_scatter_full` back to the rank's block.
   591	
   592	    AD-SAFE (custom VJP on :func:`_lon_gather_full_p`): the forward is
   593	    ``allgather`` (no native VJP), but the gather is a linear map whose
   594	    adjoint is exact — a local block contributes to the full-lon array
   595	    on EVERY rank of the row ring, so the cotangent's adjoint is
   596	    ``allreduce(SUM)`` over the row ring (AD-safe) then a slice of this
   597	    rank's lon block.  Requires an EQUAL lon split (``n_lon % proc_lon
   598	    == 0``) so the allgather blocks share a shape; non-uniform splits
   599	    need an ``alltoallv`` (future).
   600	
   601	    COLLECTIVE PRECONDITION (deadlock safety): both the forward gather
   602	    AND its reverse-mode ``allreduce`` are collectives over ``row_comm``,
   603	    so EVERY rank of the row ring MUST execute both the primal and the
   604	    backward pass.  Do NOT place a ``lon_gather_full`` result behind
   605	    rank-dependent Python control flow (some ranks skipping the VJP while
   606	    others enter the backward ``allreduce`` deadlocks).  For
   607	    rank-selective use, keep the call in the traced graph on all ranks
   608	    and gate with arithmetic masks (zero cotangents where inactive).
   609	    """
   610	    return _lon_gather_full_p(
   611	        local_block, layout.lon_start, layout.lon_end,
   612	        layout.n_lon_global, layout.proc_lon, row_comm)
   613	
   614	
   615	def lon_scatter_full(full_field: jax.Array,
   616	                     layout: LatLon2DLayout) -> jax.Array:
   617	    """Slice this rank's longitude block ``[lon_start:lon_end]`` from a
   618	    full-longitude field — inverse of :func:`lon_gather_full`."""
   619	    return full_field[:, layout.lon_start:layout.lon_end]
   620	
   621	
   622	# ============================================================================
   623	# Halo exchange
   624	# ============================================================================
   625	
   626	
   627	def _serial_pad_then_strip_lon(
   628	    field: jax.Array, halo: int, negate: bool,
   629	) -> jax.Array:
   630	    """Run the canonical serial ``pad_halo_latlon*`` then strip the lon halos.
   631	
   632	    Why this exists
   633	    ---------------
   634	    A naive 180° rotation via ``jnp.roll(field, n_lon//2, axis=1)`` does NOT
   635	    bit-reproduce serial ``pad_halo_latlon``.  The serial helper wrap-pads in
   636	    longitude first and only then rolls by ``data.shape[1] // 2 ==
   637	    (n_lon + 2*halo) // 2``; after slicing the interior lon columns the
   638	    result is an *aliased* halo that depends on ``halo``, not a clean
   639	    physical 180° rotation.  See ``tests/parallel/test_latlon_mpi_halo_serial.py``
   640	    for the concrete (4-cell) example.
   641	
   642	    To guarantee Stage-1+ produces bit-identical results to the serial
   643	    path under MPI, this helper delegates the lon-padding + fold + slice
   644	    to the canonical helper itself, then strips the lon halos so the
   645	    output has the rank-local lon shape.
   646	
   647	    This is also why we cannot fix the "physical correctness" question
   648	    here: changing the convention would diverge from the serial dycore.
   649	    That refactor belongs in a separate Stage-0.5 audit of the upstream
   650	    fold convention (tracked in CLAUDE.md follow-up debt).
   651	
   652	    Returns
   653	    -------
   654	    (south_halo, north_halo) : each shape (halo, n_lon[, nlev])
   655	        Ready for concatenation south of ``field[0]`` / north of
   656	        ``field[-1]``.
   657	    """
   658	    # IMPORTANT: call the *_local* helpers directly, NOT the
   659	    # backend-dispatched public ``pad_halo_latlon*``.  This helper
   660	    # runs INSIDE the MPI backend's pole-fold step, so calling the
   661	    # public dispatcher would re-enter the MPI path and recurse
   662	    # forever:
   663	    #   pad_halo_latlon (public)
   664	    #     → pad_halo_latlon_mpi   (MPI branch)
   665	    #       → exchange_halo_latlon
   666	    #         → _pole_fold_south / _north  (at boundary ranks)
   667	    #           → _serial_pad_then_strip_lon
   668	    #             → pad_halo_latlon (public, again!) ← recursion
   669	    # The fix: pole-fold uses the SERIAL implementation directly.
   670	    # That's what the docstring above promised; the early Stage 0
   671	    # version pre-dated the backend-dispatch refactor and called
   672	    # the dispatcher by accident.
   673	    from legoesm.grids.halo_latlon import (
   674	        pad_halo_latlon_local,
   675	        pad_halo_latlon_vector_local,
   676	        pad_halo_latlon_3d_local,
   677	        pad_halo_latlon_vector_3d_local,
   678	    )
   679	    if field.ndim == 2:
   680	        helper = (
   681	            pad_halo_latlon_vector_local if negate
   682	            else pad_halo_latlon_local
   683	        )
   684	        padded = helper(field, halo=halo)        # (n_lat + 2h, n_lon + 2h)
   685	        # Strip the lon halos to recover (halo, n_lon).
   686	        south = padded[:halo, halo:halo + field.shape[1]]
   687	        north = padded[-halo:, halo:halo + field.shape[1]]
   688	    elif field.ndim == 3:
   689	        helper = (
   690	            pad_halo_latlon_vector_3d_local if negate
   691	            else pad_halo_latlon_3d_local
   692	        )
   693	        padded = helper(field, halo=halo)        # (n_lat + 2h, n_lon + 2h, nlev)
   694	        south = padded[:halo, halo:halo + field.shape[1], :]
   695	        north = padded[-halo:, halo:halo + field.shape[1], :]
   696	    else:
   697	        raise ValueError(
   698	            f"_serial_pad_then_strip_lon: field.ndim must be 2 or 3, "
   699	            f"got {field.ndim}"
   700	        )
   701	    return south, north
   702	
   703	
   704	def _pole_fold_south(field: jax.Array, halo: int, negate: bool) -> jax.Array:
   705	    """South-pole halo, bit-identical to serial ``pad_halo_latlon*``
   706	    (stripped of lon halos).
   707	
   708	    Operates on ``field`` *without* lon padding — the helper internally
   709	    lon-pads, pole-folds the lon-padded data (matching serial's
   710	    ``(n_lon + 2*halo) // 2`` lon-shift), then strips the lon halos.
   711	    Returns shape ``(halo, n_lon[, nlev])``.
   712	
   713	    Stage-0 tests in ``tests/parallel/test_latlon_mpi_halo_serial.py``
   714	    pin this contract: ``exchange_halo_latlon(unpadded, ...)`` at a
   715	    pole-touching rank must equal serial
   716	    ``pad_halo_latlon(unpadded, halo)[:, halo:-halo]``.
   717	    """
   718	    south, _ = _serial_pad_then_strip_lon(field, halo, negate)
   719	    return south
   720	
   721	
   722	def _pole_fold_north(field: jax.Array, halo: int, negate: bool) -> jax.Array:
   723	    """North-pole halo — see :func:`_pole_fold_south`."""
   724	    _, north = _serial_pad_then_strip_lon(field, halo, negate)
   725	    return north
   726	
   727	
   728	# ----------------------------------------------------------------------------
   729	# Tripolar north fold (issue #353)
   730	# ----------------------------------------------------------------------------
   731	#
   732	# A tripolar grid (ORCA / eORCA) has a *fold seam* at its northern row
   733	# instead of a geographic pole: cell ``(i, fold_j)`` is identified with
   734	# its fold partner ``(perm[i], fold_j)`` (an i-index reversal), and
   735	# vector components flip sign across the seam.  This is fundamentally
   736	# different from the atmospheric 180°-longitude-roll pole-fold above.
   737	#
   738	# These helpers replicate the serial ocean convention
   739	# (``legoesm.ocean.dynamics.latlon_cgrid_operators.fold_row`` /
   740	# ``pad_ns_scalar`` / ``pad_ns_vector_u`` / ``pad_ns_vector_v``) so the
   741	# MPI northernmost rank produces a north halo that is *bit-identical* to
   742	# the single-rank serial fold.  The descriptor (perm + signs) travels on
   743	# ``LatLonBandLayout.fold``.
   744	
   745	
   746	def _is_tripolar_layout(layout: LatLonBandLayout) -> bool:
   747	    """True iff ``layout`` carries an active tripolar north-fold."""
   748	    fold = layout.fold
   749	    return fold is not None and bool(fold.is_active)
   750	
   751	
   752	def _tripolar_fold_perm_sign(fold, *, is_vector_u: bool, is_vector_v: bool):
   753	    """Resolve ``(perm, sign)`` for the tripolar north fold by field kind.
   754	
   755	    - scalar  (T, S, eta, depth, mask): ``perm_T``, sign ``+1``
   756	    - u-comp. (zonal face velocity):    ``perm_T``, sign ``vector_sign_u``
   757	    - v-comp. (meridional face vel.):   ``perm_v``, sign ``vector_sign_v``
   758	
   759	    Mirrors the serial operators ``pad_ns_scalar`` / ``pad_ns_vector_u``
   760	    / ``pad_ns_vector_v`` in
  1824	def pad_with_pole_bc_lat_multi_mpi(
  1825	    fields,
  1826	    layout: LatLonBandLayout,
  1827	    halo: int = 1,
  1828	    south_values=None,
  1829	    north_values=None,
  1830	):
  1831	    """Fused MPI variant of N independent ``pad_with_pole_bc_lat`` calls.
  1832	
  1833	    Pads every field in ``fields`` along the lat axis (axis 0) with
  1834	    ``halo`` rows per side: constant ``south_values[i]`` /
  1835	    ``north_values[i]`` at pole-touching boundaries, MPI-sendrecv'd
  1836	    neighbour rows at interior partition cuts.  All fields must share
  1837	    ``n_lat_local`` (axis 0); trailing shapes and dtypes may differ
  1838	    (fields are flattened and concatenated per dtype group — ONE
  1839	    sendrecv pair per cut per dtype group instead of one per field).
  1840	
  1841	    Value-identical to ``tuple(pad_with_pole_bc_lat_mpi(f, layout,
  1842	    halo, sv, nv) for ...)`` for wall-BC scalars: the single-field path
  1843	    pole-folds at boundary ranks and then overwrites the boundary slabs
  1844	    with the constants, so skipping the fold and filling constants
  1845	    directly produces the same result with less local compute.
  1846	
  1847	    AD-safe: the fused buffer goes through the same
  1848	    :func:`get_sendrecv_vjp` custom-vjp as the single-field path;
  1849	    ``concatenate``/``slice`` carry native JAX VJPs.
  1850	
  1851	    Returns a tuple of padded arrays, in input order.
  1852	    """
  1853	    fields = tuple(fields)
  1854	    n = len(fields)
  1855	    if n == 0:
  1856	        return ()
  1857	    if south_values is None:
  1858	        south_values = (0.0,) * n
  1859	    if north_values is None:
  1860	        north_values = (0.0,) * n
  1861	    south_values = tuple(south_values)
  1862	    north_values = tuple(north_values)
  1863	    if len(south_values) != n or len(north_values) != n:
  1864	        raise ValueError(
  1865	            "pad_with_pole_bc_lat_multi_mpi: south_values/north_values "
  1866	            f"must match len(fields)={n}; got {len(south_values)}/"
  1867	            f"{len(north_values)}."
  1868	        )
  1869	    if halo <= 0:
  1870	        return fields
  1871	
  1872	    n_lat_local = fields[0].shape[0]
  1873	    for i, f in enumerate(fields):
  1874	        if f.shape[0] != n_lat_local:
  1875	            raise ValueError(
  1876	                "pad_with_pole_bc_lat_multi_mpi: all fields must share "
  1877	                f"n_lat_local (axis 0); field 0 has {n_lat_local}, field "
  1878	                f"{i} has {f.shape[0]}."
  1879	            )
  1880	    if halo > n_lat_local:
  1881	        raise ValueError(
  1882	            f"pad_with_pole_bc_lat_multi_mpi: halo={halo} exceeds "
  1883	            f"n_lat_local={n_lat_local} on rank {layout.rank}."
  1884	        )
  1885	
  1886	    south_slabs: list = [None] * n
  1887	    north_slabs: list = [None] * n
  1888	
  1889	    # Pole-touching boundaries: constant wall-BC fill, no comm.
  1890	    if layout.south_rank is None:
  1891	        for i, f in enumerate(fields):
  1892	            south_slabs[i] = jnp.full(
  1893	                (halo,) + f.shape[1:],
  1894	                jnp.asarray(south_values[i], dtype=f.dtype),
  1895	            )
  1896	    if layout.north_rank is None:
  1897	        for i, f in enumerate(fields):
  1898	            north_slabs[i] = jnp.full(
  1899	                (halo,) + f.shape[1:],
  1900	                jnp.asarray(north_values[i], dtype=f.dtype),
  1901	            )
  1902	
  1903	    # Interior partition cuts: ONE fused sendrecv per cut per dtype group.
  1904	    if layout.south_rank is not None or layout.north_rank is not None:
  1905	        try:
  1906	            import mpi4jax
  1907	            from mpi4py import MPI
  1908	        except ImportError as exc:
  1909	            raise ImportError(
  1910	                "Lat-lon fused MPI halo exchange (n_ranks>1) requires "
  1911	                "mpi4jax and mpi4py."
  1912	            ) from exc
  1913	        comm = MPI.COMM_WORLD
  1914	        sendrecv = get_sendrecv_vjp(mpi4jax)
  1915	
  1916	        # Group field indices by dtype in first-appearance order — the
  1917	        # order is trace-deterministic, so every rank issues the same
  1918	        # fused-message schedule (sendrecv pairing relies on it).
  1919	        groups: dict = {}
  1920	        for i, f in enumerate(fields):
  1921	            groups.setdefault(jnp.dtype(f.dtype), []).append(i)
  1922	
  1923	        for idxs in groups.values():
  1924	            sizes = [
  1925	                halo * int(np.prod(fields[i].shape[1:], dtype=np.int64))
  1926	                for i in idxs
  1927	            ]
  1928	            offsets = np.concatenate([[0], np.cumsum(sizes)])
  1929	
  1930	            if layout.south_rank is not None:
  1931	                send_bot = jnp.concatenate(
  1932	                    [fields[i][:halo].reshape(-1) for i in idxs]
  1933	                )
  1934	                recv_south = sendrecv(
  1935	                    send_bot, jnp.zeros_like(send_bot),
  1936	                    layout.south_rank,      # source
  1937	                    layout.south_rank,      # dest
  1938	                    layout.rank,            # sendtag = sender's rank
  1939	                    layout.south_rank,      # recvtag = source's rank
  1940	                    comm,
  1941	                )
  1942	                for k, i in enumerate(idxs):
  1943	                    south_slabs[i] = recv_south[
  1944	                        offsets[k]:offsets[k + 1]
  1945	                    ].reshape((halo,) + fields[i].shape[1:])
  1946	
  1947	            if layout.north_rank is not None:
  1948	                send_top = jnp.concatenate(
  1949	                    [fields[i][-halo:].reshape(-1) for i in idxs]
  1950	                )
  1951	                recv_north = sendrecv(
  1952	                    send_top, jnp.zeros_like(send_top),
  1953	                    layout.north_rank,
  1954	                    layout.north_rank,
  1955	                    layout.rank,
  1956	                    layout.north_rank,
  1957	                    comm,
  1958	                )
  1959	                for k, i in enumerate(idxs):
  1960	                    north_slabs[i] = recv_north[
  1961	                        offsets[k]:offsets[k + 1]
  1962	                    ].reshape((halo,) + fields[i].shape[1:])
  1963	
  1964	    return tuple(
  1965	        jnp.concatenate([south_slabs[i], fields[i], north_slabs[i]], axis=0)
  1966	        for i in range(n)
  1967	    )
  1968	
  1969	
  1970	# ============================================================================
  1971	# Scatter / gather
  1972	# ============================================================================
  1973	
  1974	
  1975	def scatter_state_latlon(state, layout: LatLonBandLayout):
  1976	    """Extract the rank-local band from a global C-grid lat-lon state.
  1977	
  1978	    Each rank reads its own slice of the global arrays; we do not call
  1979	    MPI here (callers either broadcast a global state from rank 0 or
  1980	    each rank constructs the global initial state locally and slices).
  1981	
  1982	    The C-grid v-field at lat interfaces has one extra row globally
  1983	    (shape ``(n_lat+1, n_lon, ...)``); we give each rank rows
  1984	    ``[lat_start, lat_end+1)`` so neighbouring ranks duplicate the
  1985	    boundary row.  The duplicated row is the *same* face — both ranks
  1986	    must always hold the same value there.  Halo exchange of ``v``
  1987	    after a step uses ``is_vector_v=True`` so the pole-touching rank
  1988	    sign-flips correctly.
  1989	    """
  1990	    s, e = layout.lat_start, layout.lat_end
  1991	
  1992	    T_local = state.T[s:e]
  1993	    p_s_local = state.p_s[s:e]
  1994	    phis_local = state.phis[s:e]
  1995	    u_local = state.u[s:e]
  1996	    # v: lat-interface, one extra global row
  1997	    v_local = state.v[s:e + 1]
  1998	
  1999	    tracers_local = {}
  2000	    if getattr(state, "tracers", None):
  2001	        for name, tr in state.tracers.items():
  2002	            tracers_local[name] = tr[s:e]
  2003	
  2004	    return state._replace(
  2005	        u=u_local,
  2006	        v=v_local,
  2007	        T=T_local,
  2008	        p_s=p_s_local,
  2009	        phis=phis_local,
  2010	        tracers=tracers_local if tracers_local else state.tracers,
  2011	    )
  2012	
  2013	
  2014	def scatter_state_latlon_2d(state, layout: LatLon2DLayout):
  2015	    """Extract the rank-local 2-D block from a global C-grid lat-lon state.
  2016	
  2017	    The 2-D analog of :func:`scatter_state_latlon`: slice BOTH the latitude
  2018	    band ``[lat_start, lat_end)`` and the longitude pencil
  2019	    ``[lon_start, lon_end)``.  Pure indexing — no MPI (each rank slices its own
  2020	    block; callers broadcast a global state or build it locally).
  2021	
  2022	    Staggering (codex design pitfall — slice lat AND lon together, lat first):
  2023	
  2024	    * ``T``/``p_s``/``phis``/tracers : cell-centred → ``[s:e, w:x]``.
  2025	    * ``u`` : LON-face.  The dycore stores u with ``n_lon+1`` faces (faces
.github/workflows/mpi-nightly.yml:32:        # mpi-distributed.yml; bare ".[dev,mpi]" resolves jax 0.10+ which
.github/workflows/mpi-nightly.yml:47:            python -m pytest -q tests/distributed/test_ocean_mpi_conservation.py -k longrun
pyproject.toml:283:"tests/distributed/test_cube_face_scatter_mpi.py" = ["E402"]
pyproject.toml:539:    # distributed_checkpoint}; io keeps only pure-array I/O (cmor_output,
.github/workflows/mpi-distributed.yml:11:  mpi-distributed:
.github/workflows/mpi-distributed.yml:50:            echo "=== MPI distributed test, np=${n} ==="
.github/workflows/mpi-distributed.yml:52:              python -m pytest -q tests/distributed/test_halo_mpi.py
.github/workflows/mpi-distributed.yml:62:              python -m pytest -q tests/distributed/test_plane_pencil_mpi.py
.github/workflows/mpi-distributed.yml:75:              python -m pytest -q tests/distributed/test_latlon_2d_pad_wall_mpi.py
.github/workflows/mpi-distributed.yml:88:              python -m pytest -q tests/distributed/test_latlon_transpose_ad_mpi.py
.github/workflows/ci.yml:138:    # (`tests/atmosphere/`, `tests/distributed/`, etc.) have their
packages/core/legoesm/grids/halo_latlon.py:707:def pad_with_pole_bc_lat_multi(
packages/core/legoesm/grids/halo_latlon.py:896:    padded = pad_with_pole_bc_lat_multi(fields, halo=halo)
packages/core/legoesm/grids/halo_latlon.py:929:    padded = pad_with_pole_bc_lat_multi(interiors, halo=halo + 1)
packages/core/legoesm/grids/operators_latlon_cgrid.py:70:    return pad_with_pole_bc_lat_multi(fields, halo=1)
tests/parallel/test_latlon_fused_halo_serial.py:44:        fused = pad_with_pole_bc_lat_multi(fields, halo=1)
tests/parallel/test_latlon_fused_halo_serial.py:52:        fused = pad_with_pole_bc_lat_multi(
tests/parallel/test_latlon_fused_halo_serial.py:72:        a, b = pad_with_pole_bc_lat_multi((f64, f32), halo=1)
tests/parallel/test_latlon_fused_halo_serial.py:85:        fused = pad_with_pole_bc_lat_multi(fields, halo=1)
tests/parallel/test_latlon_fused_halo_serial.py:93:        assert pad_with_pole_bc_lat_multi(()) == ()
tests/parallel/test_latlon_fused_halo_serial.py:95:            pad_with_pole_bc_lat_multi(
tests/parallel/test_latlon_2d_dispatch_serial.py:143:    outs = pad_with_pole_bc_lat_multi(

exec
/usr/bin/bash -lc "nl -ba packages/core/legoesm/parallel/latlon_mpi.py | sed -n '960,1205p'; rg -n -C 5 \"_TAG_LON|exchange_halo_lon\\(\" packages/core/legoesm; nl -ba .github/workflows/mpi-distributed.yml | sed -n '1,125p'; nl -ba tests/parallel/test_latlon_2d_dispatch_serial.py | sed -n '1,205p'; nl -ba packages/core/legoesm/grids/halo_latlon.py | sed -n '850,965p'; nl -ba packages/core/legoesm/grids/operators_latlon_cgrid.py | sed -n '35,90p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 260ms:
   960	        )
   961	        north_halo = recv_north.reshape((halo,) + trailing)
   962	    else:
   963	        north_halo = _north_halo_boundary(
   964	            field, halo, layout, is_vector_u, is_vector_v,
   965	        )
   966	
   967	    return jnp.concatenate([south_halo, field, north_halo], axis=0)
   968	
   969	
   970	def exchange_halo_lon(
   971	    field: jax.Array,
   972	    west_rank: int,
   973	    east_rank: int,
   974	    rank: int,
   975	    halo: int = 1,
   976	) -> jax.Array:
   977	    """Exchange ``halo`` ghost LONGITUDE columns on each side (W/E).
   978	
   979	    Increment 1 of the lat-lon 2-D pencil decomposition
   980	    (``docs/performance/scaling/latlon_2d_decomposition_design.md``).  Longitude is
   981	    GLOBALLY PERIODIC, so — unlike the N/S :func:`exchange_halo_latlon`
   982	    — there are no pole/wall ends: every rank in the longitude ring
   983	    sends its west edge west and its east edge east, and the wrap is
   984	    just the ring topology (the rank owning the last lon block has the
   985	    rank owning the first as its east neighbour).  This is a STANDALONE
   986	    primitive (takes plain neighbour ranks, not a layout) so the future
   987	    ``LatLon2DLayout`` can call it with ``layout.west_rank`` etc.; it is
   988	    NOT yet wired into any step.
   989	
   990	    When the longitude ring has a single member (``west_rank == east_rank
   991	    == rank`` — i.e. ``proc_lon == 1``, the current 1-D latitude-band
   992	    case) the wrap is performed LOCALLY from the rank's own columns,
   993	    byte-identical to the existing ``jnp.roll`` periodic wrap and
   994	    runnable without mpi4jax.
   995	
   996	    Parameters
   997	    ----------
   998	    field : jax.Array, shape (n_lat_local, n_lon_local[, nlev])
   999	        Rank-local interior field, no lon halos in input.
  1000	    west_rank, east_rank : int
  1001	        MPI ranks of the western / eastern longitude neighbours (ring).
  1002	    rank : int
  1003	        This process's MPI rank.
  1004	    halo : int, default 1
  1005	        Number of ghost columns to add on each lon side.
  1006	
  1007	    Returns
  1008	    -------
  1009	    jax.Array, shape (n_lat_local, n_lon_local + 2*halo[, nlev])
  1010	    """
  1011	    if halo <= 0:
  1012	        return field
  1013	
  1014	    n_lon_local = field.shape[1]
  1015	    if halo > n_lon_local:
  1016	        raise ValueError(
  1017	            f"exchange_halo_lon: halo={halo} exceeds n_lon_local="
  1018	            f"{n_lon_local} on rank {rank}.  Reduce proc_lon or "
  1019	            "increase longitude resolution."
  1020	        )
  1021	
  1022	    # Single-member lon ring (proc_lon == 1): local periodic wrap — the
  1023	    # west ghost is the rank's own EAST edge, the east ghost its WEST
  1024	    # edge.  Byte-identical to the legacy ``jnp.roll`` wrap; no MPI.
  1025	    if west_rank == rank and east_rank == rank:
  1026	        west_halo = field[:, -halo:]
  1027	        east_halo = field[:, :halo]
  1028	        return jnp.concatenate([west_halo, field, east_halo], axis=1)
  1029	
  1030	    try:
  1031	        import mpi4jax
  1032	        from mpi4py import MPI
  1033	    except ImportError as exc:
  1034	        raise ImportError(
  1035	            "Lat-lon E/W MPI halo exchange (proc_lon>1) requires "
  1036	            "mpi4jax and mpi4py."
  1037	        ) from exc
  1038	
  1039	    comm = MPI.COMM_WORLD
  1040	    sendrecv = get_sendrecv_vjp(mpi4jax)
  1041	    trailing_lat = field.shape[0]
  1042	    other = field.shape[2:]  # (nlev,) or ()
  1043	
  1044	    # PHASE-CONSTANT tags with sendtag == recvtag (one per shift
  1045	    # direction, +1 for the opposite) — the AD-safe convention from
  1046	    # plane_mpi._TAG_EW.  The shared sendrecv VJP's backward swaps
  1047	    # source<->dest but KEEPS the tags, so a rank-as-tag scheme
  1048	    # (sendtag=rank, recvtag=source) mismatches on the reverse ring at
  1049	    # proc_lon>=3 (codex review MAJOR 2026-06-13: gradients would hang).
  1050	    # With a single tag per phase, forward AND backward messages match
  1051	    # for any ring size.  (Numerically distinct from plane_mpi's
  1052	    # 1000/2000; not strictly distinct from the N/S rank-tags, but a
  1053	    # cross-match would also need the same comm + same source + same
  1054	    # dest + concurrent outstanding recvs — E/W and N/S neighbour pairs
  1055	    # are disjoint at proc_lon>1, and proc_lon==1 uses the local-wrap
  1056	    # fast path, so no cross-match path exists.)
  1057	    _TAG_LON = 3_000
  1058	
  1059	    # UNIFORM-DIRECTION SHIFT (deadlock-free on a periodic RING).  The
  1060	    # same-neighbour pattern (source==dest) that the N/S
  1061	    # ``exchange_halo_latlon`` uses only unwinds on a LINE — the poles
  1062	    # terminate the chain.  Longitude is a closed ring with no ends, and
  1063	    # the AD-safe sendrecv token serialises this rank's two exchanges,
  1064	    # so a same-neighbour west-then-east pattern makes phase-1 wait on
  1065	    # the neighbour's phase-2 → circular deadlock (observed: job
  1066	    # 8476475 timed out in mpi_sendrecv).  Instead each phase is a
  1067	    # UNIFORM shift where every send is matched by a recv IN THE SAME
  1068	    # phase (a permutation), so no cross-phase ring dependency exists.
  1069	
  1070	    # Phase 1 — EASTWARD shift: send our EAST edge to the east neighbour,
  1071	    # receive the west neighbour's east edge into our WEST halo.
  1072	    send_e = field[:, -halo:].reshape(-1)
  1073	    recv_w = sendrecv(
  1074	        send_e, jnp.zeros_like(send_e),
  1075	        west_rank,   # source (recv from west)
  1076	        east_rank,   # dest   (send to east)
  1077	        _TAG_LON,    # sendtag == recvtag (phase-constant, AD-safe)
  1078	        _TAG_LON,
  1079	        comm,
  1080	    )
  1081	    west_halo = recv_w.reshape((trailing_lat, halo) + other)
  1082	
  1083	    # Phase 2 — WESTWARD shift: send our WEST edge to the west neighbour,
  1084	    # receive the east neighbour's west edge into our EAST halo.
  1085	    send_w = field[:, :halo].reshape(-1)
  1086	    recv_e = sendrecv(
  1087	        send_w, jnp.zeros_like(send_w),
  1088	        east_rank,     # source (recv from east)
  1089	        west_rank,     # dest   (send to west)
  1090	        _TAG_LON + 1,  # phase-2 tag (sendtag == recvtag)
  1091	        _TAG_LON + 1,
  1092	        comm,
  1093	    )
  1094	    east_halo = recv_e.reshape((trailing_lat, halo) + other)
  1095	
  1096	    return jnp.concatenate([west_halo, field, east_halo], axis=1)
  1097	
  1098	
  1099	def _pad_lat_wall_2d(
  1100	    field: jax.Array,
  1101	    layout: LatLon2DLayout,
  1102	    halo: int,
  1103	    south_value: float,
  1104	    north_value: float,
  1105	) -> jax.Array:
  1106	    """Lat-axis (N/S) wall pad for a 2-D pencil row — the shared core of
  1107	    :func:`pad_halo_latlon_2d` and :func:`pad_with_pole_bc_lat_2d`.
  1108	
  1109	    Interior lat cuts MPI-sendrecv their edge row with the south / north
  1110	    neighbour (the AD-safe rank-as-tag LINE pattern — the lat axis is a
  1111	    pole-terminated line, not the periodic ring); pole-touching rows
  1112	    (``south_rank``/``north_rank is None``) fill the constant wall.
  1113	    Longitude is NOT touched.  Works for 1-D lat metrics and N-D face
  1114	    fields (trailing axes ride the reshape).  ``proc_lat == 1`` (both poles
  1115	    local) runs serially without mpi4jax.
  1116	
  1117	    Returns ``(n_lat_local + 2*halo, …)`` — same trailing shape as input.
  1118	    """
  1119	    # halo must fit the SMALLEST local lat block: a neighbour owning fewer
  1120	    # than `halo` rows would send/recv a mismatched slab and the exchange
  1121	    # would truncate/hang (same guard rationale as pad_halo_latlon_2d's
  1122	    # lon side).  _even_split gives the first blocks one extra row, so the
  1123	    # smallest block is floor(n_lat_global / proc_lat).
  1124	    min_lat_block = layout.n_lat_global // layout.proc_lat
  1125	    if halo > min_lat_block:
  1126	        raise ValueError(
  1127	            f"_pad_lat_wall_2d: halo={halo} exceeds the smallest local lat "
  1128	            f"block ({min_lat_block}=n_lat_global {layout.n_lat_global}//"
  1129	            f"proc_lat {layout.proc_lat}); a neighbour would send/recv a "
  1130	            f"mismatched halo and the MPI exchange would abort/hang.")
  1131	    trailing = field.shape[1:]
  1132	    south_is_pole = layout.south_rank is None
  1133	    north_is_pole = layout.north_rank is None
  1134	
  1135	    # Both lat ends are physical poles (proc_lat==1): pure local wall pad,
  1136	    # no MPI — keep the path runnable serially (the proc_lon==1, proc_lat==1
  1137	    # single-process equivalence test, and the proc_lon-only ring case).
  1138	    if south_is_pole and north_is_pole:
  1139	        south = jnp.full((halo,) + trailing, south_value, field.dtype)
  1140	        north = jnp.full((halo,) + trailing, north_value, field.dtype)
  1141	        return jnp.concatenate([south, field, north], axis=0)
  1142	
  1143	    import mpi4jax
  1144	    from mpi4py import MPI
  1145	
  1146	    comm = MPI.COMM_WORLD
  1147	    sendrecv = get_sendrecv_vjp(mpi4jax)
  1148	
  1149	    if south_is_pole:
  1150	        south = jnp.full((halo,) + trailing, south_value, field.dtype)
  1151	    else:
  1152	        send_s = field[:halo].reshape(-1)
  1153	        recv_s = sendrecv(
  1154	            send_s, jnp.zeros_like(send_s),
  1155	            layout.south_rank, layout.south_rank,
  1156	            layout.rank, layout.south_rank, comm)
  1157	        south = recv_s.reshape((halo,) + trailing)
  1158	
  1159	    if north_is_pole:
  1160	        north = jnp.full((halo,) + trailing, north_value, field.dtype)
  1161	    else:
  1162	        send_n = field[-halo:].reshape(-1)
  1163	        recv_n = sendrecv(
  1164	            send_n, jnp.zeros_like(send_n),
  1165	            layout.north_rank, layout.north_rank,
  1166	            layout.rank, layout.north_rank, comm)
  1167	        north = recv_n.reshape((halo,) + trailing)
  1168	
  1169	    return jnp.concatenate([south, field, north], axis=0)
  1170	
  1171	
  1172	def pad_with_pole_bc_lat_2d(
  1173	    interior: jax.Array,
  1174	    layout: LatLon2DLayout,
  1175	    halo: int = 1,
  1176	    south_value: float = 0.0,
  1177	    north_value: float = 0.0,
  1178	) -> jax.Array:
  1179	    """Lat-axis-ONLY wall pad for a 2-D pencil layout (NO longitude halo).
  1180	
  1181	    The 2-D analogue of the band
  1182	    :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat` MPI path: pad
  1183	    only the lat axis (interior cut sendrecv + pole wall constant), leaving
  1184	    longitude untouched.  The band path never split longitude, so its
  1185	    wall-BC pad is lat-only; the 2-D pencil keeps that contract and the
  1186	    operator adds lon ghosts through its own dispatched lon halo.  Routed
  1187	    here from ``halo_latlon.pad_with_pole_bc_lat`` when the active topology
  1188	    is a :class:`LatLon2DLayout` (wall poles only; the tripolar north fold /
  1189	    vector-u seam is excluded upstream).  AD-safe via the shared sendrecv
  1190	    VJP.
  1191	    """
  1192	    if halo <= 0:
  1193	        return interior
  1194	    return _pad_lat_wall_2d(interior, layout, halo, south_value, north_value)
  1195	
  1196	
  1197	def pad_halo_latlon_2d(
  1198	    field: jax.Array,
  1199	    layout: LatLon2DLayout,
  1200	    halo: int = 1,
  1201	    pole_bc: str = "wall",
  1202	    south_value: float = 0.0,
  1203	    north_value: float = 0.0,
  1204	) -> jax.Array:
  1205	    """Full 2-D halo pad (lat + lon) for ANY pencil row — the increment-3
packages/core/legoesm/parallel/latlon_mpi.py-965-        )
packages/core/legoesm/parallel/latlon_mpi.py-966-
packages/core/legoesm/parallel/latlon_mpi.py-967-    return jnp.concatenate([south_halo, field, north_halo], axis=0)
packages/core/legoesm/parallel/latlon_mpi.py-968-
packages/core/legoesm/parallel/latlon_mpi.py-969-
packages/core/legoesm/parallel/latlon_mpi.py:970:def exchange_halo_lon(
packages/core/legoesm/parallel/latlon_mpi.py-971-    field: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py-972-    west_rank: int,
packages/core/legoesm/parallel/latlon_mpi.py-973-    east_rank: int,
packages/core/legoesm/parallel/latlon_mpi.py-974-    rank: int,
packages/core/legoesm/parallel/latlon_mpi.py-975-    halo: int = 1,
--
packages/core/legoesm/parallel/latlon_mpi.py-1052-    # 1000/2000; not strictly distinct from the N/S rank-tags, but a
packages/core/legoesm/parallel/latlon_mpi.py-1053-    # cross-match would also need the same comm + same source + same
packages/core/legoesm/parallel/latlon_mpi.py-1054-    # dest + concurrent outstanding recvs — E/W and N/S neighbour pairs
packages/core/legoesm/parallel/latlon_mpi.py-1055-    # are disjoint at proc_lon>1, and proc_lon==1 uses the local-wrap
packages/core/legoesm/parallel/latlon_mpi.py-1056-    # fast path, so no cross-match path exists.)
packages/core/legoesm/parallel/latlon_mpi.py:1057:    _TAG_LON = 3_000
packages/core/legoesm/parallel/latlon_mpi.py-1058-
packages/core/legoesm/parallel/latlon_mpi.py-1059-    # UNIFORM-DIRECTION SHIFT (deadlock-free on a periodic RING).  The
packages/core/legoesm/parallel/latlon_mpi.py-1060-    # same-neighbour pattern (source==dest) that the N/S
packages/core/legoesm/parallel/latlon_mpi.py-1061-    # ``exchange_halo_latlon`` uses only unwinds on a LINE — the poles
packages/core/legoesm/parallel/latlon_mpi.py-1062-    # terminate the chain.  Longitude is a closed ring with no ends, and
--
packages/core/legoesm/parallel/latlon_mpi.py-1072-    send_e = field[:, -halo:].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py-1073-    recv_w = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1074-        send_e, jnp.zeros_like(send_e),
packages/core/legoesm/parallel/latlon_mpi.py-1075-        west_rank,   # source (recv from west)
packages/core/legoesm/parallel/latlon_mpi.py-1076-        east_rank,   # dest   (send to east)
packages/core/legoesm/parallel/latlon_mpi.py:1077:        _TAG_LON,    # sendtag == recvtag (phase-constant, AD-safe)
packages/core/legoesm/parallel/latlon_mpi.py:1078:        _TAG_LON,
packages/core/legoesm/parallel/latlon_mpi.py-1079-        comm,
packages/core/legoesm/parallel/latlon_mpi.py-1080-    )
packages/core/legoesm/parallel/latlon_mpi.py-1081-    west_halo = recv_w.reshape((trailing_lat, halo) + other)
packages/core/legoesm/parallel/latlon_mpi.py-1082-
packages/core/legoesm/parallel/latlon_mpi.py-1083-    # Phase 2 — WESTWARD shift: send our WEST edge to the west neighbour,
--
packages/core/legoesm/parallel/latlon_mpi.py-1085-    send_w = field[:, :halo].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py-1086-    recv_e = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1087-        send_w, jnp.zeros_like(send_w),
packages/core/legoesm/parallel/latlon_mpi.py-1088-        east_rank,     # source (recv from east)
packages/core/legoesm/parallel/latlon_mpi.py-1089-        west_rank,     # dest   (send to west)
packages/core/legoesm/parallel/latlon_mpi.py:1090:        _TAG_LON + 1,  # phase-2 tag (sendtag == recvtag)
packages/core/legoesm/parallel/latlon_mpi.py:1091:        _TAG_LON + 1,
packages/core/legoesm/parallel/latlon_mpi.py-1092-        comm,
packages/core/legoesm/parallel/latlon_mpi.py-1093-    )
packages/core/legoesm/parallel/latlon_mpi.py-1094-    east_halo = recv_e.reshape((trailing_lat, halo) + other)
packages/core/legoesm/parallel/latlon_mpi.py-1095-
packages/core/legoesm/parallel/latlon_mpi.py-1096-    return jnp.concatenate([west_halo, field, east_halo], axis=1)
--
packages/core/legoesm/parallel/latlon_mpi.py-1264-    # N/S — lat-axis wall pad (interior cut sendrecv at the AD-safe
packages/core/legoesm/parallel/latlon_mpi.py-1265-    # rank-as-tag LINE pattern, pole side wall); shared verbatim with
packages/core/legoesm/parallel/latlon_mpi.py-1266-    # pad_with_pole_bc_lat_2d so the two lat exchanges stay bit-identical.
packages/core/legoesm/parallel/latlon_mpi.py-1267-    ns = _pad_lat_wall_2d(field, layout, halo, south_value, north_value)
packages/core/legoesm/parallel/latlon_mpi.py-1268-    # E/W ring on the lat-padded block → fills lon ghosts + corners.
packages/core/legoesm/parallel/latlon_mpi.py:1269:    return exchange_halo_lon(
packages/core/legoesm/parallel/latlon_mpi.py-1270-        ns, layout.west_rank, layout.east_rank, layout.rank, halo=halo)
packages/core/legoesm/parallel/latlon_mpi.py-1271-
packages/core/legoesm/parallel/latlon_mpi.py-1272-
packages/core/legoesm/parallel/latlon_mpi.py-1273-# ============================================================================
packages/core/legoesm/parallel/latlon_mpi.py-1274-# Backend-dispatched pad_halo_latlon implementation
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-304-        topology = get_mpi_topology()
packages/core/legoesm/grids/operators_latlon_cgrid.py-305-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/operators_latlon_cgrid.py-306-            LatLon2DLayout, exchange_halo_lon,
packages/core/legoesm/grids/operators_latlon_cgrid.py-307-        )
packages/core/legoesm/grids/operators_latlon_cgrid.py-308-        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/operators_latlon_cgrid.py:309:            return exchange_halo_lon(
packages/core/legoesm/grids/operators_latlon_cgrid.py-310-                f, topology.west_rank, topology.east_rank,
packages/core/legoesm/grids/operators_latlon_cgrid.py-311-                topology.rank, halo=halo,
packages/core/legoesm/grids/operators_latlon_cgrid.py-312-            )
packages/core/legoesm/grids/operators_latlon_cgrid.py-313-    elif backend == "spmd":
packages/core/legoesm/grids/operators_latlon_cgrid.py-314-        mesh = get_spmd_mesh()
     1	name: MPI Distributed
     2	
     3	on:
     4	  pull_request:
     5	  push:
     6	    branches:
     7	      - main
     8	  workflow_dispatch:
     9	
    10	jobs:
    11	  mpi-distributed:
    12	    runs-on: ubuntu-latest
    13	    timeout-minutes: 60
    14	
    15	    steps:
    16	      - name: Checkout
    17	        uses: actions/checkout@v4
    18	
    19	      - name: Set Up Python
    20	        uses: actions/setup-python@v5
    21	        with:
    22	          python-version: "3.12"
    23	
    24	      - name: Install OpenMPI
    25	        run: |
    26	          sudo apt-get update
    27	          sudo apt-get install -y --no-install-recommends openmpi-bin libopenmpi-dev
    28	          mpirun --version
    29	
    30	      - name: Install Python Dependencies
    31	        # Deterministic tested MPI stack (jax 0.9.2 + mpi4jax 0.8.1.post2) —
    32	        # the envelope enforced by parallel/reductions.py
    33	        # _validate_mpi_runtime_versions. A bare ".[dev,mpi]" install resolves
    34	        # the latest jax (0.10+), which removed CustomCallV1 and breaks
    35	        # mpi4jax 0.8/0.9 at runtime; this job previously passed only by
    36	        # resolver luck. requirements_mpi.txt supplies pinned mpi4py/mpi4jax.
    37	        run: |
    38	          python -m pip install --upgrade pip
    39	          pip install -e ".[dev]"
    40	          pip install -r requirements_mpi.txt
    41	
    42	      - name: Run MPI Distributed Halo Tests (np=2,3,6)
    43	        env:
    44	          # mpi4jax 0.8's own version self-check predates jax 0.9 and warns
    45	          # even inside legoESM's tested envelope; legoESM's runtime guard
    46	          # (parallel/reductions.py, NOT suppressed) is the authoritative check.
    47	          MPI4JAX_NO_WARN_JAX_VERSION: "1"
    48	        run: |
    49	          for n in 2 3 6; do
    50	            echo "=== MPI distributed test, np=${n} ==="
    51	            mpirun --oversubscribe -np "${n}" \
    52	              python -m pytest -q tests/distributed/test_halo_mpi.py
    53	          done
    54	
    55	      - name: Run Plane Pencil MPI Tests (np=2,4)
    56	        env:
    57	          MPI4JAX_NO_WARN_JAX_VERSION: "1"
    58	        run: |
    59	          for n in 2 4; do
    60	            echo "=== Plane pencil MPI test, np=${n} ==="
    61	            mpirun --oversubscribe -np "${n}" \
    62	              python -m pytest -q tests/distributed/test_plane_pencil_mpi.py
    63	          done
    64	
    65	      - name: Run Lat-Lon 2-D Pad (wall) MPI Tests (np=2,3,6)
    66	        # np=3 -> 1x3 drives the E/W periodic ring through proc_lon=3 in
    67	        # REVERSE (the phase-constant-tag VJP case); np=6 -> 2x3 drives the
    68	        # N/S reverse sendrecv AND the proc_lon=3 reverse ring together.
    69	        env:
    70	          MPI4JAX_NO_WARN_JAX_VERSION: "1"
    71	        run: |
    72	          for n in 2 3 6; do
    73	            echo "=== Lat-lon 2-D wall-pad MPI test, np=${n} ==="
    74	            mpirun --oversubscribe -np "${n}" \
    75	              python -m pytest -q tests/distributed/test_latlon_2d_pad_wall_mpi.py
    76	          done
    77	
    78	      - name: Run Lat-Lon transpose AD (lon_gather_full VJP) MPI Tests (np=2,3,6)
    79	        # Adjoint (dot-product identity) + grad-finiteness of the lat-pencil
    80	        # transpose custom VJP — the AD-safe gather for lon-split pole-fold /
    81	        # polar-filter.  proc_lon>=2 at every size gives a real lon ring.
    82	        env:
    83	          MPI4JAX_NO_WARN_JAX_VERSION: "1"
    84	        run: |
    85	          for n in 2 3 6; do
    86	            echo "=== Lat-lon transpose-AD MPI test, np=${n} ==="
    87	            mpirun --oversubscribe -np "${n}" \
    88	              python -m pytest -q tests/distributed/test_latlon_transpose_ad_mpi.py
    89	          done
     1	"""Serial (single-process) lane for the lat-lon 2-D pencil HALO DISPATCH.
     2	
     3	The 2-D pad PRIMITIVES are tested in ``test_latlon_2d_pad_wall.py`` (serial)
     4	and ``tests/distributed/test_latlon_2d_pad_wall_mpi.py`` (MPI parity + AD).
     5	This file tests the BACKEND DISPATCH layer that wires those primitives into
     6	the operators: when the global halo backend is ``"mpi"`` with a
     7	:class:`LatLon2DLayout` active, the public ``halo_latlon`` entry points
     8	(``pad_halo_latlon`` & friends, ``pad_with_pole_bc_lat[_multi]``,
     9	``zero_polar_lat_ends``) MUST route to the 2-D wall-pole path — not silently
    10	fall through to the serial pole-fold (the riskiest 2-D bug).
    11	
    12	A ``proc=1×1`` layout makes the 2-D path fully LOCAL (both poles local +
    13	single-member lon ring => no mpi4jax), so the routing is checkable without
    14	``mpirun``.  The genuine 2×N distributed routing + AD is covered in
    15	``tests/distributed/test_latlon_2d_dispatch_mpi.py``.
    16	
    17	Also pins the loud guards: ``make_latlon_2d_mpi_step`` WIRES a longitude
    18	split (proc_lon>1) for regular / wall-pole grids (operator lon ops route
    19	through the dispatched lon halo; the polar filter through the lat-pencil
    20	transpose) but refuses a TRIPOLAR grid and, with ``use_polar_filter=True``,
    21	an UNEVEN lon split (the allgather needs equal blocks) — before any
    22	mutation.  ``pad_with_pole_bc_lat`` refuses the tripolar fold seam on a
    23	2-D layout (wall-pole benchmark only).
    24	"""
    25	from __future__ import annotations
    26	
    27	import jax
    28	
    29	jax.config.update("jax_enable_x64", True)
    30	
    31	import jax.numpy as jnp
    32	import numpy as np
    33	import pytest
    34	
    35	from legoesm.grids.halo import set_halo_backend
    36	from legoesm.grids.halo_latlon import (
    37	    pad_halo_latlon,
    38	    pad_halo_latlon_3d,
    39	    pad_halo_latlon_3d_local,
    40	    pad_halo_latlon_local,
    41	    pad_halo_latlon_vector,
    42	    pad_halo_latlon_vector_3d,
    43	    pad_halo_latlon_vector_3d_local,
    44	    pad_halo_latlon_vector_local,
    45	    pad_with_pole_bc_lat,
    46	    pad_with_pole_bc_lat_multi,
    47	    zero_polar_lat_ends,
    48	)
    49	from legoesm.parallel.latlon_mpi import (
    50	    make_latlon_2d_layout,
    51	    make_latlon_2d_mpi_step,
    52	    pad_with_pole_bc_lat_2d,
    53	)
    54	
    55	SV, NV = -1.0, -2.0   # distinct non-zero walls to catch S/N confusion
    56	N_LAT, N_LON = 4, 6
    57	
    58	
    59	@pytest.fixture()
    60	def layout_1x1():
    61	    """1×1 pencil: both poles local, single-member lon ring => the 2-D
    62	    path runs serially (no mpi4jax)."""
    63	    return make_latlon_2d_layout(0, 1, 1, N_LAT, N_LON)
    64	
    65	
    66	@pytest.fixture(autouse=True)
    67	def _arm_2d_mpi_backend(layout_1x1):
    68	    """Arm the MPI halo backend with the 2-D layout, restore after.
    69	
    70	    ``set_halo_backend`` only stores the topology (no mpi4py), and the
    71	    1×1 2-D pad is fully local, so the operators' dispatch fires the 2-D
    72	    branch without any collective."""
    73	    set_halo_backend("mpi", layout_1x1)
    74	    yield
    75	    set_halo_backend("local")
    76	
    77	
    78	def test_scalar_pad_folds_at_proc_lon1(layout_1x1):
    79	    """``pad_halo_latlon`` (scalar fold family) at proc_lon==1 must reuse the
    80	    band POLE-FOLD (lon full per rank => local 180-deg fold), NOT a wall-zero
    81	    ghost — this is what keeps the global 2-D dispatch safe for every scalar
    82	    caller.  At 1×1 it equals the serial local fold bit-for-bit."""
    83	    g = jnp.asarray(np.arange(N_LAT * N_LON).reshape(N_LAT, N_LON).astype(float))
    84	    out = pad_halo_latlon(g, halo=1)
    85	    np.testing.assert_array_equal(
    86	        np.asarray(out), np.asarray(pad_halo_latlon_local(g, 1)))
    87	
    88	
    89	def test_vector_pad_folds_at_proc_lon1(layout_1x1):
    90	    """Meridional-vector fold-family pad reuses the band fold WITH the
    91	    sign-flip at proc_lon==1 (== serial vector local fold)."""
    92	    g = jnp.asarray(np.linspace(-1, 1, N_LAT * N_LON).reshape(N_LAT, N_LON))
    93	    out = pad_halo_latlon_vector(g, halo=1)
    94	    np.testing.assert_array_equal(
    95	        np.asarray(out), np.asarray(pad_halo_latlon_vector_local(g, 1)))
    96	
    97	
    98	def test_scalar_and_vector_3d_pad_fold_at_proc_lon1(layout_1x1):
    99	    """3-D (level axis) scalar + vector fold-family pads reuse the band fold
   100	    at proc_lon==1 (== the serial 3-D local fold)."""
   101	    g = jnp.asarray(
   102	        np.arange(N_LAT * N_LON * 3).reshape(N_LAT, N_LON, 3).astype(float))
   103	    np.testing.assert_array_equal(
   104	        np.asarray(pad_halo_latlon_3d(g, halo=1)),
   105	        np.asarray(pad_halo_latlon_3d_local(g, 1)))
   106	    np.testing.assert_array_equal(
   107	        np.asarray(pad_halo_latlon_vector_3d(g, halo=1)),
   108	        np.asarray(pad_halo_latlon_vector_3d_local(g, 1)))
   109	
   110	
   111	def test_pad_with_pole_bc_lat_routes_to_2d_latonly(layout_1x1):
   112	    """``pad_with_pole_bc_lat`` routes to the lat-ONLY 2-D wall pad
   113	    (longitude untouched), equal to a plain lat-axis constant pad at 1×1."""
   114	    g = jnp.asarray(np.arange(N_LAT * N_LON).reshape(N_LAT, N_LON).astype(float))
   115	    out = pad_with_pole_bc_lat(g, halo=1, south_value=SV, north_value=NV)
   116	    want = pad_with_pole_bc_lat_2d(
   117	        g, layout_1x1, halo=1, south_value=SV, north_value=NV)
   118	    np.testing.assert_array_equal(np.asarray(out), np.asarray(want))
   119	    # lat-only: lon width is unchanged, lat grew by 2*halo.
   120	    assert out.shape == (N_LAT + 2, N_LON)
   121	    # equal to a plain lat-axis constant pad (the local-backend semantics).
   122	    plain = jnp.pad(g, ((1, 1), (0, 0)),
   123	                    constant_values=((SV, NV), (0, 0)))
   124	    np.testing.assert_array_equal(np.asarray(out), np.asarray(plain))
   125	
   126	
   127	def test_pad_with_pole_bc_lat_1d_metric_routes_to_2d(layout_1x1):
   128	    """A 1-D lat metric (sin_lat etc.) also routes to the lat-only 2-D
   129	    wall pad (trailing-shape-agnostic reshape)."""
   130	    g = jnp.asarray(np.linspace(-1.0, 1.0, N_LAT))
   131	    out = pad_with_pole_bc_lat(g, halo=1, south_value=SV, north_value=NV)
   132	    np.testing.assert_array_equal(
   133	        np.asarray(out),
   134	        np.asarray(jnp.concatenate([jnp.array([SV]), g, jnp.array([NV])])))
   135	
   136	
   137	def test_pad_with_pole_bc_lat_multi_routes_per_field(layout_1x1):
   138	    """``pad_with_pole_bc_lat_multi`` on a 2-D layout routes through the
   139	    per-field path (the fused path is band-only) => each field takes the
   140	    lat-only 2-D wall pad."""
   141	    f1 = jnp.asarray(np.arange(N_LAT * N_LON).reshape(N_LAT, N_LON).astype(float))
   142	    f2 = f1 + 100.0
   143	    outs = pad_with_pole_bc_lat_multi(
   144	        (f1, f2), halo=1, south_values=(SV, 0.0), north_values=(NV, 0.0))
   145	    assert len(outs) == 2
   146	    np.testing.assert_array_equal(
   147	        np.asarray(outs[0]),
   148	        np.asarray(pad_with_pole_bc_lat_2d(
   149	            f1, layout_1x1, halo=1, south_value=SV, north_value=NV)))
   150	    np.testing.assert_array_equal(
   151	        np.asarray(outs[1]),
   152	        np.asarray(pad_with_pole_bc_lat_2d(
   153	            f2, layout_1x1, halo=1, south_value=0.0, north_value=0.0)))
   154	
   155	
   156	def test_zero_polar_lat_ends_2d_zeros_both_poles(layout_1x1):
   157	    """At 1×1 both lat ends are physical poles, so ``zero_polar_lat_ends``
   158	    must zero index 0 and -1 (the 2-D pole-touch test == band test)."""
   159	    field = jnp.asarray(
   160	        1.0 + np.arange(N_LAT * N_LON).reshape(N_LAT, N_LON).astype(float))
   161	    out = np.asarray(zero_polar_lat_ends(field))
   162	    assert np.all(out[0] == 0.0) and np.all(out[-1] == 0.0)
   163	    np.testing.assert_array_equal(out[1:-1], np.asarray(field)[1:-1])
   164	
   165	
   166	def test_step_refuses_tripolar_longitude_split():
   167	    """``make_latlon_2d_mpi_step`` must refuse proc_lon>1 on a TRIPOLAR grid
   168	    LOUDLY — the curl tripolar-cap fold keeps a local lon roll (the 180° fold
   169	    under a lon split needs the lat-pencil transpose).  Keyed on
   170	    ``fold.is_active`` ALONE (grid-global, so all ranks raise together — NOT
   171	    ``fold_j``, which is rank-local and would deadlock).  Regular / wall-pole
   172	    proc_lon>1 is now WIRED (validated by the 2-D dycore conservation gate
   173	    ``tests/distributed/test_latlon_2d_mpi_step.py``)."""
   174	    from types import SimpleNamespace
   175	    layout_1x2 = make_latlon_2d_layout(0, 1, 2, N_LAT, N_LON)
   176	    # fold active, NO fold_j attr -> the is_active-alone guard must still fire.
   177	    tri = SimpleNamespace(
   178	        grid=SimpleNamespace(fold=SimpleNamespace(is_active=True)),
   179	        config=SimpleNamespace(use_polar_filter=False))
   180	    with pytest.raises(NotImplementedError, match="TRIPOLAR"):
   181	        make_latlon_2d_mpi_step(tri, layout_1x2)
   182	
   183	
   184	def test_step_refuses_polar_filter_uneven_longitude_split():
   185	    """``make_latlon_2d_mpi_step`` with ``use_polar_filter=True`` now WIRES the
   186	    lon-split filter (via the lat-pencil transpose), but the AD-safe allgather
   187	    needs an EQUAL split — so an UNEVEN proc_lon (n_lon % proc_lon != 0) must
   188	    refuse at the FACTORY, BEFORE any mutation (the halo backend must stay
   189	    'local').  N_LON=6, proc_lon=4 -> 6%4=2 != 0.  (Even splits are exercised
   190	    end-to-end in tests/distributed/test_latlon_2d_polar_filter_mpi.py.)"""
   191	    from types import SimpleNamespace
   192	
   193	    from legoesm.grids.halo import get_halo_backend, set_halo_backend
   194	    set_halo_backend("local")
   195	    layout_1x4 = make_latlon_2d_layout(0, 1, 4, N_LAT, N_LON)
   196	    m = SimpleNamespace(
   197	        grid=SimpleNamespace(fold=None),
   198	        config=SimpleNamespace(use_polar_filter=True))
   199	    with pytest.raises(NotImplementedError, match="EQUAL split"):
   200	        make_latlon_2d_mpi_step(m, layout_1x4)
   201	    # No partial mutation: the guard fired before set_halo_backend.
   202	    assert get_halo_backend() == "local"
   203	
   204	
   205	def test_pad_with_pole_bc_lat_refuses_fold_seam_on_2d(layout_1x1):
   850	        from legoesm.parallel.latlon_mpi import (
   851	            LatLon2DLayout,
   852	            LatLonBandLayout,
   853	        )
   854	        topology = get_mpi_topology()
   855	        if isinstance(topology, (LatLonBandLayout, LatLon2DLayout)):
   856	            return topology.south_rank is None, topology.north_rank is None
   857	    return True, True
   858	
   859	
   860	def _clamp_pole_pad_rows(x_ext: jnp.ndarray, halo: int,
   861	                         south_is_pole, north_is_pole) -> jnp.ndarray:
   862	    """Overwrite beyond-pole pad rows with the nearest PHYSICAL row.
   863	
   864	    The wide exchange fills pole-side pad rows with the wall constant (0).
   865	    Metric arrays (area, dx, dy) must stay FINITE and non-zero there — the
   866	    interior operators divide by them, and ``0/0 → NaN`` would leak through
   867	    a masked row into the owned region via ``min``/flux stencils (NaN is not
   868	    absorbed by ``* mask``).  Values are irrelevant (the rows are permanently
   869	    land-masked); finiteness is the contract.
   870	
   871	    Uniform ``jnp.where`` select so the SPMD (traced pole flags) and the
   872	    static local/MPI cases share one code path.
   873	    """
   874	    n_ext = x_ext.shape[0]
   875	    idx = jnp.arange(n_ext)
   876	    shape_tail = (1,) * (x_ext.ndim - 1)
   877	    south_sel = ((idx < halo).reshape((n_ext,) + shape_tail)
   878	                 & jnp.asarray(south_is_pole))
   879	    north_sel = ((idx >= n_ext - halo).reshape((n_ext,) + shape_tail)
   880	                 & jnp.asarray(north_is_pole))
   881	    south_row = x_ext[halo][None]
   882	    north_row = x_ext[n_ext - halo - 1][None]
   883	    out = jnp.where(south_sel, south_row, x_ext)
   884	    return jnp.where(north_sel, north_row, out)
   885	
   886	
   887	def widen_band_cell_fields(fields, halo: int, *, clamp_poles: bool = False):
   888	    """Widen cell-row (leading dim ``n_lat``) fields by ``halo`` rows/side.
   889	
   890	    ONE fused exchange under the MPI band backend
   891	    (:func:`pad_with_pole_bc_lat_multi`); wall-zero at physical poles,
   892	    optionally clamped to the nearest physical row (metric arrays — see
   893	    :func:`_clamp_pole_pad_rows`).  Applies to T-point AND u-point fields
   894	    (both carry one row per cell).
   895	    """
   896	    padded = pad_with_pole_bc_lat_multi(fields, halo=halo)
   897	    if not clamp_poles:
   898	        return padded
   899	    south_is_pole, north_is_pole = band_pole_flags()
   900	    return tuple(
   901	        _clamp_pole_pad_rows(p, halo, south_is_pole, north_is_pole)
   902	        for p in padded
   903	    )
   904	
   905	
   906	def widen_band_vface_fields(fields, halo: int, *, clamp_poles: bool = False):
   907	    """Widen v-face (leading dim ``n_lat+1``) fields by ``halo`` rows/side.
   908	
   909	    A band's v array carries one duplicated boundary face, so faces cannot
   910	    be exchanged like cell rows.  Trick: drop the top face — ``v[:-1]`` is
   911	    exactly one "south face" per cell row — exchange THAT as a cell field
   912	    with ``halo+1`` rows, and re-slice: ``padded[1:]`` are the south faces
   913	    of the ``n+2*halo`` extended cells plus the top face of the northmost
   914	    extended cell, i.e. the ``(n + 2*halo) + 1`` faces of the extended
   915	    band.  Value-identical to the serial extended domain; at physical poles
   916	    the wall constant fills beyond-pole faces (repaired to the nearest
   917	    physical face for metric arrays via ``clamp_poles``).
   918	
   919	    The band's OWN TOP face (a value the ``[:-1]`` drop discarded) is
   920	    restored verbatim at extended index ``halo + n``: on an interior band
   921	    the exchange already delivered the neighbour's bit-equal copy of that
   922	    shared face, but on the NORTH-POLE band the exchange fills it with the
   923	    wall constant — zeroing a real metric row (``dy_v``/``area_q``/``f_v``
   924	    at the pole face), which poisons every division at that face.  The
   925	    restore is unconditional (bit-neutral on interior bands), so it stays
   926	    uniform under SPMD.
   927	    """
   928	    interiors = tuple(f[:-1] for f in fields)
   929	    padded = pad_with_pole_bc_lat_multi(interiors, halo=halo + 1)
   930	    out = tuple(
   931	        p[1:].at[halo + f.shape[0] - 1].set(f[-1])
   932	        for p, f in zip(padded, fields)
   933	    )
   934	    if not clamp_poles:
   935	        return out
   936	    south_is_pole, north_is_pole = band_pole_flags()
   937	    return tuple(
   938	        _clamp_pole_pad_rows(p, halo, south_is_pole, north_is_pole)
   939	        for p in out
   940	    )
   941	
   942	
   943	def widen_cgrid_geometry_band(geom, halo: int):
   944	    """Extended-band twin of ``slice_cgrid_geometry_to_band``: widen a
   945	    band-local :class:`~legoesm.grids.latlon.LatLonCGridGeometry` by
   946	    ``halo`` ghost rows per side via the ACTIVE halo backend.
   947	
   948	    Field rules mirror the slicer (stagger-aware):
   949	
   950	    * T-/u-point metrics + 1-D lat arrays → cell-row widening;
   951	    * v-/q-point metrics → v-face widening;
   952	    * ``lon``/``dlon``/``dlat``/``radius``/``total_area`` unchanged
   953	      (``total_area`` stays the GLOBAL denominator);
   954	    * ``n_lat`` → ``n_lat + 2*halo`` (static Python int);
   955	    * ``fold`` must be inactive/non-local — the wide-halo path refuses
   956	      tripolar folds (caller-validated; this helper asserts).
   957	
   958	    ALL metric rows are pole-clamped (finite beyond-pole values): the
   959	    extended rows are permanently land-masked, so their values never enter
   960	    the owned region, but a zero metric would create ``0/0 = NaN`` that
   961	    masks do NOT absorb.
   962	
   963	    Traced (per-step) op: two fused exchanges (cell group + v-face group).
   964	    """
   965	    # Refuse on an ACTIVE fold (``is_active`` is rank-CONSISTENT on sliced
    35	        the neighbour rank so the gradient / divergence stencil sees
    36	        continuous data across the cut.  Routes through
    37	        :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat` which
    38	        in turn delegates to
    39	        :func:`legoesm.parallel.latlon_mpi.pad_with_pole_bc_lat_mpi`.
    40	
    41	    Parameters
    42	    ----------
    43	    interior : (..., n_interior, n_lon, ...) — missing the first and last
    44	        rows along the latitude axis (axis 0 for 2D/3D fields).
    45	
    46	    Returns
    47	    -------
    48	    padded : (..., n_interior+2, n_lon, ...)
    49	    """
    50	    # Deferred import to avoid cycles — ocean.dynamics is imported by
    51	    # many grid-related modules.
    52	    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
    53	    return pad_with_pole_bc_lat(
    54	        interior, halo=1, south_value=0.0, north_value=0.0,
    55	    )
    56	
    57	
    58	def pad_ns_zero_multi(*fields: jnp.ndarray) -> tuple:
    59	    """Batched :func:`pad_ns_zero` for independent same-``n_lat`` fields.
    60	
    61	    Value-identical to ``tuple(pad_ns_zero(f) for f in fields)``; under
    62	    the MPI lat-lon band backend the interior partition cuts are
    63	    exchanged in ONE fused sendrecv pair per cut per dtype group instead
    64	    of one pair per field (see
    65	    :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat_multi`).
    66	    Use for clusters of pads at the same dataflow level (no data
    67	    dependency between the fields).
    68	    """
    69	    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat_multi
    70	    return pad_with_pole_bc_lat_multi(fields, halo=1)
    71	
    72	
    73	def _vface_cos_lat_core(grid) -> jnp.ndarray:
    74	    """Raw ``cos(lat_v)`` at the ``n_lat+1`` v-face midpoints (#515 consolidation).
    75	
    76	    The single source for the regular-branch v-face zonal-metric cosine shared by
    77	    ``divergence_cgrid`` and ``gradient_curl_to_v``: pad ``grid.lat`` with the
    78	    halo-aware pole/cut BC (``pad_with_pole_bc_lat`` — at an interior MPI band cut
    79	    the ghost row is the neighbour rank's true edge latitude via the AD-safe
    80	    sendrecv), take the v-face midpoint latitude, return its cosine.  Callers apply
    81	    their OWN pole step (``zero_polar_lat_ends`` vs a ``1e-30`` floor) and the
    82	    ``R*dlon`` scaling.  Kept on the STORED ``grid.lat`` dtype (NOT
    83	    ``result_type``-cast — unlike the ocean ``vface_zonal_cos_lat``) so the core
    84	    path stays byte-identical to the former inline recompute.
    85	    """
    86	    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
    87	    lat_pad = pad_with_pole_bc_lat(
    88	        grid.lat, halo=1, south_value=0.0, north_value=0.0,
    89	    )
    90	    lat_v = 0.5 * (lat_pad[:-1] + lat_pad[1:])  # (n_lat+1,)

codex
The lat-only contract and smallest-block guard match the existing 2-D primitive. I’m now checking the two places most likely to hide a production-only failure: cross-rank schema assumptions in dtype grouping, and whether the new MPI test is actually collected with 64-bit dtype grouping enabled.
exec
/usr/bin/bash -lc "nl -ba .github/workflows/mpi-distributed.yml | sed -n '42,100p'; rg -n -C 4 \"jax_enable_x64|JAX_ENABLE_X64|testpaths|tests/distributed|pytest\" pyproject.toml pytest.ini setup.cfg .github scripts | head -500; rg -n -C 5 \"def get_sendrecv_vjp|sendrecv_vjp|custom_vjp\" packages/core/legoesm/parallel/halo_exchange.py packages/core/legoesm | head -420; rg -n -C 3 \"LatLon2DLayout|set_halo_backend\\(\\\"mpi\\\"|pad_ns_zero_multi|widen_band_cell_fields|widen_cgrid_geometry_band\" packages/core/legoesm | head -800" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 831ms:
    42	      - name: Run MPI Distributed Halo Tests (np=2,3,6)
    43	        env:
    44	          # mpi4jax 0.8's own version self-check predates jax 0.9 and warns
    45	          # even inside legoESM's tested envelope; legoESM's runtime guard
    46	          # (parallel/reductions.py, NOT suppressed) is the authoritative check.
    47	          MPI4JAX_NO_WARN_JAX_VERSION: "1"
    48	        run: |
    49	          for n in 2 3 6; do
    50	            echo "=== MPI distributed test, np=${n} ==="
    51	            mpirun --oversubscribe -np "${n}" \
    52	              python -m pytest -q tests/distributed/test_halo_mpi.py
    53	          done
    54	
    55	      - name: Run Plane Pencil MPI Tests (np=2,4)
    56	        env:
    57	          MPI4JAX_NO_WARN_JAX_VERSION: "1"
    58	        run: |
    59	          for n in 2 4; do
    60	            echo "=== Plane pencil MPI test, np=${n} ==="
    61	            mpirun --oversubscribe -np "${n}" \
    62	              python -m pytest -q tests/distributed/test_plane_pencil_mpi.py
    63	          done
    64	
    65	      - name: Run Lat-Lon 2-D Pad (wall) MPI Tests (np=2,3,6)
    66	        # np=3 -> 1x3 drives the E/W periodic ring through proc_lon=3 in
    67	        # REVERSE (the phase-constant-tag VJP case); np=6 -> 2x3 drives the
    68	        # N/S reverse sendrecv AND the proc_lon=3 reverse ring together.
    69	        env:
    70	          MPI4JAX_NO_WARN_JAX_VERSION: "1"
    71	        run: |
    72	          for n in 2 3 6; do
    73	            echo "=== Lat-lon 2-D wall-pad MPI test, np=${n} ==="
    74	            mpirun --oversubscribe -np "${n}" \
    75	              python -m pytest -q tests/distributed/test_latlon_2d_pad_wall_mpi.py
    76	          done
    77	
    78	      - name: Run Lat-Lon transpose AD (lon_gather_full VJP) MPI Tests (np=2,3,6)
    79	        # Adjoint (dot-product identity) + grad-finiteness of the lat-pencil
    80	        # transpose custom VJP — the AD-safe gather for lon-split pole-fold /
    81	        # polar-filter.  proc_lon>=2 at every size gives a real lon ring.
    82	        env:
    83	          MPI4JAX_NO_WARN_JAX_VERSION: "1"
    84	        run: |
    85	          for n in 2 3 6; do
    86	            echo "=== Lat-lon transpose-AD MPI test, np=${n} ==="
    87	            mpirun --oversubscribe -np "${n}" \
    88	              python -m pytest -q tests/distributed/test_latlon_transpose_ad_mpi.py
    89	          done
rg: pytest.ini: No such file or directory (os error 2)
rg: setup.cfg: No such file or directory (os error 2)
pyproject.toml-85-    "fsspec>=2024.2",
pyproject.toml-86-]
pyproject.toml-87-# Apple-Silicon local-GPU backend via MLX (PJRT plugin). Select with
pyproject.toml-88-# ``JAX_PLATFORMS=mps``.
pyproject.toml:89:# float32 ONLY — MLX has no float64, so ``JAX_ENABLE_X64=1`` arrays raise and
pyproject.toml-90-# complex128 downcasts to complex64; spectral/x64/conservation-grade and any
pyproject.toml-91-# mpi4jax-collective run must stay on ``JAX_PLATFORMS=cpu``. Single MpsDevice:
pyproject.toml-92-# no SPMD/MPI multi-device. Good for single-device float32 FV dycore / ML work.
pyproject.toml-93-mps = ["jax-mps"]
--
pyproject.toml-104-    "mpi4py>=4.1,<5",
pyproject.toml-105-    "mpi4jax>=0.8,<0.10",
pyproject.toml-106-]
pyproject.toml-107-dev = [
pyproject.toml:108:    "pytest>=8.0",
pyproject.toml:109:    "pytest-xdist",
pyproject.toml:110:    "pytest-timeout",
pyproject.toml-111-    "ruff",
pyproject.toml-112-    "mypy",
pyproject.toml-113-    "pre-commit",
pyproject.toml-114-    "import-linter>=2.0",
--
pyproject.toml-203-"packages/atmosphere/legoesm/atmosphere/dynamics/column_les_diagnosis.py" = ["N803", "N806", "N815"]
pyproject.toml-204-"packages/atmosphere/legoesm/atmosphere/dynamics/column_les.py" = ["N803", "N806", "N815"]
pyproject.toml-205-# Column large-scale extractor: physical symbol names (T, T_v for temperature /
pyproject.toml-206-# virtual temperature) read clearer than snake_case; same rationale. Test files
pyproject.toml:207:# use the same symbols (T) + set jax_enable_x64 before importing (E402).
pyproject.toml-208-"packages/atmosphere/legoesm/atmosphere/dynamics/column_large_scale_extract.py" = ["N803", "N806"]
pyproject.toml-209-"tests/atmosphere/test_column_large_scale_extract*.py" = ["E402", "N806"]
pyproject.toml-210-# Compare-to-reanalysis bridge + its tests: physical symbol names (T for
pyproject.toml-211-# temperature, T_v for virtual temperature) + CLAUDE.md-mandated unit suffixes
--
pyproject.toml-253-# clearer than snake_case. Same rationale as the column-extractor tests above.
pyproject.toml-254-"tests/atmosphere/hydrostatic/unit/test_semi_implicit_gw_stability.py" = ["N803", "N806"]
pyproject.toml-255-"tests/unit/test_check_cross_grid_deploy.py" = ["N803"]
pyproject.toml-256-# NEMO Roquet EOS-80 test: physical symbols ``T``/``S`` as args + locals, and
pyproject.toml:257:# ``jax_enable_x64`` set before the module import (E402) -- same rationale as the
pyproject.toml-258-# scientific-kernel test entries above.
pyproject.toml-259-"tests/ocean/unit/test_nemo_roquet_eos.py" = ["E402", "N803", "N806"]
pyproject.toml-260-# Run-as-a-script sys.path bootstrap (iter 321) must precede the ``scripts.*`` import, so the
pyproject.toml-261-# subsequent module-level imports trip E402; the bootstrap makes ``python scripts/...`` work.
--
pyproject.toml-266-"packages/ml/legoesm/training/era5_to_state.py" = ["E402", "N803", "N806", "N815"]
pyproject.toml-267-"tests/unit/test_era5_mpas_carry.py" = ["N806"]
pyproject.toml-268-# CLUBB port tests use the same physical symbol names as the modules under test
pyproject.toml-269-# (T for temperature, Tc for deg C, etc.) — clearer than snake_case here. E402:
pyproject.toml:270:# these tests set jax_enable_x64 before importing the modules under test (a
pyproject.toml-271-# standard JAX-testing idiom), so module-level imports follow that call.
pyproject.toml-272-"tests/unit/test_clubb_*.py" = ["E402", "N802", "N803", "N806"]
pyproject.toml:273:# Same idiom: sets jax_enable_x64 before importing the module under test (real
pyproject.toml-274-# fp64 needed for the #618 fp32-vs-fp64 saturation comparison) and uses the
pyproject.toml-275-# physical symbol T for temperature.  I001: the config call sits between imports.
pyproject.toml-276-"tests/unit/test_warm_rain_fp32_saturation.py" = ["E402", "I001", "N803"]
pyproject.toml-277-# CLUBB validation/characterization scripts use the same physical symbol names
pyproject.toml:278:# (T for temperature) and set jax_enable_x64 before importing (E402).
pyproject.toml-279-"scripts/validate/clubb_*.py" = ["E402", "N802", "N803", "N806"]
pyproject.toml:280:# Cubed-sphere MPI face-scatter tests set jax_enable_x64 before importing the
pyproject.toml-281-# dycore (fp64 replicated-vs-scattered bit-equivalence), so imports follow.
pyproject.toml-282-"tests/parallel/test_cube_face_scatter.py" = ["E402"]
pyproject.toml:283:"tests/distributed/test_cube_face_scatter_mpi.py" = ["E402"]
pyproject.toml-284-# MPAS native-SPMD parity gates use the MPAS canonical symbol names (nCells,
pyproject.toml-285-# nEdges, W) plus the mandated T_sfc / dT unit-suffixed physics names; same
pyproject.toml-286-# rationale as the column-extractor tests above.
pyproject.toml-287-"tests/parallel/test_mpas_atm_native_step.py" = ["N806"]
--
pyproject.toml-299-# Vendored 3rd-party CLM-ML-JAX backend (BSD-3) — not type-annotated to legoESM's
pyproject.toml-300-# standard; excluded from type-checking here (audited upstream).
pyproject.toml-301-exclude = ['packages/land/legoesm/land/canopy/clm_ml_backend/']
pyproject.toml-302-
pyproject.toml:303:[tool.pytest.ini_options]
pyproject.toml:304:testpaths = ["tests"]
pyproject.toml-305-addopts = "-v --tb=short -m 'not slow'"
pyproject.toml-306-markers = [
pyproject.toml-307-    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
pyproject.toml-308-    # Complexity-tier ladder (docs/validation/TESTING.md). Opt-in selectors: run a rung
--
scripts/run/run_rce_convection_sweep.py-71-  conservation budget over one step"; no assertions).
scripts/run/run_rce_convection_sweep.py-72-
scripts/run/run_rce_convection_sweep.py-73-Usage
scripts/run/run_rce_convection_sweep.py-74------
scripts/run/run_rce_convection_sweep.py:75:    JAX_ENABLE_X64=1 python scripts/run_rce_convection_sweep.py \\
scripts/run/run_rce_convection_sweep.py-76-        --schemes all --days 5 --N 12 --nlev 20
scripts/run/run_rce_convection_sweep.py-77-
scripts/run/run_rce_convection_sweep.py:78:    JAX_ENABLE_X64=1 python scripts/run_rce_convection_sweep.py \\
scripts/run/run_rce_convection_sweep.py-79-        --schemes tiedtke,zhang_mcfarlane --days 10
scripts/run/run_rce_convection_sweep.py-80-"""
scripts/run/run_rce_convection_sweep.py-81-
scripts/run/run_rce_convection_sweep.py-82-from __future__ import annotations
--
.github/workflows/ci.yml-93-        with:
.github/workflows/ci.yml-94-          python-version: "3.11"
.github/workflows/ci.yml-95-      - run: pip install -e ".[dev]"
.github/workflows/ci.yml-96-      - name: Verify all tests collect
.github/workflows/ci.yml:97:        run: python -m pytest --collect-only tests/ 2>&1
.github/workflows/ci.yml-98-
.github/workflows/ci.yml-99-  unit-tests:
.github/workflows/ci.yml-100-    name: Unit Tests
.github/workflows/ci.yml-101-    runs-on: ubuntu-latest
--
.github/workflows/ci.yml-108-      - uses: actions/setup-python@v5
.github/workflows/ci.yml-109-        with:
.github/workflows/ci.yml-110-          python-version: ${{ matrix.python-version }}
.github/workflows/ci.yml-111-      - run: pip install -e ".[dev]"
.github/workflows/ci.yml:112:      # Deliberately NO JAX_ENABLE_X64 here: the unit tier is fp32-by-default
.github/workflows/ci.yml-113-      # (CLAUDE.md: "finite-volume can float32"); unit tests that need x64
.github/workflows/ci.yml-114-      # self-enable it at module scope (e.g. test_spectral_plane_ops.py,
.github/workflows/ci.yml-115-      # test_precision_modes.py). Setting it job-wide would change what the
.github/workflows/ci.yml-116-      # fp32 paths test. x64 jobs: top-level-fidelity-tests, visual-regression.
.github/workflows/ci.yml:117:      # Known wart: a module-level jax.config.update("jax_enable_x64", True)
.github/workflows/ci.yml:118:      # leaks across the pytest session for subsequently-imported modules.
.github/workflows/ci.yml-119-      - name: Run unit tests
.github/workflows/ci.yml-120-        run: |
.github/workflows/ci.yml:121:          python -m pytest tests/unit/ -x --timeout=300 \
.github/workflows/ci.yml-122-            -k "not test_coupler_with_3d_ocean_fc_gram" \
.github/workflows/ci.yml-123-            --tb=short -q
.github/workflows/ci.yml-124-        timeout-minutes: 30
.github/workflows/ci.yml-125-
--
.github/workflows/ci.yml-134-    # on every push.
.github/workflows/ci.yml-135-    #
.github/workflows/ci.yml-136-    # Scope: only files matching `tests/test_*.py` at the top level
.github/workflows/ci.yml-137-    # of `tests/` — not subdirectories.  Subdirectory tests
.github/workflows/ci.yml:138:    # (`tests/atmosphere/`, `tests/distributed/`, etc.) have their
.github/workflows/ci.yml-139-    # own CI jobs or are deliberately excluded for cost.
.github/workflows/ci.yml-140-    name: Top-level Fortran-fidelity Tests
.github/workflows/ci.yml-141-    runs-on: ubuntu-latest
.github/workflows/ci.yml-142-    needs: [install-smoke, test-collect]
--
.github/workflows/ci.yml-150-          python-version: ${{ matrix.python-version }}
.github/workflows/ci.yml-151-      - run: pip install -e ".[dev]"
.github/workflows/ci.yml-152-      - name: Run top-level tests/test_*.py
.github/workflows/ci.yml-153-        env:
.github/workflows/ci.yml:154:          JAX_ENABLE_X64: "1"
.github/workflows/ci.yml-155-        # Use shell glob to restrict to top-level files only.
.github/workflows/ci.yml:156:        # `bash -c` is required so the glob expands before pytest
.github/workflows/ci.yml:157:        # sees the arguments; otherwise pytest gets the literal
.github/workflows/ci.yml-158-        # `tests/test_*.py` and reports "no such file".
.github/workflows/ci.yml-159-        run: |
.github/workflows/ci.yml-160-          shopt -s nullglob
.github/workflows/ci.yml-161-          files=( tests/test_*.py )
--
.github/workflows/ci.yml-177-          # subset passes on its own, so breaking the sequence up avoids it —
.github/workflows/ci.yml-178-          # and it is ~4x faster.  NOTE this is a mitigation, not a guarantee:
.github/workflows/ci.yml-179-          # loadfile keeps a worker alive across the files it is handed, so
.github/workflows/ci.yml-180-          # process state still accumulates within a worker.  If the crash
.github/workflows/ci.yml:181:          # returns, the deterministic fallback is one pytest process per file.
.github/workflows/ci.yml:182:          python -m pytest "${files[@]}" --timeout=300 \
.github/workflows/ci.yml-183-            --tb=short -q -n 4 --dist loadfile
.github/workflows/ci.yml-184-        shell: bash
.github/workflows/ci.yml-185-        timeout-minutes: 30
.github/workflows/ci.yml-186-      - name: FV3-native grid fidelity tests (phase 1)
--
.github/workflows/ci.yml-190-        # iter62 (fixed create-layout centres) + iter73 (seam no-collapse)
.github/workflows/ci.yml-191-        # protect the _GNOMONIC_ED_FACE_PERM/_ROT remap TOPOLOGY, which the
.github/workflows/ci.yml-192-        # unordered point-cloud oracle test deliberately cannot see.
.github/workflows/ci.yml-193-        env:
.github/workflows/ci.yml:194:          JAX_ENABLE_X64: "1"
.github/workflows/ci.yml-195-        run: |
.github/workflows/ci.yml:196:          python -m pytest tests/grids/test_fv3_native_grid_phase1.py \
.github/workflows/ci.yml-197-            tests/grids/test_fv3_native_metrics_phase2.py \
.github/workflows/ci.yml-198-            tests/grids/test_fv3_native_halos_phase3.py \
.github/workflows/ci.yml-199-            tests/grids/test_gnomonic_ed_gate_iter68.py \
.github/workflows/ci.yml-200-            tests/grids/test_gnomonic_ed_centers_iter62.py \
--
.github/workflows/ci.yml-205-        # must EXECUTE in CI, not sit behind a skipif.  gfortran is
.github/workflows/ci.yml-206-        # asserted so a toolchain regression fails loudly instead of
.github/workflows/ci.yml-207-        # silently skipping the oracle.
.github/workflows/ci.yml-208-        env:
.github/workflows/ci.yml:209:          JAX_ENABLE_X64: "1"
.github/workflows/ci.yml-210-        run: |
.github/workflows/ci.yml-211-          sudo apt-get update -qq && sudo apt-get install -y -qq gfortran
.github/workflows/ci.yml-212-          gfortran --version
.github/workflows/ci.yml:213:          python -m pytest tests/grids/test_fv3_native_swcore_phase4.py \
.github/workflows/ci.yml-214-            -x --timeout=900 --tb=short -q
.github/workflows/ci.yml-215-        timeout-minutes: 30
.github/workflows/ci.yml-216-
.github/workflows/ci.yml-217-  restart-smoke:
--
.github/workflows/ci.yml-225-          python-version: "3.11"
.github/workflows/ci.yml-226-      - run: pip install -e ".[dev]"
.github/workflows/ci.yml-227-      - name: Restart roundtrip test
.github/workflows/ci.yml-228-        run: |
.github/workflows/ci.yml:229:          python -m pytest tests/unit/test_cmor_experiments_restart.py \
.github/workflows/ci.yml-230-            tests/unit/test_zarr_checkpoint.py -v --tb=short
.github/workflows/ci.yml-231-        timeout-minutes: 15
.github/workflows/ci.yml-232-
.github/workflows/ci.yml-233-  integration-smoke:
--
.github/workflows/ci.yml-241-          python-version: "3.11"
.github/workflows/ci.yml-242-      - run: pip install -e ".[dev]"
.github/workflows/ci.yml-243-      - name: Atmosphere integration smoke
.github/workflows/ci.yml-244-        run: |
.github/workflows/ci.yml:245:          python -m pytest tests/atmosphere/ -x --timeout=600 \
.github/workflows/ci.yml-246-            -k "test_williamson or test_held_suarez" \
.github/workflows/ci.yml-247-            --tb=short -q 2>&1 | head -50
.github/workflows/ci.yml-248-        timeout-minutes: 30
.github/workflows/ci.yml-249-
--
.github/workflows/ci.yml-265-          python-version: "3.11"
.github/workflows/ci.yml-266-      - run: pip install -e ".[dev]"
.github/workflows/ci.yml-267-      - name: W2 cube v-wind visual-regression check
.github/workflows/ci.yml-268-        env:
.github/workflows/ci.yml:269:          JAX_ENABLE_X64: "1"
.github/workflows/ci.yml-270-          JAX_PLATFORMS: "cpu"
.github/workflows/ci.yml-271-        run: |
.github/workflows/ci.yml-272-          python scripts/validate/visual_regression.py --check 2>&1 | tail -20
.github/workflows/ci.yml-273-        timeout-minutes: 20
--
.github/workflows/mpi-distributed.yml-48-        run: |
.github/workflows/mpi-distributed.yml-49-          for n in 2 3 6; do
.github/workflows/mpi-distributed.yml-50-            echo "=== MPI distributed test, np=${n} ==="
.github/workflows/mpi-distributed.yml-51-            mpirun --oversubscribe -np "${n}" \
.github/workflows/mpi-distributed.yml:52:              python -m pytest -q tests/distributed/test_halo_mpi.py
.github/workflows/mpi-distributed.yml-53-          done
.github/workflows/mpi-distributed.yml-54-
.github/workflows/mpi-distributed.yml-55-      - name: Run Plane Pencil MPI Tests (np=2,4)
.github/workflows/mpi-distributed.yml-56-        env:
--
.github/workflows/mpi-distributed.yml-58-        run: |
.github/workflows/mpi-distributed.yml-59-          for n in 2 4; do
.github/workflows/mpi-distributed.yml-60-            echo "=== Plane pencil MPI test, np=${n} ==="
.github/workflows/mpi-distributed.yml-61-            mpirun --oversubscribe -np "${n}" \
.github/workflows/mpi-distributed.yml:62:              python -m pytest -q tests/distributed/test_plane_pencil_mpi.py
.github/workflows/mpi-distributed.yml-63-          done
.github/workflows/mpi-distributed.yml-64-
.github/workflows/mpi-distributed.yml-65-      - name: Run Lat-Lon 2-D Pad (wall) MPI Tests (np=2,3,6)
.github/workflows/mpi-distributed.yml-66-        # np=3 -> 1x3 drives the E/W periodic ring through proc_lon=3 in
--
.github/workflows/mpi-distributed.yml-71-        run: |
.github/workflows/mpi-distributed.yml-72-          for n in 2 3 6; do
.github/workflows/mpi-distributed.yml-73-            echo "=== Lat-lon 2-D wall-pad MPI test, np=${n} ==="
.github/workflows/mpi-distributed.yml-74-            mpirun --oversubscribe -np "${n}" \
.github/workflows/mpi-distributed.yml:75:              python -m pytest -q tests/distributed/test_latlon_2d_pad_wall_mpi.py
.github/workflows/mpi-distributed.yml-76-          done
.github/workflows/mpi-distributed.yml-77-
.github/workflows/mpi-distributed.yml-78-      - name: Run Lat-Lon transpose AD (lon_gather_full VJP) MPI Tests (np=2,3,6)
.github/workflows/mpi-distributed.yml-79-        # Adjoint (dot-product identity) + grad-finiteness of the lat-pencil
--
.github/workflows/mpi-distributed.yml-84-        run: |
.github/workflows/mpi-distributed.yml-85-          for n in 2 3 6; do
.github/workflows/mpi-distributed.yml-86-            echo "=== Lat-lon transpose-AD MPI test, np=${n} ==="
.github/workflows/mpi-distributed.yml-87-            mpirun --oversubscribe -np "${n}" \
.github/workflows/mpi-distributed.yml:88:              python -m pytest -q tests/distributed/test_latlon_transpose_ad_mpi.py
.github/workflows/mpi-distributed.yml-89-          done
--
scripts/experiment/train_mc3d_emulator.py-6-train/val MSE + relative error vs the Monte-Carlo target. The trained emulator
scripts/experiment/train_mc3d_emulator.py-7-is a smooth, ``jax.grad``-able surrogate for the (non-differentiable) photon MC.
scripts/experiment/train_mc3d_emulator.py-8-
scripts/experiment/train_mc3d_emulator.py-9-Usage:
scripts/experiment/train_mc3d_emulator.py:10:    JAX_ENABLE_X64=1 python scripts/experiment/train_mc3d_emulator.py \
scripts/experiment/train_mc3d_emulator.py-11-        --data scripts/tmp/_mc3d_emu.npz --epochs 50 --out scripts/tmp/_emu.eqx
scripts/experiment/train_mc3d_emulator.py-12-"""
scripts/experiment/train_mc3d_emulator.py-13-
scripts/experiment/train_mc3d_emulator.py-14-from __future__ import annotations
--
scripts/experiment/train_mc3d_emulator.py-20-import jax.numpy as jnp
scripts/experiment/train_mc3d_emulator.py-21-import numpy as np
scripts/experiment/train_mc3d_emulator.py-22-
scripts/experiment/train_mc3d_emulator.py-23-# NN training runs fine in fp32 (the default) and fp32 is required on backends
scripts/experiment/train_mc3d_emulator.py:24:# without x64 (e.g. Apple MPS / many GPUs). Honor JAX_ENABLE_X64 from the env
scripts/experiment/train_mc3d_emulator.py-25-# (set it to 1 for an fp64 CPU run); do NOT force x64 here.
scripts/experiment/train_mc3d_emulator.py-26-
scripts/experiment/train_mc3d_emulator.py-27-from legoesm.atmosphere.physics.radiation.mc3d.emulator import (
scripts/experiment/train_mc3d_emulator.py-28-    EmulatorConfig,
--
.github/workflows/mpi-nightly.yml-43-          # (parallel/reductions.py, NOT suppressed) is the authoritative check.
.github/workflows/mpi-nightly.yml-44-          MPI4JAX_NO_WARN_JAX_VERSION: "1"
.github/workflows/mpi-nightly.yml-45-        run: |
.github/workflows/mpi-nightly.yml-46-          mpirun --oversubscribe -np 3 \
.github/workflows/mpi-nightly.yml:47:            python -m pytest -q tests/distributed/test_ocean_mpi_conservation.py -k longrun
--
scripts/run/run_aimip_headtohead_t106.py-219-mkdir -p {HEADTOHEAD_RESULTS_REL}/slurm_logs
scripts/run/run_aimip_headtohead_t106.py-220-
scripts/run/run_aimip_headtohead_t106.py-221-export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:${{PYTHONPATH:-}}
scripts/run/run_aimip_headtohead_t106.py-222-export JAX_PLATFORMS=cuda
scripts/run/run_aimip_headtohead_t106.py:223:export JAX_ENABLE_X64=1
scripts/run/run_aimip_headtohead_t106.py-224-
scripts/run/run_aimip_headtohead_t106.py-225-/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \\
scripts/run/run_aimip_headtohead_t106.py-226-    scripts/run/run_aimip.py \\
scripts/run/run_aimip_headtohead_t106.py-227-    --suite {HEADTOHEAD_DIR_REL}/suite.yaml \\
--
scripts/validate/scm_microphysics_cases.py-129-
scripts/validate/scm_microphysics_cases.py-130-
scripts/validate/scm_microphysics_cases.py-131-def run_case(scheme, case, dtype, n_steps=None, verbose=False):
scripts/validate/scm_microphysics_cases.py-132-    """Run one scheme on one case at one precision; return a result dict."""
scripts/validate/scm_microphysics_cases.py:133:    jax.config.update("jax_enable_x64", dtype == jnp.float64)
scripts/validate/scm_microphysics_cases.py-134-    base_case = case.replace("_cold", "")
scripts/validate/scm_microphysics_cases.py-135-    # Cold variants shift θ_l down so the cloud is supercooled (mixed-phase) —
scripts/validate/scm_microphysics_cases.py-136-    # exercises freezing / ice / melt in the ice-capable schemes that the warm
scripts/validate/scm_microphysics_cases.py-137-    # cases leave untested. q_t is scaled DOWN with Clausius-Clapeyron so the
--
scripts/validate/scm_microphysics_cases.py-221-
scripts/validate/scm_microphysics_cases.py-222-    micro_jit = jax.jit(micro_step)
scripts/validate/scm_microphysics_cases.py-223-
scripts/validate/scm_microphysics_cases.py-224-    # Codex methodology review: the float32 run must be GENUINELY float32
scripts/validate/scm_microphysics_cases.py:225:    # (toggling jax_enable_x64 alone is not proof) — assert the scheme output
scripts/validate/scm_microphysics_cases.py-226-    # dtype before trusting the precision result.
scripts/validate/scm_microphysics_cases.py-227-    out0 = micro_jit(state)
scripts/validate/scm_microphysics_cases.py-228-    for fld in (out0.dT_dt, out0.dq_v_dt, out0.dq_c_dt, out0.precipitation):
scripts/validate/scm_microphysics_cases.py-229-        if fld.dtype != dtype:
--
scripts/cluster/les_scm/test_bridge.sbatch-15-echo "PINNED sha: $(git rev-parse HEAD)"
scripts/cluster/les_scm/test_bridge.sbatch-16-echo "host: $(hostname)"
scripts/cluster/les_scm/test_bridge.sbatch-17-export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
scripts/cluster/les_scm/test_bridge.sbatch-18-export JAX_PLATFORMS=cpu
scripts/cluster/les_scm/test_bridge.sbatch:19:export JAX_ENABLE_X64=1
scripts/cluster/les_scm/test_bridge.sbatch-20-PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
scripts/cluster/les_scm/test_bridge.sbatch-21-fail=0
scripts/cluster/les_scm/test_bridge.sbatch-22-
scripts/cluster/les_scm/test_bridge.sbatch-23-echo "=== les_record (numpy only, no conftest) ==="
scripts/cluster/les_scm/test_bridge.sbatch:24:$PY -m pytest tests/unit/test_les_record.py -q -p no:cacheprovider --noconftest
scripts/cluster/les_scm/test_bridge.sbatch-25-rc=$?; echo "les_record rc=$rc"; [ $rc -ne 0 ] && fail=1
scripts/cluster/les_scm/test_bridge.sbatch-26-
scripts/cluster/les_scm/test_bridge.sbatch-27-echo "=== LES reference loader ==="
scripts/cluster/les_scm/test_bridge.sbatch:28:$PY -m pytest tests/unit/test_les_reference.py -q -p no:cacheprovider
scripts/cluster/les_scm/test_bridge.sbatch-29-rc=$?; echo "les_reference rc=$rc"; [ $rc -ne 0 ] && fail=1
scripts/cluster/les_scm/test_bridge.sbatch-30-
scripts/cluster/les_scm/test_bridge.sbatch-31-echo "=== gSAM deck -> SCM bridge ==="
scripts/cluster/les_scm/test_bridge.sbatch:32:$PY -m pytest tests/atmosphere/hydrostatic/unit/test_sam_case_scm.py -q \
scripts/cluster/les_scm/test_bridge.sbatch-33-    -p no:cacheprovider
scripts/cluster/les_scm/test_bridge.sbatch-34-rc=$?; echo "bridge rc=$rc"; [ $rc -ne 0 ] && fail=1
scripts/cluster/les_scm/test_bridge.sbatch-35-
scripts/cluster/les_scm/test_bridge.sbatch-36-echo "=== turbulence tuning driver ==="
scripts/cluster/les_scm/test_bridge.sbatch:37:$PY -m pytest tests/atmosphere/hydrostatic/test_run_scm_les_turbulence_tuning.py \
scripts/cluster/les_scm/test_bridge.sbatch-38-    -q -p no:cacheprovider
scripts/cluster/les_scm/test_bridge.sbatch-39-rc=$?; echo "driver rc=$rc"; [ $rc -ne 0 ] && fail=1
scripts/cluster/les_scm/test_bridge.sbatch-40-
scripts/cluster/les_scm/test_bridge.sbatch-41-echo "=== existing SCM + forcing + metrics regression (must stay green) ==="
scripts/cluster/les_scm/test_bridge.sbatch:42:$PY -m pytest tests/atmosphere/hydrostatic/unit/test_scm.py \
scripts/cluster/les_scm/test_bridge.sbatch-43-             tests/atmosphere/hydrostatic/unit/test_scm_forcing.py \
scripts/cluster/les_scm/test_bridge.sbatch-44-             tests/unit/test_sam_case_forcing.py \
scripts/cluster/les_scm/test_bridge.sbatch-45-             tests/unit/test_scm_rce_metrics.py -q -p no:cacheprovider
scripts/cluster/les_scm/test_bridge.sbatch-46-rc=$?; echo "regression rc=$rc"; [ $rc -ne 0 ] && fail=1
--
scripts/cluster/unified_training/bench_wb2_ace2loss.sbatch-14-set -euo pipefail
scripts/cluster/unified_training/bench_wb2_ace2loss.sbatch-15-
scripts/cluster/unified_training/bench_wb2_ace2loss.sbatch-16-source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/_env.sh
scripts/cluster/unified_training/bench_wb2_ace2loss.sbatch-17-cd "$UT_REPO"
scripts/cluster/unified_training/bench_wb2_ace2loss.sbatch:18:export JAX_ENABLE_X64=1
scripts/cluster/unified_training/bench_wb2_ace2loss.sbatch-19-export JAX_PLATFORMS=cuda,cpu
scripts/cluster/unified_training/bench_wb2_ace2loss.sbatch-20-export XLA_PYTHON_CLIENT_PREALLOCATE=false
scripts/cluster/unified_training/bench_wb2_ace2loss.sbatch-21-export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
scripts/cluster/unified_training/bench_wb2_ace2loss.sbatch-22-
--
scripts/experiment/init_experiment.py-98-    Single source of truth for the launcher shell, shared by ``init_experiment``
scripts/experiment/init_experiment.py-99-    (``run_cmd='legoesm run config.yaml'``) and the interactive ``wizard`` (which
scripts/experiment/init_experiment.py-100-    routes to ``run_coupled``, the ocean/SCM matrices, the plane-LES scripts, or
scripts/experiment/init_experiment.py-101-    training — none of which are ``legoesm run``).  ``enable_x64`` adds
scripts/experiment/init_experiment.py:102:    ``export JAX_ENABLE_X64=1`` for 64-bit runs.  ``workdir`` overrides the launch
scripts/experiment/init_experiment.py-103-    directory (default: the bundle dir, ``$(dirname "$0")``) — needed when the
scripts/experiment/init_experiment.py-104-    command references repo-root-relative paths (e.g. the AIMIP suite manifest).
scripts/experiment/init_experiment.py-105-    Honors the machine profile's ``jax_platforms`` / ``venv_activate`` / SLURM
scripts/experiment/init_experiment.py-106-    directives just as before.
--
scripts/experiment/init_experiment.py-127-    if venv:
scripts/experiment/init_experiment.py-128-        lines.append(venv)
scripts/experiment/init_experiment.py-129-    lines.append(f"export JAX_PLATFORMS={plat}")
scripts/experiment/init_experiment.py-130-    if enable_x64:
scripts/experiment/init_experiment.py:131:        lines.append("export JAX_ENABLE_X64=1")
scripts/experiment/init_experiment.py-132-    lines.append(run_cmd)
scripts/experiment/init_experiment.py-133-    lines.append("")
scripts/experiment/init_experiment.py-134-    return "\n".join(lines)
scripts/experiment/init_experiment.py-135-
--
scripts/run/run_wb_sweep_stage1.py-91-cd "$WORKTREE"
scripts/run/run_wb_sweep_stage1.py-92-# worktree-pinned env (jn2808 conda python + worktree packages on PYTHONPATH)
scripts/run/run_wb_sweep_stage1.py-93-source "$WORKTREE/scripts/cluster/wb_forecast/env.sh"
scripts/run/run_wb_sweep_stage1.py-94-export JAX_PLATFORMS=cuda
scripts/run/run_wb_sweep_stage1.py:95:export JAX_ENABLE_X64=1
scripts/run/run_wb_sweep_stage1.py-96-
scripts/run/run_wb_sweep_stage1.py-97-MANIFEST="{manifest_rel}"
scripts/run/run_wb_sweep_stage1.py-98-SUITE_PATH=$("$WB_PYTHON" -c "
scripts/run/run_wb_sweep_stage1.py-99-import json, sys
--
scripts/run/run_spectral_cbl.py-33-_F32 = "--f32" in sys.argv
scripts/run/run_spectral_cbl.py-34-import jax  # noqa: E402
scripts/run/run_spectral_cbl.py-35-
scripts/run/run_spectral_cbl.py-36-if not _F32:
scripts/run/run_spectral_cbl.py:37:    jax.config.update("jax_enable_x64", True)
scripts/run/run_spectral_cbl.py-38-import jax.numpy as jnp  # noqa: E402
scripts/run/run_spectral_cbl.py-39-
scripts/run/run_spectral_cbl.py-40-from legoesm import constants  # noqa: E402
scripts/run/run_spectral_cbl.py-41-from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402
--
scripts/experiment/ck_sensitivity_vs_era5.py-26-
scripts/experiment/ck_sensitivity_vs_era5.py-27-import numpy as np
scripts/experiment/ck_sensitivity_vs_era5.py-28-
scripts/experiment/ck_sensitivity_vs_era5.py-29-# Make the sibling ``scripts.*`` entry points (``scripts.data.load_local_era5``) importable
scripts/experiment/ck_sensitivity_vs_era5.py:30:# when this file is run as a standalone CLI; under pytest the repo root is already on the
scripts/experiment/ck_sensitivity_vs_era5.py-31-# path, so the guard makes this a no-op there.
scripts/experiment/ck_sensitivity_vs_era5.py-32-_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
scripts/experiment/ck_sensitivity_vs_era5.py-33-if str(_REPO_ROOT) not in sys.path:
scripts/experiment/ck_sensitivity_vs_era5.py-34-    sys.path.insert(0, str(_REPO_ROOT))
--
scripts/experiment/ck_sensitivity_vs_era5.py-276-    build the ocean-only mask the campaign ranks over; an empty path keeps the flat model
scripts/experiment/ck_sensitivity_vs_era5.py-277-    (all-ocean land fraction → an ocean-only restriction is a safe no-op).  ``grid_type``
scripts/experiment/ck_sensitivity_vs_era5.py-278-    matches the campaign's grid family (iter 493)."""
scripts/experiment/ck_sensitivity_vs_era5.py-279-    import jax
scripts/experiment/ck_sensitivity_vs_era5.py:280:    jax.config.update("jax_enable_x64", True)
scripts/experiment/ck_sensitivity_vs_era5.py-281-    from legoesm.driver.coupled_config import PRESETS
scripts/experiment/ck_sensitivity_vs_era5.py-282-    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
scripts/experiment/ck_sensitivity_vs_era5.py-283-    from legoesm.grids.latlon import create_latlon_grid
scripts/experiment/ck_sensitivity_vs_era5.py-284-    from legoesm.training.compare_reanalysis import column_state_from_hydrostatic
--
scripts/experiment/wizard_core.py-376-
scripts/experiment/wizard_core.py-377-    kind: str                      # atm_global | coupled | ocean | les_crm | scm | train
scripts/experiment/wizard_core.py-378-    run_cmd: str                   # the command run.sh will execute
scripts/experiment/wizard_core.py-379-    jax_platforms: str             # cpu | cuda
scripts/experiment/wizard_core.py:380:    enable_x64: bool               # 64-bit → export JAX_ENABLE_X64=1
scripts/experiment/wizard_core.py-381-    summary: dict[str, Any]        # axis → chosen value (for display + record)
scripts/experiment/wizard_core.py-382-    config: Any = None             # legoesm.config.Config for the template path; else None
scripts/experiment/wizard_core.py-383-    notes: list[str] = field(default_factory=list)  # honest caveats / prerequisites
scripts/experiment/wizard_core.py-384-    workdir: str | None = None     # run.sh cwd; None → bundle dir, else this path
--
scripts/bench/bench_barotropic_mcut.py-23-
scripts/bench/bench_barotropic_mcut.py-24-import argparse
scripts/bench/bench_barotropic_mcut.py-25-import os
scripts/bench/bench_barotropic_mcut.py-26-
scripts/bench/bench_barotropic_mcut.py:27:os.environ.setdefault("JAX_ENABLE_X64", "1")
scripts/bench/bench_barotropic_mcut.py-28-
scripts/bench/bench_barotropic_mcut.py-29-import jax
scripts/bench/bench_barotropic_mcut.py:30:jax.config.update("jax_enable_x64", True)
scripts/bench/bench_barotropic_mcut.py-31-import jax.numpy as jnp
scripts/bench/bench_barotropic_mcut.py-32-import numpy as np
scripts/bench/bench_barotropic_mcut.py-33-
scripts/bench/bench_barotropic_mcut.py-34-from mpi4py import MPI
--
scripts/validate/validate_baro_solver.py-125-    args = p.parse_args()
scripts/validate/validate_baro_solver.py-126-
scripts/validate/validate_baro_solver.py-127-    # Set JAX x64 if needed before any JAX import via build helpers.
scripts/validate/validate_baro_solver.py-128-    if args.precision == "float64":
scripts/validate/validate_baro_solver.py:129:        os.environ["JAX_ENABLE_X64"] = "1"
scripts/validate/validate_baro_solver.py-130-
scripts/validate/validate_baro_solver.py-131-    if args.grid == "latlon":
scripts/validate/validate_baro_solver.py-132-        key = args.n_lat
scripts/validate/validate_baro_solver.py-133-        m1, s1, _ = _build_latlon(key, args.precision == "float64",
--
scripts/cluster/les_scm/tune_turbulence.sbatch-28-echo "case: ${LES_CASE:?set LES_CASE}"
scripts/cluster/les_scm/tune_turbulence.sbatch-29-export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
scripts/cluster/les_scm/tune_turbulence.sbatch-30-export XLA_PYTHON_CLIENT_PREALLOCATE=false
scripts/cluster/les_scm/tune_turbulence.sbatch-31-export JAX_PLATFORMS=cuda
scripts/cluster/les_scm/tune_turbulence.sbatch:32:export JAX_ENABLE_X64=1
scripts/cluster/les_scm/tune_turbulence.sbatch-33-PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
scripts/cluster/les_scm/tune_turbulence.sbatch-34-
scripts/cluster/les_scm/tune_turbulence.sbatch-35-REPO=/burg-archive/glab/users/pg2328/legoESM
scripts/cluster/les_scm/tune_turbulence.sbatch-36-LES_DIR=$REPO/results/les_ref/$LES_CASE
--
scripts/validate/validate_sdm_vs_pysdm.py-10-
scripts/validate/validate_sdm_vs_pysdm.py-11-Two-process design (PySDM's numba stack must not enter the repo's jax venv):
scripts/validate/validate_sdm_vs_pysdm.py-12-
scripts/validate/validate_sdm_vs_pysdm.py-13-1. ``/tmp/pysdm_venv/bin/python scripts/validate/pysdm_golovin_reference.py ref.npz``
scripts/validate/validate_sdm_vs_pysdm.py:14:2. ``JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
scripts/validate/validate_sdm_vs_pysdm.py-15-       scripts/validate/validate_sdm_vs_pysdm.py ref.npz``
scripts/validate/validate_sdm_vs_pysdm.py-16-
scripts/validate/validate_sdm_vs_pysdm.py-17-Compares, at t = 0/1200/2400/3600 s: total number density N(t), LWC (must be
scripts/validate/validate_sdm_vs_pysdm.py-18-conserved by both), the 2nd mass moment M2, and the mass-density spectra
--
scripts/validate/validate_sdm_vs_pysdm.py-186-    sys.exit(0 if ok else 1)
scripts/validate/validate_sdm_vs_pysdm.py-187-
scripts/validate/validate_sdm_vs_pysdm.py-188-
scripts/validate/validate_sdm_vs_pysdm.py-189-if __name__ == "__main__":
scripts/validate/validate_sdm_vs_pysdm.py:190:    jax.config.update("jax_enable_x64", True)
scripts/validate/validate_sdm_vs_pysdm.py-191-    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pysdm_golovin_ref.npz")
--
scripts/cluster/les_scm/pipeline_smoke.sbatch-29-echo "host: $(hostname)"; nvidia-smi -L
scripts/cluster/les_scm/pipeline_smoke.sbatch-30-export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
scripts/cluster/les_scm/pipeline_smoke.sbatch-31-export XLA_PYTHON_CLIENT_PREALLOCATE=false
scripts/cluster/les_scm/pipeline_smoke.sbatch-32-export JAX_PLATFORMS=cuda
scripts/cluster/les_scm/pipeline_smoke.sbatch:33:export JAX_ENABLE_X64=1
scripts/cluster/les_scm/pipeline_smoke.sbatch-34-PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
scripts/cluster/les_scm/pipeline_smoke.sbatch-35-
scripts/cluster/les_scm/pipeline_smoke.sbatch-36-REPO=/burg-archive/glab/users/pg2328/legoESM
scripts/cluster/les_scm/pipeline_smoke.sbatch-37-OUT=$REPO/results/scm_les_turbulence/_pipeline_smoke_bomex
--
scripts/experiment/run_mpi_tests.sh-6-# Running MULTIPLE files in ONE ``mpirun`` session desyncs the ranks — a
scripts/experiment/run_mpi_tests.sh-7-# leftover/unmatched request from one file trips MPICH's ``dtp_ != NULL``
scripts/experiment/run_mpi_tests.sh-8-# assertion (ch3u_request.c) or deadlocks the next file. Each file passes in
scripts/experiment/run_mpi_tests.sh-9-# isolation, so we launch a FRESH ``mpirun`` per file (fresh MPI_Init/Finalize),
scripts/experiment/run_mpi_tests.sh:10:# guarded by a HARD wall-clock ``timeout`` (pytest's signal-timeout can't
scripts/experiment/run_mpi_tests.sh-11-# interrupt a blocked MPI C call).
scripts/experiment/run_mpi_tests.sh-12-#
scripts/experiment/run_mpi_tests.sh-13-# Usage:
scripts/experiment/run_mpi_tests.sh-14-#   bash scripts/experiment/run_mpi_tests.sh [np] [per_file_timeout_s]
--
scripts/experiment/run_mpi_tests.sh-23-source scripts/experiment/setup_mpi_local.sh env
scripts/experiment/run_mpi_tests.sh-24-
scripts/experiment/run_mpi_tests.sh-25-PY=".venv-mpi/bin/python"
scripts/experiment/run_mpi_tests.sh-26-declare -a FILES
scripts/experiment/run_mpi_tests.sh:27:mapfile -t FILES < <(ls tests/distributed/test_*.py | sort)
scripts/experiment/run_mpi_tests.sh-28-
scripts/experiment/run_mpi_tests.sh-29-pass=0; fail=0; hang=0
scripts/experiment/run_mpi_tests.sh-30-declare -a FAILED
scripts/experiment/run_mpi_tests.sh-31-for f in "${FILES[@]}"; do
scripts/experiment/run_mpi_tests.sh-32-  printf '%-58s ' "$(basename "$f")"
scripts/experiment/run_mpi_tests.sh-33-  log="/tmp/mpitest_$(basename "$f" .py).log"
scripts/experiment/run_mpi_tests.sh:34:  timeout -s KILL "$TMO" mpirun -np "$NP" "$PY" -m pytest "$f" \
scripts/experiment/run_mpi_tests.sh-35-      -q -p no:cacheprovider --timeout=$((TMO - 30)) > "$log" 2>&1
scripts/experiment/run_mpi_tests.sh-36-  rc=$?
scripts/experiment/run_mpi_tests.sh-37-  if [[ $rc -eq 137 || $rc -eq 124 ]]; then
scripts/experiment/run_mpi_tests.sh-38-    echo "HANG/TIMEOUT (rc=$rc)"; hang=$((hang+1)); FAILED+=("$f [hang]")
packages/core/legoesm/parallel/halo_exchange.py-143-# tag value = 3 * 100_000 + 99_999 = 399_999, well within 2^31-1.
packages/core/legoesm/parallel/halo_exchange.py-144-_MPI_TAG_RANK_STRIDE = 100_000
packages/core/legoesm/parallel/halo_exchange.py-145-
packages/core/legoesm/parallel/halo_exchange.py-146-
packages/core/legoesm/parallel/halo_exchange.py-147-# ======================================================================
packages/core/legoesm/parallel/halo_exchange.py:148:# AD-safe sendrecv wrapper (custom_vjp)
packages/core/legoesm/parallel/halo_exchange.py-149-# ======================================================================
packages/core/legoesm/parallel/halo_exchange.py-150-
packages/core/legoesm/parallel/halo_exchange.py:151:def _make_sendrecv_vjp(mpi4jax_mod):
packages/core/legoesm/parallel/halo_exchange.py-152-    """Build an AD-safe sendrecv once mpi4jax is imported.
packages/core/legoesm/parallel/halo_exchange.py-153-
packages/core/legoesm/parallel/halo_exchange.py-154-    mpi4jax.sendrecv has a transpose rule that swaps source/dest, but
packages/core/legoesm/parallel/halo_exchange.py-155-    the XLA lowering raises RuntimeError when ``_must_transpose=True``.
packages/core/legoesm/parallel/halo_exchange.py:156:    This wrapper bypasses that by using ``@jax.custom_vjp``: the forward
packages/core/legoesm/parallel/halo_exchange.py-157-    calls sendrecv normally, and the backward calls sendrecv with
packages/core/legoesm/parallel/halo_exchange.py-158-    swapped endpoints as a fresh forward call (no transpose flag).
packages/core/legoesm/parallel/halo_exchange.py-159-
packages/core/legoesm/parallel/halo_exchange.py-160-    Non-JAX arguments (source, dest, sendtag, recvtag, comm) are
packages/core/legoesm/parallel/halo_exchange.py-161-    declared via ``nondiff_argnums`` so JAX does not attempt to trace
--
packages/core/legoesm/parallel/halo_exchange.py-170-                source=source, dest=dest,
packages/core/legoesm/parallel/halo_exchange.py-171-                sendtag=sendtag, recvtag=recvtag, comm=comm,
packages/core/legoesm/parallel/halo_exchange.py-172-            )
packages/core/legoesm/parallel/halo_exchange.py-173-        )
packages/core/legoesm/parallel/halo_exchange.py-174-
packages/core/legoesm/parallel/halo_exchange.py:175:    _sendrecv = jax.custom_vjp(
packages/core/legoesm/parallel/halo_exchange.py-176-        _sendrecv_impl, nondiff_argnums=(2, 3, 4, 5, 6),
packages/core/legoesm/parallel/halo_exchange.py-177-    )
packages/core/legoesm/parallel/halo_exchange.py-178-
packages/core/legoesm/parallel/halo_exchange.py-179-    def _fwd(send_buf, recv_template, source, dest, sendtag, recvtag, comm):
packages/core/legoesm/parallel/halo_exchange.py-180-        # fwd has the same signature as the primal function.
--
packages/core/legoesm/parallel/halo_exchange.py-220-    _sendrecv.defvjp(_fwd, _bwd)
packages/core/legoesm/parallel/halo_exchange.py-221-    return _sendrecv
packages/core/legoesm/parallel/halo_exchange.py-222-
packages/core/legoesm/parallel/halo_exchange.py-223-
packages/core/legoesm/parallel/halo_exchange.py-224-# Module-level cache: built lazily on first MPI import.
packages/core/legoesm/parallel/halo_exchange.py:225:_sendrecv_vjp_fn = None
packages/core/legoesm/parallel/halo_exchange.py-226-
packages/core/legoesm/parallel/halo_exchange.py-227-
packages/core/legoesm/parallel/halo_exchange.py:228:def get_sendrecv_vjp(mpi4jax_mod):
packages/core/legoesm/parallel/halo_exchange.py-229-    """Return the cached AD-safe sendrecv wrapper.
packages/core/legoesm/parallel/halo_exchange.py-230-
packages/core/legoesm/parallel/halo_exchange.py-231-    This is the single choke point every halo ``sendrecv`` routes through, so
packages/core/legoesm/parallel/halo_exchange.py-232-    it is where the mpi4jax GPU-transport preflight runs — on EVERY call, BEFORE
packages/core/legoesm/parallel/halo_exchange.py-233-    the (memoised) wrapper is returned. Halo entry points reach mpi4jax by a
--
packages/core/legoesm/parallel/halo_exchange.py-235-    this call a GPU-direct misconfiguration (``MPI4JAX_USE_CUDA_MPI=1`` against a
packages/core/legoesm/parallel/halo_exchange.py-236-    non-CUDA mpi4jax) would slip past the preflight and segfault at the first
packages/core/legoesm/parallel/halo_exchange.py-237-    device-buffer exchange.
packages/core/legoesm/parallel/halo_exchange.py-238-    """
packages/core/legoesm/parallel/halo_exchange.py-239-    check_mpi4jax_transport(mpi4jax_mod)
packages/core/legoesm/parallel/halo_exchange.py:240:    global _sendrecv_vjp_fn
packages/core/legoesm/parallel/halo_exchange.py:241:    if _sendrecv_vjp_fn is None:
packages/core/legoesm/parallel/halo_exchange.py:242:        _sendrecv_vjp_fn = _make_sendrecv_vjp(mpi4jax_mod)
packages/core/legoesm/parallel/halo_exchange.py:243:    return _sendrecv_vjp_fn
packages/core/legoesm/parallel/halo_exchange.py-244-
packages/core/legoesm/parallel/halo_exchange.py-245-
packages/core/legoesm/parallel/halo_exchange.py-246-def _place_strip(padded: jax.Array, face: int, edge: int, strip: jax.Array) -> jax.Array:
packages/core/legoesm/parallel/halo_exchange.py-247-    """Place a received strip into the correct halo position."""
packages/core/legoesm/parallel/halo_exchange.py-248-    if edge == WEST:
--
packages/core/legoesm/parallel/halo_exchange.py-482-
packages/core/legoesm/parallel/halo_exchange.py-483-        # Single sendrecv per neighbor.  Tags + neighbor ranks remain
packages/core/legoesm/parallel/halo_exchange.py-484-        # global — they identify the *MPI peer*, not the array slot.
packages/core/legoesm/parallel/halo_exchange.py-485-        send_tag = rank
packages/core/legoesm/parallel/halo_exchange.py-486-        recv_tag = nbr_rank
packages/core/legoesm/parallel/halo_exchange.py:487:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/halo_exchange.py-488-        recv_buf = sendrecv(
packages/core/legoesm/parallel/halo_exchange.py-489-            send_buf, jnp.zeros_like(send_buf),
packages/core/legoesm/parallel/halo_exchange.py-490-            nbr_rank, nbr_rank,
packages/core/legoesm/parallel/halo_exchange.py-491-            send_tag, recv_tag, comm,
packages/core/legoesm/parallel/halo_exchange.py-492-        )
--
packages/core/legoesm/parallel/halo_exchange.py-634-        send_buf = jnp.concatenate(send_parts)
packages/core/legoesm/parallel/halo_exchange.py-635-
packages/core/legoesm/parallel/halo_exchange.py-636-        # Single sendrecv for all edges to this neighbor.
packages/core/legoesm/parallel/halo_exchange.py-637-        send_tag = rank
packages/core/legoesm/parallel/halo_exchange.py-638-        recv_tag = nbr_rank
packages/core/legoesm/parallel/halo_exchange.py:639:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/halo_exchange.py-640-        recv_buf = sendrecv(
packages/core/legoesm/parallel/halo_exchange.py-641-            send_buf, jnp.zeros_like(send_buf),
packages/core/legoesm/parallel/halo_exchange.py-642-            nbr_rank, nbr_rank,
packages/core/legoesm/parallel/halo_exchange.py-643-            send_tag, recv_tag, comm,
packages/core/legoesm/parallel/halo_exchange.py-644-        )
--
packages/core/legoesm/parallel/halo_exchange.py-1034-                    )
packages/core/legoesm/parallel/halo_exchange.py-1035-        send_buf = jnp.concatenate(send_parts)
packages/core/legoesm/parallel/halo_exchange.py-1036-
packages/core/legoesm/parallel/halo_exchange.py-1037-        send_tag = rank
packages/core/legoesm/parallel/halo_exchange.py-1038-        recv_tag = nbr_rank
packages/core/legoesm/parallel/halo_exchange.py:1039:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/halo_exchange.py-1040-        recv_buf = sendrecv(
packages/core/legoesm/parallel/halo_exchange.py-1041-            send_buf, jnp.zeros_like(send_buf),
packages/core/legoesm/parallel/halo_exchange.py-1042-            nbr_rank, nbr_rank,
packages/core/legoesm/parallel/halo_exchange.py-1043-            send_tag, recv_tag, comm,
packages/core/legoesm/parallel/halo_exchange.py-1044-        )
--
packages/core/legoesm/parallel/halo_exchange.py-1179-                    )
packages/core/legoesm/parallel/halo_exchange.py-1180-        send_buf = jnp.concatenate(send_parts)
packages/core/legoesm/parallel/halo_exchange.py-1181-
packages/core/legoesm/parallel/halo_exchange.py-1182-        send_tag = rank
packages/core/legoesm/parallel/halo_exchange.py-1183-        recv_tag = nbr_rank
packages/core/legoesm/parallel/halo_exchange.py:1184:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/halo_exchange.py-1185-        recv_buf = sendrecv(
packages/core/legoesm/parallel/halo_exchange.py-1186-            send_buf, jnp.zeros_like(send_buf),
packages/core/legoesm/parallel/halo_exchange.py-1187-            nbr_rank, nbr_rank,
packages/core/legoesm/parallel/halo_exchange.py-1188-            send_tag, recv_tag, comm,
packages/core/legoesm/parallel/halo_exchange.py-1189-        )
--
packages/core/legoesm/parallel/cube_face_scatter.py-25-   halo-pads the STATE fields and applies LOCAL-face metrics (indexed at their
packages/core/legoesm/parallel/cube_face_scatter.py-26-   interior positions) to the haloed state.  No operator reads a metric at a
packages/core/legoesm/parallel/cube_face_scatter.py-27-   halo cell, so metrics need slicing, NOT halo exchange.
packages/core/legoesm/parallel/cube_face_scatter.py-28-2. **The scattered face-only halo already works.** ``_pad_halo_mpi_face_only``
packages/core/legoesm/parallel/cube_face_scatter.py-29-   keys off ``data.shape[0]`` and supports ``shape[0] == len(local_face_ids)``
packages/core/legoesm/parallel/cube_face_scatter.py:30:   for scalar / 4D / vector / interp_offsets / AD (custom_vjp sendrecv).
packages/core/legoesm/parallel/cube_face_scatter.py-31-3. **interp_offsets and duogrid stay FULL ``(6, ...)``.** The halo machinery
packages/core/legoesm/parallel/cube_face_scatter.py-32-   indexes them by the GLOBAL face id (``interp_offsets[global_face, edge]``),
packages/core/legoesm/parallel/cube_face_scatter.py-33-   so they must NOT be sliced — see ``_GLOBAL_FACE_INDEXED_FIELDS``.  This is
packages/core/legoesm/parallel/cube_face_scatter.py-34-   the one field-selectivity subtlety that a naive "slice anything with
packages/core/legoesm/parallel/cube_face_scatter.py-35-   ``shape[0]==6``" would get wrong.
--
packages/core/legoesm/parallel/latlon_mpi.py-40-
packages/core/legoesm/parallel/latlon_mpi.py-41-  For ``v`` we keep the duplicated boundary row across neighbours; the
packages/core/legoesm/parallel/latlon_mpi.py-42-  scatter and halo-exchange helpers handle this explicitly.
packages/core/legoesm/parallel/latlon_mpi.py-43-
packages/core/legoesm/parallel/latlon_mpi.py-44-- AD safety: halo sendrecv uses
packages/core/legoesm/parallel/latlon_mpi.py:45:  :func:`legoesm.parallel.halo_exchange.get_sendrecv_vjp` (the AD-safe
packages/core/legoesm/parallel/latlon_mpi.py-46-  wrapper around ``mpi4jax.sendrecv``).  Backward swaps source/dest as
packages/core/legoesm/parallel/latlon_mpi.py-47-  required by the reverse-mode rule.
packages/core/legoesm/parallel/latlon_mpi.py-48-
packages/core/legoesm/parallel/latlon_mpi.py-49-Status
packages/core/legoesm/parallel/latlon_mpi.py-50-------
--
packages/core/legoesm/parallel/latlon_mpi.py-79-
packages/core/legoesm/parallel/latlon_mpi.py-80-from legoesm.grids.halo_latlon import (
packages/core/legoesm/parallel/latlon_mpi.py-81-    fold_pole_rows,
packages/core/legoesm/parallel/latlon_mpi.py-82-    fold_pole_rows_3d,
packages/core/legoesm/parallel/latlon_mpi.py-83-)
packages/core/legoesm/parallel/latlon_mpi.py:84:from legoesm.parallel.halo_exchange import get_sendrecv_vjp
packages/core/legoesm/parallel/latlon_mpi.py-85-
packages/core/legoesm/parallel/latlon_mpi.py-86-
packages/core/legoesm/parallel/latlon_mpi.py-87-# ============================================================================
packages/core/legoesm/parallel/latlon_mpi.py-88-# Layout
packages/core/legoesm/parallel/latlon_mpi.py-89-# ============================================================================
--
packages/core/legoesm/parallel/latlon_mpi.py-512-    from mpi4py import MPI
packages/core/legoesm/parallel/latlon_mpi.py-513-
packages/core/legoesm/parallel/latlon_mpi.py-514-    return MPI.COMM_WORLD.Split(color=layout.proc_row, key=layout.proc_col)
packages/core/legoesm/parallel/latlon_mpi.py-515-
packages/core/legoesm/parallel/latlon_mpi.py-516-
packages/core/legoesm/parallel/latlon_mpi.py:517:# custom_vjp on SCALAR lon metadata only (NOT the whole layout): the
packages/core/legoesm/parallel/latlon_mpi.py-518-# layout's ``fold`` field can carry jax.Array permutations, which must
packages/core/legoesm/parallel/latlon_mpi.py-519-# not become custom-VJP static args (codex MAJOR 2026-06-13).  The public
packages/core/legoesm/parallel/latlon_mpi.py-520-# wrapper below extracts the scalar fields.  nondiff args = lon_start,
packages/core/legoesm/parallel/latlon_mpi.py-521-# lon_end, n_lon_global, proc_lon, row_comm (indices 1..5).
packages/core/legoesm/parallel/latlon_mpi.py:522:@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
packages/core/legoesm/parallel/latlon_mpi.py-523-def _lon_gather_full_p(local_block, lon_start, lon_end, n_lon_global,
packages/core/legoesm/parallel/latlon_mpi.py-524-                       proc_lon, row_comm):
packages/core/legoesm/parallel/latlon_mpi.py-525-    from legoesm.parallel.reductions import mpi4jax_array_result, require_mpi_stack
packages/core/legoesm/parallel/latlon_mpi.py-526-
packages/core/legoesm/parallel/latlon_mpi.py-527-    # Checked accessor (not a bare ``import mpi4jax``): runs the GPU-transport
--
packages/core/legoesm/parallel/latlon_mpi.py-922-        raise ImportError(
packages/core/legoesm/parallel/latlon_mpi.py-923-            "Lat-lon MPI halo exchange (n_ranks>1) requires mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-924-        ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-925-
packages/core/legoesm/parallel/latlon_mpi.py-926-    comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:927:    sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-928-
packages/core/legoesm/parallel/latlon_mpi.py-929-    # ---- South halo ----
packages/core/legoesm/parallel/latlon_mpi.py-930-    if layout.south_rank is not None:
packages/core/legoesm/parallel/latlon_mpi.py-931-        # Receive from southern neighbour's TOP `halo` rows (their
packages/core/legoesm/parallel/latlon_mpi.py-932-        # ``field[-halo:]``), placed as our ``[0:halo)``.  We send our
--
packages/core/legoesm/parallel/latlon_mpi.py-1035-            "Lat-lon E/W MPI halo exchange (proc_lon>1) requires "
packages/core/legoesm/parallel/latlon_mpi.py-1036-            "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1037-        ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1038-
packages/core/legoesm/parallel/latlon_mpi.py-1039-    comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1040:    sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1041-    trailing_lat = field.shape[0]
packages/core/legoesm/parallel/latlon_mpi.py-1042-    other = field.shape[2:]  # (nlev,) or ()
packages/core/legoesm/parallel/latlon_mpi.py-1043-
packages/core/legoesm/parallel/latlon_mpi.py-1044-    # PHASE-CONSTANT tags with sendtag == recvtag (one per shift
packages/core/legoesm/parallel/latlon_mpi.py-1045-    # direction, +1 for the opposite) — the AD-safe convention from
--
packages/core/legoesm/parallel/latlon_mpi.py-1142-
packages/core/legoesm/parallel/latlon_mpi.py-1143-    import mpi4jax
packages/core/legoesm/parallel/latlon_mpi.py-1144-    from mpi4py import MPI
packages/core/legoesm/parallel/latlon_mpi.py-1145-
packages/core/legoesm/parallel/latlon_mpi.py-1146-    comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1147:    sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1148-
packages/core/legoesm/parallel/latlon_mpi.py-1149-    if south_is_pole:
packages/core/legoesm/parallel/latlon_mpi.py-1150-        south = jnp.full((halo,) + trailing, south_value, field.dtype)
packages/core/legoesm/parallel/latlon_mpi.py-1151-    else:
packages/core/legoesm/parallel/latlon_mpi.py-1152-        send_s = field[:halo].reshape(-1)
--
packages/core/legoesm/parallel/latlon_mpi.py-1356-            south_halo, _ = fold_pole_rows_3d(
packages/core/legoesm/parallel/latlon_mpi.py-1357-                lon_padded, halo, negate=is_vector_v,
packages/core/legoesm/parallel/latlon_mpi.py-1358-            )
packages/core/legoesm/parallel/latlon_mpi.py-1359-    else:
packages/core/legoesm/parallel/latlon_mpi.py-1360-        # MPI sendrecv with south neighbour — exchanges the
packages/core/legoesm/parallel/latlon_mpi.py:1361:        # lon-padded boundary rows.  AD-safe via get_sendrecv_vjp.
packages/core/legoesm/parallel/latlon_mpi.py-1362-        try:
packages/core/legoesm/parallel/latlon_mpi.py-1363-            import mpi4jax
packages/core/legoesm/parallel/latlon_mpi.py-1364-            from mpi4py import MPI
packages/core/legoesm/parallel/latlon_mpi.py-1365-        except ImportError as exc:
packages/core/legoesm/parallel/latlon_mpi.py-1366-            raise ImportError(
packages/core/legoesm/parallel/latlon_mpi.py-1367-                "pad_halo_latlon_mpi: multi-rank lat halo requires "
packages/core/legoesm/parallel/latlon_mpi.py-1368-                "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1369-            ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1370-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1371:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1372-        send_bot = lon_padded[:halo].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py-1373-        recv_template = jnp.zeros_like(send_bot)
packages/core/legoesm/parallel/latlon_mpi.py-1374-        recv_south = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1375-            send_bot, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-1376-            layout.south_rank, layout.south_rank,
--
packages/core/legoesm/parallel/latlon_mpi.py-1409-            raise ImportError(
packages/core/legoesm/parallel/latlon_mpi.py-1410-                "pad_halo_latlon_mpi: multi-rank lat halo requires "
packages/core/legoesm/parallel/latlon_mpi.py-1411-                "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1412-            ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1413-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1414:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1415-        send_top = lon_padded[-halo:].reshape(-1)
packages/core/legoesm/parallel/latlon_mpi.py-1416-        recv_template = jnp.zeros_like(send_top)
packages/core/legoesm/parallel/latlon_mpi.py-1417-        recv_north = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1418-            send_top, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-1419-            layout.north_rank, layout.north_rank,
--
packages/core/legoesm/parallel/latlon_mpi.py-1466-            raise ImportError(
packages/core/legoesm/parallel/latlon_mpi.py-1467-                "Lat-lon MPI 1-D pad (interior partition cut) requires "
packages/core/legoesm/parallel/latlon_mpi.py-1468-                "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1469-            ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1470-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1471:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1472-        send_bot = interior[:halo]
packages/core/legoesm/parallel/latlon_mpi.py-1473-        recv_template = jnp.zeros_like(send_bot)
packages/core/legoesm/parallel/latlon_mpi.py-1474-        south_slab = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1475-            send_bot, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-1476-            layout.south_rank, layout.south_rank,
--
packages/core/legoesm/parallel/latlon_mpi.py-1490-            raise ImportError(
packages/core/legoesm/parallel/latlon_mpi.py-1491-                "Lat-lon MPI 1-D pad (interior partition cut) requires "
packages/core/legoesm/parallel/latlon_mpi.py-1492-                "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1493-            ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1494-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1495:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1496-        send_top = interior[-halo:]
packages/core/legoesm/parallel/latlon_mpi.py-1497-        recv_template = jnp.zeros_like(send_top)
packages/core/legoesm/parallel/latlon_mpi.py-1498-        north_slab = sendrecv(
packages/core/legoesm/parallel/latlon_mpi.py-1499-            send_top, recv_template,
packages/core/legoesm/parallel/latlon_mpi.py-1500-            layout.north_rank, layout.north_rank,
--
packages/core/legoesm/parallel/latlon_mpi.py-1722-    the lat axis is touched (the pencil's lon halo is a separate
packages/core/legoesm/parallel/latlon_mpi.py-1723-    dispatched exchange) — the same contract as the single-field
packages/core/legoesm/parallel/latlon_mpi.py-1724-    :func:`pad_with_pole_bc_lat_2d`, which this is value-identical to
packages/core/legoesm/parallel/latlon_mpi.py-1725-    (both fill pole rows with the constants and sendrecv interior cuts;
packages/core/legoesm/parallel/latlon_mpi.py-1726-    concatenate/slice carry native VJPs and the exchange is the shared
packages/core/legoesm/parallel/latlon_mpi.py:1727:    AD-safe :func:`get_sendrecv_vjp`).
packages/core/legoesm/parallel/latlon_mpi.py-1728-
packages/core/legoesm/parallel/latlon_mpi.py-1729-    Guard: the halo must fit the SMALLEST local lat block, the same
packages/core/legoesm/parallel/latlon_mpi.py-1730-    condition :func:`_pad_lat_wall_2d` enforces — a neighbour owning
packages/core/legoesm/parallel/latlon_mpi.py-1731-    fewer rows would send a mismatched slab and hang.
packages/core/legoesm/parallel/latlon_mpi.py-1732-    """
--
packages/core/legoesm/parallel/latlon_mpi.py-1778-    if layout.south_rank is not None or layout.north_rank is not None:
packages/core/legoesm/parallel/latlon_mpi.py-1779-        import mpi4jax
packages/core/legoesm/parallel/latlon_mpi.py-1780-        from mpi4py import MPI
packages/core/legoesm/parallel/latlon_mpi.py-1781-
packages/core/legoesm/parallel/latlon_mpi.py-1782-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1783:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1784-        # dtype groups in first-appearance order — trace-deterministic, so
packages/core/legoesm/parallel/latlon_mpi.py-1785-        # every rank issues the same fused schedule (pairing depends on it).
packages/core/legoesm/parallel/latlon_mpi.py-1786-        groups: dict = {}
packages/core/legoesm/parallel/latlon_mpi.py-1787-        for i, f in enumerate(fields):
packages/core/legoesm/parallel/latlon_mpi.py-1788-            groups.setdefault(jnp.dtype(f.dtype), []).append(i)
--
packages/core/legoesm/parallel/latlon_mpi.py-1843-    pole-folds at boundary ranks and then overwrites the boundary slabs
packages/core/legoesm/parallel/latlon_mpi.py-1844-    with the constants, so skipping the fold and filling constants
packages/core/legoesm/parallel/latlon_mpi.py-1845-    directly produces the same result with less local compute.
packages/core/legoesm/parallel/latlon_mpi.py-1846-
packages/core/legoesm/parallel/latlon_mpi.py-1847-    AD-safe: the fused buffer goes through the same
packages/core/legoesm/parallel/latlon_mpi.py:1848:    :func:`get_sendrecv_vjp` custom-vjp as the single-field path;
packages/core/legoesm/parallel/latlon_mpi.py-1849-    ``concatenate``/``slice`` carry native JAX VJPs.
packages/core/legoesm/parallel/latlon_mpi.py-1850-
packages/core/legoesm/parallel/latlon_mpi.py-1851-    Returns a tuple of padded arrays, in input order.
packages/core/legoesm/parallel/latlon_mpi.py-1852-    """
packages/core/legoesm/parallel/latlon_mpi.py-1853-    fields = tuple(fields)
--
packages/core/legoesm/parallel/latlon_mpi.py-1909-            raise ImportError(
packages/core/legoesm/parallel/latlon_mpi.py-1910-                "Lat-lon fused MPI halo exchange (n_ranks>1) requires "
packages/core/legoesm/parallel/latlon_mpi.py-1911-                "mpi4jax and mpi4py."
packages/core/legoesm/parallel/latlon_mpi.py-1912-            ) from exc
packages/core/legoesm/parallel/latlon_mpi.py-1913-        comm = MPI.COMM_WORLD
packages/core/legoesm/parallel/latlon_mpi.py:1914:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/parallel/latlon_mpi.py-1915-
packages/core/legoesm/parallel/latlon_mpi.py-1916-        # Group field indices by dtype in first-appearance order — the
packages/core/legoesm/parallel/latlon_mpi.py-1917-        # order is trace-deterministic, so every rank issues the same
packages/core/legoesm/parallel/latlon_mpi.py-1918-        # fused-message schedule (sendrecv pairing relies on it).
packages/core/legoesm/parallel/latlon_mpi.py-1919-        groups: dict = {}
--
packages/core/legoesm/grids/halo.py-2503-    The function leaves non-owned face boundary values UNCHANGED in
packages/core/legoesm/grids/halo.py-2504-    the returned arrays — only owned faces get the averaged result.
packages/core/legoesm/grids/halo.py-2505-    Callers comparing across local/MPI must compare owned faces only
packages/core/legoesm/grids/halo.py-2506-    (the MPI-replicated-mode contract).
packages/core/legoesm/grids/halo.py-2507-    """
packages/core/legoesm/grids/halo.py:2508:    from legoesm.parallel.halo_exchange import get_sendrecv_vjp
packages/core/legoesm/grids/halo.py-2509-    from collections import defaultdict
packages/core/legoesm/grids/halo.py-2510-    try:
packages/core/legoesm/grids/halo.py-2511-        import mpi4jax
packages/core/legoesm/grids/halo.py-2512-        from mpi4py import MPI as _MPI
packages/core/legoesm/grids/halo.py-2513-    except ImportError as exc:
packages/core/legoesm/grids/halo.py-2514-        raise ImportError(
packages/core/legoesm/grids/halo.py-2515-            "MPI synchronize_cgrid_fluxes requires mpi4jax + mpi4py."
packages/core/legoesm/grids/halo.py-2516-        ) from exc
packages/core/legoesm/grids/halo.py:2517:    sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/grids/halo.py-2518-    comm = _MPI.COMM_WORLD
packages/core/legoesm/grids/halo.py-2519-    rank = topology.rank
packages/core/legoesm/grids/halo.py-2520-
packages/core/legoesm/grids/halo.py-2521-    # Classify each (owned face, edge) as local or remote.
packages/core/legoesm/grids/halo.py-2522-    local_edges = []
--
packages/core/legoesm/timestepping/tridiagonal.py-24-import jax.numpy as jnp
packages/core/legoesm/timestepping/tridiagonal.py-25-
packages/core/legoesm/timestepping/tridiagonal.py-26-_TINY = float(jnp.finfo(jnp.float32).tiny)  # Smallest normal float32 (~1.18e-38)
packages/core/legoesm/timestepping/tridiagonal.py-27-
packages/core/legoesm/timestepping/tridiagonal.py-28-
packages/core/legoesm/timestepping/tridiagonal.py:29:@jax.custom_vjp
packages/core/legoesm/timestepping/tridiagonal.py-30-def thomas_solve(
packages/core/legoesm/timestepping/tridiagonal.py-31-    a: jax.Array,
packages/core/legoesm/timestepping/tridiagonal.py-32-    b: jax.Array,
packages/core/legoesm/timestepping/tridiagonal.py-33-    c: jax.Array,
packages/core/legoesm/timestepping/tridiagonal.py-34-    d: jax.Array,
--
packages/core/legoesm/timestepping/tridiagonal.py-54-    Notes
packages/core/legoesm/timestepping/tridiagonal.py-55-    -----
packages/core/legoesm/timestepping/tridiagonal.py-56-    The leading dimensions ``...`` are batched over (column-parallel).
packages/core/legoesm/timestepping/tridiagonal.py-57-    The last dimension is the tridiagonal system size.
packages/core/legoesm/timestepping/tridiagonal.py-58-
packages/core/legoesm/timestepping/tridiagonal.py:59:    Differentiability: a ``custom_vjp`` supplies the *analytic* adjoint of the
packages/core/legoesm/timestepping/tridiagonal.py-60-    linear solve (differentiate the solution, not the algorithm).  The reverse
packages/core/legoesm/timestepping/tridiagonal.py-61-    mode of the raw Thomas recursion divides by ``denom**2`` per pivot, so a
packages/core/legoesm/timestepping/tridiagonal.py-62-    pivot clamped to ``_TINY`` (a near-singular system — e.g. an implicit
packages/core/legoesm/timestepping/tridiagonal.py-63-    soil-thermal matrix whose heat capacity collapses) makes the *forward*
packages/core/legoesm/timestepping/tridiagonal.py-64-    finite but the *backward* overflow to NaN.  The adjoint here instead solves
--
packages/core/legoesm/timestepping/tridiagonal.py-71-        clamps a (near-singular) pivot to ``_TINY`` it is a surrogate, not the
packages/core/legoesm/timestepping/tridiagonal.py-72-        exact derivative of the clamped map — that is the whole point (the exact
packages/core/legoesm/timestepping/tridiagonal.py-73-        derivative is the NaN we are avoiding), and on a well-conditioned system
packages/core/legoesm/timestepping/tridiagonal.py-74-        it equals the raw element-wise autodiff to machine precision.
packages/core/legoesm/timestepping/tridiagonal.py-75-      * Reverse-only: ``jax.jvp``/``jacfwd`` through ``thomas_solve`` now raise
packages/core/legoesm/timestepping/tridiagonal.py:76:        (a ``custom_vjp`` defines no JVP).  No production/test path forward-diffs
packages/core/legoesm/timestepping/tridiagonal.py-77-        this solver; the LAPACK ``thomas_solve_batched`` keeps both modes.
packages/core/legoesm/timestepping/tridiagonal.py-78-      * Higher-order reverse mode recurses through this same rule (the bwd's
packages/core/legoesm/timestepping/tridiagonal.py-79-        ``λ`` solve uses the wrapper), so grad-of-grad stays clamp-protected too.
packages/core/legoesm/timestepping/tridiagonal.py-80-    """
packages/core/legoesm/timestepping/tridiagonal.py-81-    return _thomas_solve_impl(a, b, c, d)
--
packages/core/legoesm/timestepping/tridiagonal.py-175-    aw = jnp.asarray(a, work); bw = jnp.asarray(b, work); cw = jnp.asarray(c, work)
packages/core/legoesm/timestepping/tridiagonal.py-176-    xw = jnp.asarray(x, work); xbar = jnp.asarray(x_bar, work)
packages/core/legoesm/timestepping/tridiagonal.py-177-
packages/core/legoesm/timestepping/tridiagonal.py-178-    # Transposed system Aᵀ λ = x̄.  Aᵀ has sub-diag aT[k]=c[k-1], super-diag
packages/core/legoesm/timestepping/tridiagonal.py-179-    # cT[k]=a[k+1], same main diag b.  Solve with the SAME stable forward sweep
packages/core/legoesm/timestepping/tridiagonal.py:180:    # (custom_vjp wrapper -> stable higher-order too).
packages/core/legoesm/timestepping/tridiagonal.py-181-    zc = jnp.zeros_like(cw[..., :1])
packages/core/legoesm/timestepping/tridiagonal.py-182-    aT = jnp.concatenate([zc, cw[..., :-1]], axis=-1)
packages/core/legoesm/timestepping/tridiagonal.py-183-    cT = jnp.concatenate([aw[..., 1:], jnp.zeros_like(aw[..., :1])], axis=-1)
packages/core/legoesm/timestepping/tridiagonal.py-184-    lam = thomas_solve(aT, bw, cT, xbar)
packages/core/legoesm/timestepping/tridiagonal.py-185-
--
packages/core/legoesm/grids/dgrid_halo.py-870-            send_parts.append(u_strip.reshape(-1))
packages/core/legoesm/grids/dgrid_halo.py-871-            send_parts.append(v_strip.reshape(-1))
packages/core/legoesm/grids/dgrid_halo.py-872-        send_buf = jnp.concatenate(send_parts)
packages/core/legoesm/grids/dgrid_halo.py-873-
packages/core/legoesm/grids/dgrid_halo.py-874-        # Single sendrecv per neighbor rank.  Use the AD-safe
packages/core/legoesm/grids/dgrid_halo.py:875:        # ``_sendrecv_vjp`` wrapper (custom_vjp) so jax.grad flows
packages/core/legoesm/grids/dgrid_halo.py-876-        # through MPI sendrecv (raw mpi4jax.sendrecv chokes on the
packages/core/legoesm/grids/dgrid_halo.py-877-        # symbolic Zero cotangent JAX emits during backward).
packages/core/legoesm/grids/dgrid_halo.py:878:        from legoesm.parallel.halo_exchange import get_sendrecv_vjp
packages/core/legoesm/grids/dgrid_halo.py:879:        sendrecv = get_sendrecv_vjp(mpi4jax)
packages/core/legoesm/grids/dgrid_halo.py-880-        recv_buf = sendrecv(
packages/core/legoesm/grids/dgrid_halo.py-881-            send_buf, jnp.zeros_like(send_buf),
packages/core/legoesm/grids/dgrid_halo.py-882-            nbr_rank, nbr_rank,
packages/core/legoesm/grids/dgrid_halo.py-883-            rank, nbr_rank, comm,
packages/core/legoesm/grids/dgrid_halo.py-884-        )
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-458-      ``f_v[0] = f[0]``, ``f_v[-1] = f[-1]``.
packages/core/legoesm/grids/operators_latlon_cgrid.py-459-    * **Band MPI with at least one interior cut** — pads ``f`` by one
packages/core/legoesm/grids/operators_latlon_cgrid.py-460-      latitude row through the backend-dispatched
packages/core/legoesm/grids/operators_latlon_cgrid.py-461-      :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat` (interior
packages/core/legoesm/grids/operators_latlon_cgrid.py-462-      cuts receive the neighbour rank's row via the AD-safe
packages/core/legoesm/grids/operators_latlon_cgrid.py:463:      ``_sendrecv_vjp`` sendrecv; a pole-touching end receives a
packages/core/legoesm/grids/operators_latlon_cgrid.py-464-      constant ghost that never reaches the output, see below),
packages/core/legoesm/grids/operators_latlon_cgrid.py-465-      computes every face with the interior average on the padded
packages/core/legoesm/grids/operators_latlon_cgrid.py-466-      array, then restores the legacy edge copy at any pole-touching
packages/core/legoesm/grids/operators_latlon_cgrid.py-467-      end so pole-touching ranks reproduce serial bit-for-bit.
packages/core/legoesm/grids/operators_latlon_cgrid.py-468-
--
packages/core/legoesm/parallel/halo_exchange.py-143-# tag value = 3 * 100_000 + 99_999 = 399_999, well within 2^31-1.
packages/core/legoesm/parallel/halo_exchange.py-144-_MPI_TAG_RANK_STRIDE = 100_000
packages/core/legoesm/parallel/halo_exchange.py-145-
packages/core/legoesm/parallel/halo_exchange.py-146-
packages/core/legoesm/parallel/halo_exchange.py-147-# ======================================================================
packages/core/legoesm/parallel/halo_exchange.py:148:# AD-safe sendrecv wrapper (custom_vjp)
packages/core/legoesm/parallel/halo_exchange.py-149-# ======================================================================
packages/core/legoesm/parallel/halo_exchange.py-150-
packages/core/legoesm/parallel/halo_exchange.py:151:def _make_sendrecv_vjp(mpi4jax_mod):
packages/core/legoesm/parallel/halo_exchange.py-152-    """Build an AD-safe sendrecv once mpi4jax is imported.
packages/core/legoesm/parallel/halo_exchange.py-153-
packages/core/legoesm/parallel/halo_exchange.py-154-    mpi4jax.sendrecv has a transpose rule that swaps source/dest, but
packages/core/legoesm/parallel/halo_exchange.py-155-    the XLA lowering raises RuntimeError when ``_must_transpose=True``.
packages/core/legoesm/parallel/halo_exchange.py:156:    This wrapper bypasses that by using ``@jax.custom_vjp``: the forward
packages/core/legoesm/parallel/halo_exchange.py-157-    calls sendrecv normally, and the backward calls sendrecv with
packages/core/legoesm/parallel/halo_exchange.py-158-    swapped endpoints as a fresh forward call (no transpose flag).
packages/core/legoesm/parallel/halo_exchange.py-159-
packages/core/legoesm/parallel/halo_exchange.py-160-    Non-JAX arguments (source, dest, sendtag, recvtag, comm) are
packages/core/legoesm/runtime/devices.py-112-            )
packages/core/legoesm/runtime/devices.py-113-        if grid_type == "latlon":
packages/core/legoesm/runtime/devices.py-114-            # Lat-lon band MPI: build a LatLonBandLayout and activate
packages/core/legoesm/runtime/devices.py:115:            # set_halo_backend("mpi", layout).  See
packages/core/legoesm/runtime/devices.py-116-            # legoesm.parallel.distributed.initialize_distributed_latlon.
packages/core/legoesm/runtime/devices.py-117-            # Returns a *layout* (not a DeviceConfig) — we wrap it
packages/core/legoesm/runtime/devices.py-118-            # in a minimal DeviceConfig for ModelDriver's
--
packages/core/legoesm/parallel/latlon_mpi.py-318-    )
packages/core/legoesm/parallel/latlon_mpi.py-319-
packages/core/legoesm/parallel/latlon_mpi.py-320-
packages/core/legoesm/parallel/latlon_mpi.py:321:class LatLon2DLayout(NamedTuple):
packages/core/legoesm/parallel/latlon_mpi.py-322-    """2-D pencil (lat × lon) decomposition layout for MPI.
packages/core/legoesm/parallel/latlon_mpi.py-323-
packages/core/legoesm/parallel/latlon_mpi.py-324-    Increment 2 of the lat-lon 2-D decomposition
--
packages/core/legoesm/parallel/latlon_mpi.py-372-    n_lat: int,
packages/core/legoesm/parallel/latlon_mpi.py-373-    n_lon: int,
packages/core/legoesm/parallel/latlon_mpi.py-374-    fold: "FoldDescriptor | None" = None,
packages/core/legoesm/parallel/latlon_mpi.py:375:) -> LatLon2DLayout:
packages/core/legoesm/parallel/latlon_mpi.py-376-    """Build a 2-D pencil decomposition layout for ``rank``.
packages/core/legoesm/parallel/latlon_mpi.py-377-
packages/core/legoesm/parallel/latlon_mpi.py-378-    ``rank = proc_row * proc_lon + proc_col`` (row-major).  Latitude is
--
packages/core/legoesm/parallel/latlon_mpi.py-413-    west_rank = _rank_at(proc_row, (proc_col - 1) % proc_lon)
packages/core/legoesm/parallel/latlon_mpi.py-414-    east_rank = _rank_at(proc_row, (proc_col + 1) % proc_lon)
packages/core/legoesm/parallel/latlon_mpi.py-415-
packages/core/legoesm/parallel/latlon_mpi.py:416:    return LatLon2DLayout(
packages/core/legoesm/parallel/latlon_mpi.py-417-        rank=rank, n_ranks=n_ranks, proc_lat=proc_lat, proc_lon=proc_lon,
packages/core/legoesm/parallel/latlon_mpi.py-418-        proc_row=proc_row, proc_col=proc_col,
packages/core/legoesm/parallel/latlon_mpi.py-419-        n_lat_global=n_lat, n_lon_global=n_lon,
--
packages/core/legoesm/parallel/latlon_mpi.py-426-
packages/core/legoesm/parallel/latlon_mpi.py-427-
packages/core/legoesm/parallel/latlon_mpi.py-428-def scatter_field_latlon_2d(
packages/core/legoesm/parallel/latlon_mpi.py:429:    global_field: jax.Array, layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-430-) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-431-    """Slice the rank's 2-D block ``[lat_start:lat_end, lon_start:lon_end]``
packages/core/legoesm/parallel/latlon_mpi.py-432-    from a global (n_lat, n_lon[, ...]) field.  Deterministic slice (every
--
packages/core/legoesm/parallel/latlon_mpi.py-438-
packages/core/legoesm/parallel/latlon_mpi.py-439-
packages/core/legoesm/parallel/latlon_mpi.py-440-def gather_field_latlon_2d(
packages/core/legoesm/parallel/latlon_mpi.py:441:    local_field, layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-442-    *, is_u_face: bool = False, is_v_face: bool = False,
packages/core/legoesm/parallel/latlon_mpi.py-443-):
packages/core/legoesm/parallel/latlon_mpi.py-444-    """Reassemble the global field from every rank's 2-D block (I/O only).
--
packages/core/legoesm/parallel/latlon_mpi.py-499-    return out
packages/core/legoesm/parallel/latlon_mpi.py-500-
packages/core/legoesm/parallel/latlon_mpi.py-501-
packages/core/legoesm/parallel/latlon_mpi.py:502:def make_lon_row_comm(layout: LatLon2DLayout):
packages/core/legoesm/parallel/latlon_mpi.py-503-    """Create the longitude-ring sub-communicator for this rank's
packages/core/legoesm/parallel/latlon_mpi.py-504-    ``proc_row`` (the ``proc_lon`` ranks that share a latitude band).
packages/core/legoesm/parallel/latlon_mpi.py-505-
--
packages/core/legoesm/parallel/latlon_mpi.py-576-_lon_gather_full_p.defvjp(_lon_gather_full_p_fwd, _lon_gather_full_p_bwd)
packages/core/legoesm/parallel/latlon_mpi.py-577-
packages/core/legoesm/parallel/latlon_mpi.py-578-
packages/core/legoesm/parallel/latlon_mpi.py:579:def lon_gather_full(local_block: jax.Array, layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-580-                    row_comm) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-581-    """In-trace lat-pencil transpose: assemble the FULL longitude axis
packages/core/legoesm/parallel/latlon_mpi.py-582-    for this rank's latitude band from the ``proc_lon`` lon-ring blocks.
--
packages/core/legoesm/parallel/latlon_mpi.py-613-
packages/core/legoesm/parallel/latlon_mpi.py-614-
packages/core/legoesm/parallel/latlon_mpi.py-615-def lon_scatter_full(full_field: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py:616:                     layout: LatLon2DLayout) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-617-    """Slice this rank's longitude block ``[lon_start:lon_end]`` from a
packages/core/legoesm/parallel/latlon_mpi.py-618-    full-longitude field — inverse of :func:`lon_gather_full`."""
packages/core/legoesm/parallel/latlon_mpi.py-619-    return full_field[:, layout.lon_start:layout.lon_end]
--
packages/core/legoesm/parallel/latlon_mpi.py-984-    just the ring topology (the rank owning the last lon block has the
packages/core/legoesm/parallel/latlon_mpi.py-985-    rank owning the first as its east neighbour).  This is a STANDALONE
packages/core/legoesm/parallel/latlon_mpi.py-986-    primitive (takes plain neighbour ranks, not a layout) so the future
packages/core/legoesm/parallel/latlon_mpi.py:987:    ``LatLon2DLayout`` can call it with ``layout.west_rank`` etc.; it is
packages/core/legoesm/parallel/latlon_mpi.py-988-    NOT yet wired into any step.
packages/core/legoesm/parallel/latlon_mpi.py-989-
packages/core/legoesm/parallel/latlon_mpi.py-990-    When the longitude ring has a single member (``west_rank == east_rank
--
packages/core/legoesm/parallel/latlon_mpi.py-1098-
packages/core/legoesm/parallel/latlon_mpi.py-1099-def _pad_lat_wall_2d(
packages/core/legoesm/parallel/latlon_mpi.py-1100-    field: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py:1101:    layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-1102-    halo: int,
packages/core/legoesm/parallel/latlon_mpi.py-1103-    south_value: float,
packages/core/legoesm/parallel/latlon_mpi.py-1104-    north_value: float,
--
packages/core/legoesm/parallel/latlon_mpi.py-1171-
packages/core/legoesm/parallel/latlon_mpi.py-1172-def pad_with_pole_bc_lat_2d(
packages/core/legoesm/parallel/latlon_mpi.py-1173-    interior: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py:1174:    layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-1175-    halo: int = 1,
packages/core/legoesm/parallel/latlon_mpi.py-1176-    south_value: float = 0.0,
packages/core/legoesm/parallel/latlon_mpi.py-1177-    north_value: float = 0.0,
--
packages/core/legoesm/parallel/latlon_mpi.py-1185-    wall-BC pad is lat-only; the 2-D pencil keeps that contract and the
packages/core/legoesm/parallel/latlon_mpi.py-1186-    operator adds lon ghosts through its own dispatched lon halo.  Routed
packages/core/legoesm/parallel/latlon_mpi.py-1187-    here from ``halo_latlon.pad_with_pole_bc_lat`` when the active topology
packages/core/legoesm/parallel/latlon_mpi.py:1188:    is a :class:`LatLon2DLayout` (wall poles only; the tripolar north fold /
packages/core/legoesm/parallel/latlon_mpi.py-1189-    vector-u seam is excluded upstream).  AD-safe via the shared sendrecv
packages/core/legoesm/parallel/latlon_mpi.py-1190-    VJP.
packages/core/legoesm/parallel/latlon_mpi.py-1191-    """
--
packages/core/legoesm/parallel/latlon_mpi.py-1196-
packages/core/legoesm/parallel/latlon_mpi.py-1197-def pad_halo_latlon_2d(
packages/core/legoesm/parallel/latlon_mpi.py-1198-    field: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py:1199:    layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-1200-    halo: int = 1,
packages/core/legoesm/parallel/latlon_mpi.py-1201-    pole_bc: str = "wall",
packages/core/legoesm/parallel/latlon_mpi.py-1202-    south_value: float = 0.0,
--
packages/core/legoesm/parallel/latlon_mpi.py-1709-
packages/core/legoesm/parallel/latlon_mpi.py-1710-def pad_with_pole_bc_lat_multi_2d(
packages/core/legoesm/parallel/latlon_mpi.py-1711-    fields,
packages/core/legoesm/parallel/latlon_mpi.py:1712:    layout: "LatLon2DLayout",
packages/core/legoesm/parallel/latlon_mpi.py-1713-    halo: int = 1,
packages/core/legoesm/parallel/latlon_mpi.py-1714-    south_values=None,
packages/core/legoesm/parallel/latlon_mpi.py-1715-    north_values=None,
--
packages/core/legoesm/parallel/latlon_mpi.py-2011-    )
packages/core/legoesm/parallel/latlon_mpi.py-2012-
packages/core/legoesm/parallel/latlon_mpi.py-2013-
packages/core/legoesm/parallel/latlon_mpi.py:2014:def scatter_state_latlon_2d(state, layout: LatLon2DLayout):
packages/core/legoesm/parallel/latlon_mpi.py-2015-    """Extract the rank-local 2-D block from a global C-grid lat-lon state.
packages/core/legoesm/parallel/latlon_mpi.py-2016-
packages/core/legoesm/parallel/latlon_mpi.py-2017-    The 2-D analog of :func:`scatter_state_latlon`: slice BOTH the latitude
--
packages/core/legoesm/parallel/latlon_mpi.py-2573-
packages/core/legoesm/parallel/latlon_mpi.py-2574-
packages/core/legoesm/parallel/latlon_mpi.py-2575-def slice_latlon_grid_to_block_2d(
packages/core/legoesm/parallel/latlon_mpi.py:2576:    grid, layout: LatLon2DLayout, *, skip_total_area_reduce: bool = False,
packages/core/legoesm/parallel/latlon_mpi.py-2577-):
packages/core/legoesm/parallel/latlon_mpi.py-2578-    """Slice a global ``LatLonGrid`` to this rank's 2-D lat x lon block.
packages/core/legoesm/parallel/latlon_mpi.py-2579-
--
packages/core/legoesm/parallel/latlon_mpi.py-2791-
packages/core/legoesm/parallel/latlon_mpi.py-2792-    Architecture
packages/core/legoesm/parallel/latlon_mpi.py-2793-    ------------
packages/core/legoesm/parallel/latlon_mpi.py:2794:    1. ``set_halo_backend("mpi", layout)`` is activated once at
packages/core/legoesm/parallel/latlon_mpi.py-2795-       factory time.  Every subsequent ``pad_halo_latlon*``,
packages/core/legoesm/parallel/latlon_mpi.py-2796-       ``pad_ns_zero``, and ``pad_with_pole_bc_lat`` call inside the
packages/core/legoesm/parallel/latlon_mpi.py-2797-       dycore operators dispatches through the MPI sendrecv path.
--
packages/core/legoesm/parallel/latlon_mpi.py-2866-
packages/core/legoesm/parallel/latlon_mpi.py-2867-    Side effect
packages/core/legoesm/parallel/latlon_mpi.py-2868-    -----------
packages/core/legoesm/parallel/latlon_mpi.py:2869:    Activates ``set_halo_backend("mpi", layout)`` at the global
packages/core/legoesm/parallel/latlon_mpi.py-2870-    module level.  Caller is responsible for resetting via
packages/core/legoesm/parallel/latlon_mpi.py-2871-    ``set_halo_backend("local")`` once the run is finished — see
packages/core/legoesm/parallel/latlon_mpi.py-2872-    ``ModelDriver._finalize_run`` for the cubed-sphere precedent.
--
packages/core/legoesm/parallel/latlon_mpi.py-2901-    # through inter-rank sendrecv at partition cuts and constant /
packages/core/legoesm/parallel/latlon_mpi.py-2902-    # pole-fold pads at boundary ranks.  This also flips
packages/core/legoesm/parallel/latlon_mpi.py-2903-    # ``is_distributed()`` so the mass-fixer reductions allreduce.
packages/core/legoesm/parallel/latlon_mpi.py:2904:    set_halo_backend("mpi", layout)
packages/core/legoesm/parallel/latlon_mpi.py-2905-    from legoesm.grids.polar_filter import (
packages/core/legoesm/parallel/latlon_mpi.py-2906-        get_polar_filter_lon_gather,
packages/core/legoesm/parallel/latlon_mpi.py-2907-        set_polar_filter_lon_gather,
--
packages/core/legoesm/parallel/latlon_mpi.py-3010-
packages/core/legoesm/parallel/latlon_mpi.py-3011-def make_latlon_2d_mpi_step(
packages/core/legoesm/parallel/latlon_mpi.py-3012-    model,
packages/core/legoesm/parallel/latlon_mpi.py:3013:    layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-3014-    *,
packages/core/legoesm/parallel/latlon_mpi.py-3015-    physics_fn: Callable | None = None,
packages/core/legoesm/parallel/latlon_mpi.py-3016-) -> Callable:
--
packages/core/legoesm/parallel/latlon_mpi.py-3018-
packages/core/legoesm/parallel/latlon_mpi.py-3019-    The 2-D ``(proc_lat × proc_lon)`` analogue of :func:`make_latlon_mpi_step`.
packages/core/legoesm/parallel/latlon_mpi.py-3020-    Identical architecture — activate the MPI halo backend once with the
packages/core/legoesm/parallel/latlon_mpi.py:3021:    :class:`LatLon2DLayout`, build a rank-local model with the global
packages/core/legoesm/parallel/latlon_mpi.py-3022-    sphere area (mass-fixer divisor) and ``pole_v_bc`` tracking which lat
packages/core/legoesm/parallel/latlon_mpi.py-3023-    ends touch a physical pole, then delegate each step to
packages/core/legoesm/parallel/latlon_mpi.py-3024-    ``model._step_cgrid`` whose backend-aware operators fetch halo data via
--
packages/core/legoesm/parallel/latlon_mpi.py-3105-        physics_requires_phys_state,
packages/core/legoesm/parallel/latlon_mpi.py-3106-    )
packages/core/legoesm/parallel/latlon_mpi.py-3107-
packages/core/legoesm/parallel/latlon_mpi.py:3108:    set_halo_backend("mpi", layout)
packages/core/legoesm/parallel/latlon_mpi.py-3109-
packages/core/legoesm/parallel/latlon_mpi.py-3110-    global_total_area = global_sum_mpi(jnp.sum(model.grid.area))
packages/core/legoesm/parallel/latlon_mpi.py-3111-    mpi_grid = model.grid._replace(total_area=global_total_area)
--
packages/core/legoesm/parallel/distributed.py-271-        # tests) must not silently outlive a follow-up `initialize_distributed`.
packages/core/legoesm/parallel/distributed.py-272-        from legoesm.grids.halo import get_halo_backend, set_halo_backend
packages/core/legoesm/parallel/distributed.py-273-        if get_halo_backend() != "mpi":
packages/core/legoesm/parallel/distributed.py:274:            set_halo_backend("mpi", _active_topology)
packages/core/legoesm/parallel/distributed.py-275-        # FV3_3D iter-1058 (codex iter-1056 WARN #2): rebuild
packages/core/legoesm/parallel/distributed.py-276-        # ``_active_layout`` when the caller's ``global_n`` differs
packages/core/legoesm/parallel/distributed.py-277-        # from the first-init layout's ``global_n``.  Previously the
--
packages/core/legoesm/parallel/distributed.py-353-
packages/core/legoesm/parallel/distributed.py-354-    # Set up MPI halo exchange.
packages/core/legoesm/parallel/distributed.py-355-    from legoesm.grids.halo import set_halo_backend
packages/core/legoesm/parallel/distributed.py:356:    set_halo_backend("mpi", topology)
packages/core/legoesm/parallel/distributed.py-357-
packages/core/legoesm/parallel/distributed.py-358-    # Create device mesh with local devices.
packages/core/legoesm/parallel/distributed.py-359-    local_devices = list(jax.local_devices())
--
packages/core/legoesm/parallel/distributed.py-463-    1. Reads ``rank`` and ``n_processes`` from ``MPI.COMM_WORLD``.
packages/core/legoesm/parallel/distributed.py-464-    2. Builds a :class:`LatLonBandLayout` for the band this rank
packages/core/legoesm/parallel/distributed.py-465-       owns.
packages/core/legoesm/parallel/distributed.py:466:    3. Activates ``set_halo_backend("mpi", layout)`` — every
packages/core/legoesm/parallel/distributed.py-467-       subsequent ``pad_halo_latlon`` / ``pad_with_pole_bc_lat``
packages/core/legoesm/parallel/distributed.py-468-       call inside the dycore dispatches through MPI sendrecv at
packages/core/legoesm/parallel/distributed.py-469-       partition cuts + pole-fold / wall-BC constants at boundary
--
packages/core/legoesm/parallel/distributed.py-605-            updated = _active_topology._replace(fold=None)
packages/core/legoesm/parallel/distributed.py-606-            _active_topology = updated
packages/core/legoesm/parallel/distributed.py-607-            from legoesm.grids.halo import set_halo_backend
packages/core/legoesm/parallel/distributed.py:608:            set_halo_backend("mpi", updated)
packages/core/legoesm/parallel/distributed.py-609-            return updated
packages/core/legoesm/parallel/distributed.py-610-        if requested_on and not active_on:
packages/core/legoesm/parallel/distributed.py-611-            # A prior fold-less init must NOT mask a later tripolar (ORCA)
--
packages/core/legoesm/parallel/distributed.py-621-            updated = _active_topology._replace(fold=fold)
packages/core/legoesm/parallel/distributed.py-622-            _active_topology = updated
packages/core/legoesm/parallel/distributed.py-623-            from legoesm.grids.halo import set_halo_backend
packages/core/legoesm/parallel/distributed.py:624:            set_halo_backend("mpi", updated)
packages/core/legoesm/parallel/distributed.py-625-            return updated
packages/core/legoesm/parallel/distributed.py-626-        warnings.warn(
packages/core/legoesm/parallel/distributed.py-627-            "initialize_distributed_latlon() called more than once. "
--
packages/core/legoesm/parallel/distributed.py-650-    _active_topology = layout
packages/core/legoesm/parallel/distributed.py-651-
packages/core/legoesm/parallel/distributed.py-652-    from legoesm.grids.halo import set_halo_backend
packages/core/legoesm/parallel/distributed.py:653:    set_halo_backend("mpi", layout)
packages/core/legoesm/parallel/distributed.py-654-    return layout
packages/core/legoesm/parallel/distributed.py-655-
packages/core/legoesm/parallel/distributed.py-656-
--
packages/core/legoesm/grids/halo_latlon.py-182-
packages/core/legoesm/grids/halo_latlon.py-183-
packages/core/legoesm/grids/halo_latlon.py-184-def _dispatch_latlon_2d_fold(data, topology, halo, *, is_vector_v):
packages/core/legoesm/grids/halo_latlon.py:185:    """Fold-family ``pad_halo_latlon*`` dispatch for a ``LatLon2DLayout``.
packages/core/legoesm/grids/halo_latlon.py-186-
packages/core/legoesm/grids/halo_latlon.py-187-    ``proc_lon == 1`` is a pure latitude band (every rank owns the full lon
packages/core/legoesm/grids/halo_latlon.py-188-    circle), so the 180-deg pole fold is LOCAL — reuse the validated band
--
packages/core/legoesm/grids/halo_latlon.py-257-        # back to the local serial path rather than crashing in the
packages/core/legoesm/grids/halo_latlon.py-258-        # MPI dispatch with an opaque error.
packages/core/legoesm/grids/halo_latlon.py-259-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:260:            LatLon2DLayout, LatLonBandLayout, pad_halo_latlon_mpi,
packages/core/legoesm/grids/halo_latlon.py-261-        )
packages/core/legoesm/grids/halo_latlon.py-262-        if isinstance(topology, LatLonBandLayout):
packages/core/legoesm/grids/halo_latlon.py-263-            return pad_halo_latlon_mpi(
packages/core/legoesm/grids/halo_latlon.py-264-                data, topology, halo=halo, is_vector_v=False,
packages/core/legoesm/grids/halo_latlon.py-265-            )
packages/core/legoesm/grids/halo_latlon.py:266:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/halo_latlon.py-267-            return _dispatch_latlon_2d_fold(
packages/core/legoesm/grids/halo_latlon.py-268-                data, topology, halo, is_vector_v=False,
packages/core/legoesm/grids/halo_latlon.py-269-            )
--
packages/core/legoesm/grids/halo_latlon.py-311-    if get_halo_backend() == "mpi":
packages/core/legoesm/grids/halo_latlon.py-312-        topology = get_mpi_topology()
packages/core/legoesm/grids/halo_latlon.py-313-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:314:            LatLon2DLayout, LatLonBandLayout, pad_halo_latlon_mpi,
packages/core/legoesm/grids/halo_latlon.py-315-        )
packages/core/legoesm/grids/halo_latlon.py-316-        if isinstance(topology, LatLonBandLayout):
packages/core/legoesm/grids/halo_latlon.py-317-            return pad_halo_latlon_mpi(
packages/core/legoesm/grids/halo_latlon.py-318-                data, topology, halo=halo, is_vector_v=True,
packages/core/legoesm/grids/halo_latlon.py-319-            )
packages/core/legoesm/grids/halo_latlon.py:320:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/halo_latlon.py-321-            return _dispatch_latlon_2d_fold(
packages/core/legoesm/grids/halo_latlon.py-322-                data, topology, halo, is_vector_v=True,
packages/core/legoesm/grids/halo_latlon.py-323-            )
--
packages/core/legoesm/grids/halo_latlon.py-387-    if get_halo_backend() == "mpi":
packages/core/legoesm/grids/halo_latlon.py-388-        topology = get_mpi_topology()
packages/core/legoesm/grids/halo_latlon.py-389-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:390:            LatLon2DLayout, LatLonBandLayout, pad_halo_latlon_mpi,
packages/core/legoesm/grids/halo_latlon.py-391-        )
packages/core/legoesm/grids/halo_latlon.py-392-        if isinstance(topology, LatLonBandLayout):
packages/core/legoesm/grids/halo_latlon.py-393-            return pad_halo_latlon_mpi(
packages/core/legoesm/grids/halo_latlon.py-394-                data, topology, halo=halo, is_vector_v=False,
packages/core/legoesm/grids/halo_latlon.py-395-            )
packages/core/legoesm/grids/halo_latlon.py:396:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/halo_latlon.py-397-            return _dispatch_latlon_2d_fold(
packages/core/legoesm/grids/halo_latlon.py-398-                data, topology, halo, is_vector_v=False,
packages/core/legoesm/grids/halo_latlon.py-399-            )
--
packages/core/legoesm/grids/halo_latlon.py-420-    if get_halo_backend() == "mpi":
packages/core/legoesm/grids/halo_latlon.py-421-        topology = get_mpi_topology()
packages/core/legoesm/grids/halo_latlon.py-422-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:423:            LatLon2DLayout, LatLonBandLayout, pad_halo_latlon_mpi,
packages/core/legoesm/grids/halo_latlon.py-424-        )
packages/core/legoesm/grids/halo_latlon.py-425-        if isinstance(topology, LatLonBandLayout):
packages/core/legoesm/grids/halo_latlon.py-426-            return pad_halo_latlon_mpi(
packages/core/legoesm/grids/halo_latlon.py-427-                data, topology, halo=halo, is_vector_v=True,
packages/core/legoesm/grids/halo_latlon.py-428-            )
packages/core/legoesm/grids/halo_latlon.py:429:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/halo_latlon.py-430-            return _dispatch_latlon_2d_fold(
packages/core/legoesm/grids/halo_latlon.py-431-                data, topology, halo, is_vector_v=True,
packages/core/legoesm/grids/halo_latlon.py-432-            )
--
packages/core/legoesm/grids/halo_latlon.py-524-    if get_halo_backend() == "mpi":
packages/core/legoesm/grids/halo_latlon.py-525-        topology = get_mpi_topology()
packages/core/legoesm/grids/halo_latlon.py-526-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:527:            LatLon2DLayout, LatLonBandLayout,
packages/core/legoesm/grids/halo_latlon.py-528-        )
packages/core/legoesm/grids/halo_latlon.py-529-        # Band AND 2-D pencil share the pole-touch test: a rank zeros its
packages/core/legoesm/grids/halo_latlon.py-530-        # south end only if it OWNS the south pole (``south_rank is None``)
--
packages/core/legoesm/grids/halo_latlon.py-532-        # pencil this is the proc_row-0 / proc_row-last test (the riskiest
packages/core/legoesm/grids/halo_latlon.py-533-        # 2-D bug — an interior proc row must NOT wall its lat cut, or the
packages/core/legoesm/grids/halo_latlon.py-534-        # cross-cut gradient sendrecv'd from the neighbour is destroyed).
packages/core/legoesm/grids/halo_latlon.py:535:        if isinstance(topology, (LatLonBandLayout, LatLon2DLayout)):
packages/core/legoesm/grids/halo_latlon.py-536-            out = field
packages/core/legoesm/grids/halo_latlon.py-537-            if topology.south_rank is None:
packages/core/legoesm/grids/halo_latlon.py-538-                out = out.at[0].set(jnp.zeros_like(out[0]))
--
packages/core/legoesm/grids/halo_latlon.py-664-    # if the active backend is MPI but for a different grid, fall
packages/core/legoesm/grids/halo_latlon.py-665-    # back to the local serial pad.
packages/core/legoesm/grids/halo_latlon.py-666-    from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:667:        LatLon2DLayout,
packages/core/legoesm/grids/halo_latlon.py-668-        LatLonBandLayout,
packages/core/legoesm/grids/halo_latlon.py-669-        pad_with_pole_bc_lat_2d,
packages/core/legoesm/grids/halo_latlon.py-670-        pad_with_pole_bc_lat_mpi,
packages/core/legoesm/grids/halo_latlon.py-671-    )
packages/core/legoesm/grids/halo_latlon.py:672:    if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/halo_latlon.py-673-        # 2-D pencil: lat-axis-ONLY wall pad (interior lat cut sendrecv +
packages/core/legoesm/grids/halo_latlon.py-674-        # pole wall constant).  Longitude is left untouched — the band path
packages/core/legoesm/grids/halo_latlon.py-675-        # never split lon, so its wall-BC pad is lat-only; the 2-D path
--
packages/core/legoesm/grids/halo_latlon.py-788-    fused = os.environ.get("LEGOESM_LATLON_FUSED_HALO", "1") != "0"
packages/core/legoesm/grids/halo_latlon.py-789-    if get_halo_backend() == "mpi" and fused:
packages/core/legoesm/grids/halo_latlon.py-790-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:791:            LatLon2DLayout,
packages/core/legoesm/grids/halo_latlon.py-792-            LatLonBandLayout,
packages/core/legoesm/grids/halo_latlon.py-793-            pad_with_pole_bc_lat_multi_2d,
packages/core/legoesm/grids/halo_latlon.py-794-            pad_with_pole_bc_lat_multi_mpi,
--
packages/core/legoesm/grids/halo_latlon.py-799-                fields, topology, halo=halo,
packages/core/legoesm/grids/halo_latlon.py-800-                south_values=south_values, north_values=north_values,
packages/core/legoesm/grids/halo_latlon.py-801-            )
packages/core/legoesm/grids/halo_latlon.py:802:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/halo_latlon.py-803-            # 2-D pencil twin (codex consult #3, 2026-08-04): same
packages/core/legoesm/grids/halo_latlon.py-804-            # dtype-group fusion on the lat axis; lon stays a separate
packages/core/legoesm/grids/halo_latlon.py-805-            # dispatched exchange, as in the single-field 2-D path.
--
packages/core/legoesm/grids/halo_latlon.py-848-    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
packages/core/legoesm/grids/halo_latlon.py-849-    if get_halo_backend() == "mpi":
packages/core/legoesm/grids/halo_latlon.py-850-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:851:            LatLon2DLayout,
packages/core/legoesm/grids/halo_latlon.py-852-            LatLonBandLayout,
packages/core/legoesm/grids/halo_latlon.py-853-        )
packages/core/legoesm/grids/halo_latlon.py-854-        topology = get_mpi_topology()
packages/core/legoesm/grids/halo_latlon.py:855:        if isinstance(topology, (LatLonBandLayout, LatLon2DLayout)):
packages/core/legoesm/grids/halo_latlon.py-856-            return topology.south_rank is None, topology.north_rank is None
packages/core/legoesm/grids/halo_latlon.py-857-    return True, True
packages/core/legoesm/grids/halo_latlon.py-858-
--
packages/core/legoesm/grids/halo_latlon.py-884-    return jnp.where(north_sel, north_row, out)
packages/core/legoesm/grids/halo_latlon.py-885-
packages/core/legoesm/grids/halo_latlon.py-886-
packages/core/legoesm/grids/halo_latlon.py:887:def widen_band_cell_fields(fields, halo: int, *, clamp_poles: bool = False):
packages/core/legoesm/grids/halo_latlon.py-888-    """Widen cell-row (leading dim ``n_lat``) fields by ``halo`` rows/side.
packages/core/legoesm/grids/halo_latlon.py-889-
packages/core/legoesm/grids/halo_latlon.py-890-    ONE fused exchange under the MPI band backend
--
packages/core/legoesm/grids/halo_latlon.py-940-    )
packages/core/legoesm/grids/halo_latlon.py-941-
packages/core/legoesm/grids/halo_latlon.py-942-
packages/core/legoesm/grids/halo_latlon.py:943:def widen_cgrid_geometry_band(geom, halo: int):
packages/core/legoesm/grids/halo_latlon.py-944-    """Extended-band twin of ``slice_cgrid_geometry_to_band``: widen a
packages/core/legoesm/grids/halo_latlon.py-945-    band-local :class:`~legoesm.grids.latlon.LatLonCGridGeometry` by
packages/core/legoesm/grids/halo_latlon.py-946-    ``halo`` ghost rows per side via the ACTIVE halo backend.
--
packages/core/legoesm/grids/halo_latlon.py-969-    fold = getattr(geom, "fold", None)
packages/core/legoesm/grids/halo_latlon.py-970-    if fold is not None and getattr(fold, "is_active", False):
packages/core/legoesm/grids/halo_latlon.py-971-        raise NotImplementedError(
packages/core/legoesm/grids/halo_latlon.py:972:            "widen_cgrid_geometry_band: tripolar north fold is not "
packages/core/legoesm/grids/halo_latlon.py-973-            "supported by the wide-halo barotropic path (the fold row "
packages/core/legoesm/grids/halo_latlon.py-974-            "needs a permuted, sign-flipped wide exchange — follow-up)."
packages/core/legoesm/grids/halo_latlon.py-975-        )
--
packages/core/legoesm/grids/halo_latlon.py-985-        cell_names = cell_names + ("seam_wall_rows",)
packages/core/legoesm/grids/halo_latlon.py-986-    vface_names = ("dx_v", "dy_v", "area_q", "f_v", "cos_alpha_v",
packages/core/legoesm/grids/halo_latlon.py-987-                   "sin_alpha_v", "cos_lat_v")
packages/core/legoesm/grids/halo_latlon.py:988:    cell_wide = widen_band_cell_fields(
packages/core/legoesm/grids/halo_latlon.py-989-        tuple(getattr(geom, n) for n in cell_names), halo, clamp_poles=True)
packages/core/legoesm/grids/halo_latlon.py-990-    vface_wide = widen_band_vface_fields(
packages/core/legoesm/grids/halo_latlon.py-991-        tuple(getattr(geom, n) for n in vface_names), halo, clamp_poles=True)
--
packages/core/legoesm/grids/polar_filter.py-28-# lon SLICE, so the filter must first gather the full lon axis, rFFT/mask/irFFT
packages/core/legoesm/grids/polar_filter.py-29-# on it, then scatter its block back.  The 2-D MPI step
packages/core/legoesm/grids/polar_filter.py-30-# (``make_latlon_2d_mpi_step``) injects an AD-safe gather/scatter pair here
packages/core/legoesm/grids/polar_filter.py:31:# (bound to its ``LatLon2DLayout`` + longitude row sub-communicator) BEFORE it
packages/core/legoesm/grids/polar_filter.py-32-# builds the jitted step; the filter reads this process-global at trace time,
packages/core/legoesm/grids/polar_filter.py-33-# exactly like the halo backend (``grids.halo.get_halo_backend``).  This keeps
packages/core/legoesm/grids/polar_filter.py-34-# ``polar_filter`` free of a ``parallel.latlon_mpi`` import (no cycle) and the
--
packages/core/legoesm/parallel/halo_exchange.py-25-    from legoesm.grids.halo import set_halo_backend
packages/core/legoesm/parallel/halo_exchange.py-26-
packages/core/legoesm/parallel/halo_exchange.py-27-    topology = build_comm_topology(rank, n_processes)
packages/core/legoesm/parallel/halo_exchange.py:28:    set_halo_backend("mpi", topology)
packages/core/legoesm/parallel/halo_exchange.py-29-
packages/core/legoesm/parallel/halo_exchange.py-30-After that, all operators automatically use MPI halo exchange.
packages/core/legoesm/parallel/halo_exchange.py-31-
--
packages/core/legoesm/grids/latlon.py-1286-    # "seam_wall_rows", None)``.  APPENDED at the NamedTuple end with a
packages/core/legoesm/grids/latlon.py-1287-    # default.  As a per-lat-row (n_lat,) array it is a real pytree leaf
packages/core/legoesm/grids/latlon.py-1288-    # only when set, and the MPI/SPMD band slicers (``slice_cgrid_geometry
packages/core/legoesm/grids/latlon.py:1289:    # _to_band``, ``widen_cgrid_geometry_band``) slice/widen it like the
packages/core/legoesm/grids/latlon.py-1290-    # other T-point cell-row fields so it stays aligned with band-local
packages/core/legoesm/grids/latlon.py-1291-    # ``n_lat`` (``None`` passes through unchanged).
packages/core/legoesm/grids/latlon.py-1292-    seam_wall_rows: jax.Array | None = None
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-55-    )
packages/core/legoesm/grids/operators_latlon_cgrid.py-56-
packages/core/legoesm/grids/operators_latlon_cgrid.py-57-
packages/core/legoesm/grids/operators_latlon_cgrid.py:58:def pad_ns_zero_multi(*fields: jnp.ndarray) -> tuple:
packages/core/legoesm/grids/operators_latlon_cgrid.py-59-    """Batched :func:`pad_ns_zero` for independent same-``n_lat`` fields.
packages/core/legoesm/grids/operators_latlon_cgrid.py-60-
packages/core/legoesm/grids/operators_latlon_cgrid.py-61-    Value-identical to ``tuple(pad_ns_zero(f) for f in fields)``; under
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-276-
packages/core/legoesm/grids/operators_latlon_cgrid.py-277-    * local / band / SPMD-lat / non-2-D MPI — every rank owns the full
packages/core/legoesm/grids/operators_latlon_cgrid.py-278-      longitude circle, so the wrap is LOCAL: ``jnp.pad(mode="wrap")``.
packages/core/legoesm/grids/operators_latlon_cgrid.py:279:    * 2-D pencil (``LatLon2DLayout``) — longitude is split, so the wrap
packages/core/legoesm/grids/operators_latlon_cgrid.py-280-      becomes an MPI ring exchange with the W/E neighbour
packages/core/legoesm/grids/operators_latlon_cgrid.py-281-      (:func:`legoesm.parallel.latlon_mpi.exchange_halo_lon`).
packages/core/legoesm/grids/operators_latlon_cgrid.py-282-    * 2-D SPMD ``("lat", "lon")`` mesh (M3a) — the wrap becomes the cyclic
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-303-    if backend == "mpi":
packages/core/legoesm/grids/operators_latlon_cgrid.py-304-        topology = get_mpi_topology()
packages/core/legoesm/grids/operators_latlon_cgrid.py-305-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/operators_latlon_cgrid.py:306:            LatLon2DLayout, exchange_halo_lon,
packages/core/legoesm/grids/operators_latlon_cgrid.py-307-        )
packages/core/legoesm/grids/operators_latlon_cgrid.py:308:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/operators_latlon_cgrid.py-309-            return exchange_halo_lon(
packages/core/legoesm/grids/operators_latlon_cgrid.py-310-                f, topology.west_rank, topology.east_rank,
packages/core/legoesm/grids/operators_latlon_cgrid.py-311-                topology.rank, halo=halo,
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-399-        return None
packages/core/legoesm/grids/operators_latlon_cgrid.py-400-    topology = get_mpi_topology()
packages/core/legoesm/grids/operators_latlon_cgrid.py-401-    from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/operators_latlon_cgrid.py:402:        LatLon2DLayout, LatLonBandLayout,
packages/core/legoesm/grids/operators_latlon_cgrid.py-403-    )
packages/core/legoesm/grids/operators_latlon_cgrid.py-404-    # Band AND 2-D pencil expose the same pole-terminated lat-LINE
packages/core/legoesm/grids/operators_latlon_cgrid.py-405-    # semantics (``south_rank``/``north_rank is None`` at the physical
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-408-    # ``interp_cell_to_vface_halo`` correct on a ``proc_lat>1`` pencil rank
packages/core/legoesm/grids/operators_latlon_cgrid.py-409-    # — without it an interior lat-cut rank would clamp its band edge to a
packages/core/legoesm/grids/operators_latlon_cgrid.py-410-    # physical pole (e.g. curl_vertex's sin clamp), corrupting metrics.
packages/core/legoesm/grids/operators_latlon_cgrid.py:411:    if isinstance(topology, (LatLonBandLayout, LatLon2DLayout)) and (
packages/core/legoesm/grids/operators_latlon_cgrid.py-412-        topology.south_rank is not None
packages/core/legoesm/grids/operators_latlon_cgrid.py-413-        or topology.north_rank is not None
packages/core/legoesm/grids/operators_latlon_cgrid.py-414-    ):

exec
/usr/bin/bash -lc "rg -n -C 12 \"def get_sendrecv_vjp|sendrecv.*defvjp|def _sendrecv|sendrecv_fwd|sendrecv_bwd|custom_vjp\" packages/core/legoesm/parallel/halo_exchange.py; nl -ba packages/core/legoesm/parallel/halo_exchange.py | sed -n '1,250p'; nl -ba packages/core/legoesm/parallel/latlon_mpi.py | sed -n '1200,1285p'; nl -ba packages/core/legoesm/parallel/distributed.py | sed -n '525,675p'; rg -n -C 4 \"def make_latlon_2d_mpi_step|set_halo_backend\\(\" packages/core/legoesm/parallel/latlon_mpi.py | head -360; rg -n \"__all__\" packages/core/legoesm/parallel/latlon_mpi.py packages/core/legoesm/grids/halo_latlon.py packages/core/legoesm/parallel/__init__.py packages/core/legoesm/grids/__init__.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 439ms:
136-# most implementations support up to 2^31-1).
137-#
138-# Old scheme ``edge * 1000 + rank`` collides when rank >= 1000 because
139-# the edge term (0-3) * 1000 overlaps with the rank offset.
140-#
141-# New scheme: ``edge * _MPI_TAG_RANK_STRIDE + rank`` with a stride of
142-# 100_000, supporting up to 99_999 ranks without collision.  Maximum
143-# tag value = 3 * 100_000 + 99_999 = 399_999, well within 2^31-1.
144-_MPI_TAG_RANK_STRIDE = 100_000
145-
146-
147-# ======================================================================
148:# AD-safe sendrecv wrapper (custom_vjp)
149-# ======================================================================
150-
151-def _make_sendrecv_vjp(mpi4jax_mod):
152-    """Build an AD-safe sendrecv once mpi4jax is imported.
153-
154-    mpi4jax.sendrecv has a transpose rule that swaps source/dest, but
155-    the XLA lowering raises RuntimeError when ``_must_transpose=True``.
156:    This wrapper bypasses that by using ``@jax.custom_vjp``: the forward
157-    calls sendrecv normally, and the backward calls sendrecv with
158-    swapped endpoints as a fresh forward call (no transpose flag).
159-
160-    Non-JAX arguments (source, dest, sendtag, recvtag, comm) are
161-    declared via ``nondiff_argnums`` so JAX does not attempt to trace
162-    them.  They are passed through to fwd/bwd as leading static args.
163-    """
164-
165:    def _sendrecv_impl(send_buf, recv_template, source, dest,
166-                       sendtag, recvtag, comm):
167-        return mpi4jax_array_result(
168-            mpi4jax_mod.sendrecv(
169-                send_buf, recv_template,
170-                source=source, dest=dest,
171-                sendtag=sendtag, recvtag=recvtag, comm=comm,
172-            )
173-        )
174-
175:    _sendrecv = jax.custom_vjp(
176-        _sendrecv_impl, nondiff_argnums=(2, 3, 4, 5, 6),
177-    )
178-
179-    def _fwd(send_buf, recv_template, source, dest, sendtag, recvtag, comm):
180-        # fwd has the same signature as the primal function.
181-        result = mpi4jax_array_result(
182-            mpi4jax_mod.sendrecv(
183-                send_buf, recv_template,
184-                source=source, dest=dest,
185-                sendtag=sendtag, recvtag=recvtag, comm=comm,
186-            )
187-        )
--
208-        # receive back carries the peer's cotangent for OUR send_buf, so
209-        # the template must be SEND-shaped.
210-        (send_buf,) = res
211-        d_send = mpi4jax_array_result(
212-            mpi4jax_mod.sendrecv(
213-                g, jnp.zeros_like(send_buf),
214-                source=dest, dest=source,
215-                sendtag=sendtag, recvtag=recvtag, comm=comm,
216-            )
217-        )
218-        return d_send, jnp.zeros_like(g)
219-
220:    _sendrecv.defvjp(_fwd, _bwd)
221-    return _sendrecv
222-
223-
224-# Module-level cache: built lazily on first MPI import.
225-_sendrecv_vjp_fn = None
226-
227-
228:def get_sendrecv_vjp(mpi4jax_mod):
229-    """Return the cached AD-safe sendrecv wrapper.
230-
231-    This is the single choke point every halo ``sendrecv`` routes through, so
232-    it is where the mpi4jax GPU-transport preflight runs — on EVERY call, BEFORE
233-    the (memoised) wrapper is returned. Halo entry points reach mpi4jax by a
234-    direct ``import mpi4jax`` rather than :func:`require_mpi_stack`, so without
235-    this call a GPU-direct misconfiguration (``MPI4JAX_USE_CUDA_MPI=1`` against a
236-    non-CUDA mpi4jax) would slip past the preflight and segfault at the first
237-    device-buffer exchange.
238-    """
239-    check_mpi4jax_transport(mpi4jax_mod)
240-    global _sendrecv_vjp_fn
     1	"""Distributed halo exchange using mpi4jax.
     2	
     3	Provides MPI-based halo exchange for multi-node cubed-sphere runs.
     4	The single-node version (using JAX's automatic SPMD) is in
     5	:mod:`legoesm.grids.halo` and requires no changes.
     6	
     7	This module is only imported when MPI is active.  ``mpi4jax`` is
     8	an optional dependency.
     9	
    10	Supports two modes:
    11	
    12	1. **Face-only** (≤6 MPI ranks): each rank owns one or more full faces.
    13	   Halo exchange happens between faces on different ranks.
    14	
    15	2. **Sub-face tiling** (6 × k² ranks): each rank owns one tile of one
    16	   face.  Halo exchange happens both between tiles on the same face
    17	   (intra-face) and between tiles on adjacent faces (inter-face).
    18	
    19	Usage
    20	-----
    21	The halo exchange is set up once at initialization via
    22	:func:`legoesm.grids.halo.set_halo_backend`::
    23	
    24	    from legoesm.parallel.comm import build_comm_topology
    25	    from legoesm.grids.halo import set_halo_backend
    26	
    27	    topology = build_comm_topology(rank, n_processes)
    28	    set_halo_backend("mpi", topology)
    29	
    30	After that, all operators automatically use MPI halo exchange.
    31	
    32	Optimization
    33	------------
    34	Face-only mode: edges destined for the same neighbor rank are packed
    35	into a single send buffer and exchanged via one sendrecv per neighbor
    36	(at most 4 point-to-point messages per rank).
    37	
    38	Sub-face tiling mode: one sendrecv per edge direction (4 total).
    39	
    40	Limitations and Known Bottlenecks
    41	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    42	1. **Blocking sendrecv**: All halo exchanges use blocking ``sendrecv``
    43	   via ``mpi4jax``.  ``mpi4jax`` does not yet expose ``Isend``/``Irecv``
    44	   non-blocking primitives, so true computation-communication overlap
    45	   is not possible at this level.
    46	
    47	2. **Face-only mode uses neighbor sendrecv**: When rank count ≤ 6,
    48	   edges are grouped by neighbor rank and exchanged via one sendrecv
    49	   per unique neighbor (≤4 messages).  For >6 ranks, the tiled mode
    50	   uses point-to-point sendrecv with only actual neighbors.
    51	
    52	3. **Tiled mode is sequential per edge**: The 4 edge directions are
    53	   exchanged in sequence (not pipelined).  Each sendrecv blocks until
    54	   both send and receive complete.
    55	
    56	Near-term improvement path:
    57	  - Batch edge packing into a single contiguous buffer per neighbor
    58	    rank and do one sendrecv per neighbor (reduces from 4 to ≤4
    59	    messages per rank, with larger messages for better bandwidth).
    60	  - When ``mpi4jax`` gains ``Isend``/``Irecv``, convert to non-blocking
    61	    with ``Waitall`` after all sends/receives are posted.
    62	  - For pure multi-GPU (no MPI), ``jax.lax.ppermute`` is the preferred
    63	    path and integrates with XLA's SPMD partitioner.
    64	"""
    65	
    66	from __future__ import annotations
    67	
    68	import jax
    69	import jax.numpy as jnp
    70	
    71	from legoesm.grids.halo import (
    72	    WEST,
    73	    EAST,
    74	    SOUTH,
    75	    NORTH,
    76	    extract_edge_strip,
    77	    extract_edge_strip_at_depth,
    78	    fill_corners_h1,
    79	    fill_corners_h2,
    80	    fill_corners_h3,  # iter-628: needed for halo=3 MPI port
    81	    interp_strip,     # FV3 3D iter (2026-05-27): duogrid Lagrange remap under MPI
    82	)
    83	from legoesm.parallel.comm import CommTopology
    84	from legoesm.parallel.profiling import mpi_timer
    85	from legoesm.parallel.reductions import (
    86	    check_mpi4jax_transport,
    87	    mpi4jax_array_result,
    88	)
    89	
    90	
    91	_EDGES = (WEST, EAST, SOUTH, NORTH)
    92	_DIR_TO_EDGE = {"west": WEST, "east": EAST, "south": SOUTH, "north": NORTH}
    93	_OPPOSITE_EDGE = {WEST: EAST, EAST: WEST, SOUTH: NORTH, NORTH: SOUTH}
    94	
    95	
    96	def _maybe_interp_strip_h1(
    97	    strip: jax.Array,
    98	    interp_offsets: jax.Array | None,
    99	    face_global: int,
   100	    edge: int,
   101	) -> jax.Array:
   102	    """Apply Lagrange remap to a halo=1 strip if offsets provided.
   103	
   104	    ``interp_offsets`` has shape ``(6, 4, n)`` indexed by
   105	    ``[face_global, edge_idx, j]``.  ``edge`` already equals the
   106	    edge_idx since :data:`WEST`/:data:`EAST`/:data:`SOUTH`/:data:`NORTH`
   107	    are integers 0..3 matching the offsets layout produced by
   108	    :func:`compute_halo_interp_offsets`.
   109	
   110	    Caller must have already applied the ``is_reversed`` flip — offsets
   111	    are pre-baked to expect the strip in the owner's coordinate frame.
   112	    """
   113	    if interp_offsets is None:
   114	        return strip
   115	    return interp_strip(strip, interp_offsets[face_global, edge])
   116	
   117	
   118	def _maybe_interp_strip_hN(
   119	    strip: jax.Array,
   120	    interp_offsets: jax.Array | None,
   121	    face_global: int,
   122	    edge: int,
   123	    depth: int,
   124	) -> jax.Array:
   125	    """Apply Lagrange remap to a halo>=2 strip at a given depth.
   126	
   127	    ``interp_offsets`` has shape ``(6, 4, halo, n)`` indexed by
   128	    ``[face_global, edge_idx, depth, j]``.
   129	    """
   130	    if interp_offsets is None:
   131	        return strip
   132	    return interp_strip(strip, interp_offsets[face_global, edge, depth])
   133	
   134	# MPI tag computation.  Tags must be unique per (edge, rank) pair and
   135	# fit within MPI's tag space (guaranteed at least 2^15-1 = 32767, but
   136	# most implementations support up to 2^31-1).
   137	#
   138	# Old scheme ``edge * 1000 + rank`` collides when rank >= 1000 because
   139	# the edge term (0-3) * 1000 overlaps with the rank offset.
   140	#
   141	# New scheme: ``edge * _MPI_TAG_RANK_STRIDE + rank`` with a stride of
   142	# 100_000, supporting up to 99_999 ranks without collision.  Maximum
   143	# tag value = 3 * 100_000 + 99_999 = 399_999, well within 2^31-1.
   144	_MPI_TAG_RANK_STRIDE = 100_000
   145	
   146	
   147	# ======================================================================
   148	# AD-safe sendrecv wrapper (custom_vjp)
   149	# ======================================================================
   150	
   151	def _make_sendrecv_vjp(mpi4jax_mod):
   152	    """Build an AD-safe sendrecv once mpi4jax is imported.
   153	
   154	    mpi4jax.sendrecv has a transpose rule that swaps source/dest, but
   155	    the XLA lowering raises RuntimeError when ``_must_transpose=True``.
   156	    This wrapper bypasses that by using ``@jax.custom_vjp``: the forward
   157	    calls sendrecv normally, and the backward calls sendrecv with
   158	    swapped endpoints as a fresh forward call (no transpose flag).
   159	
   160	    Non-JAX arguments (source, dest, sendtag, recvtag, comm) are
   161	    declared via ``nondiff_argnums`` so JAX does not attempt to trace
   162	    them.  They are passed through to fwd/bwd as leading static args.
   163	    """
   164	
   165	    def _sendrecv_impl(send_buf, recv_template, source, dest,
   166	                       sendtag, recvtag, comm):
   167	        return mpi4jax_array_result(
   168	            mpi4jax_mod.sendrecv(
   169	                send_buf, recv_template,
   170	                source=source, dest=dest,
   171	                sendtag=sendtag, recvtag=recvtag, comm=comm,
   172	            )
   173	        )
   174	
   175	    _sendrecv = jax.custom_vjp(
   176	        _sendrecv_impl, nondiff_argnums=(2, 3, 4, 5, 6),
   177	    )
   178	
   179	    def _fwd(send_buf, recv_template, source, dest, sendtag, recvtag, comm):
   180	        # fwd has the same signature as the primal function.
   181	        result = mpi4jax_array_result(
   182	            mpi4jax_mod.sendrecv(
   183	                send_buf, recv_template,
   184	                source=source, dest=dest,
   185	                sendtag=sendtag, recvtag=recvtag, comm=comm,
   186	            )
   187	        )
   188	        # Save the send buffer as residual: the BACKWARD recv template must
   189	        # be SEND-shaped (the returned cotangent d_send pairs with send_buf).
   190	        # Cubed-sphere strips have send.shape == recv.shape so zeros_like(g)
   191	        # used to work by accident; Voronoi per-neighbor send/recv counts
   192	        # generally DIFFER, which would mis-shape the backward message and
   193	        # the returned cotangent.  Bit-identical for equal-shape callers.
   194	        #
   195	        # NOTE: a metadata-only residual (shape, dtype) to avoid retaining the
   196	        # send activation (codex integration review 2026-06-10, MINOR) was
   197	        # tried and REVERTED — both numpy-dtype-object and dtype-name-str
   198	        # forms hit "not a valid JAX type" in the backward zeros/sendrecv on
   199	        # this jax/mpi4jax stack.  Correctness of this shared AD primitive
   200	        # outranks the memory micro-opt; revisit with an on-device-verified
   201	        # ShapeDtypeStruct idiom, not a login-node guess.
   202	        return result, (send_buf,)
   203	
   204	    def _bwd(source, dest, sendtag, recvtag, comm, res, g):
   205	        # bwd receives nondiff args first, then residuals, then cotangent.
   206	        # Reverse: swap source<->dest so cotangent flows back to sender.
   207	        # ``g`` is RECV-shaped (cotangent of the output); the message we
   208	        # receive back carries the peer's cotangent for OUR send_buf, so
   209	        # the template must be SEND-shaped.
   210	        (send_buf,) = res
   211	        d_send = mpi4jax_array_result(
   212	            mpi4jax_mod.sendrecv(
   213	                g, jnp.zeros_like(send_buf),
   214	                source=dest, dest=source,
   215	                sendtag=sendtag, recvtag=recvtag, comm=comm,
   216	            )
   217	        )
   218	        return d_send, jnp.zeros_like(g)
   219	
   220	    _sendrecv.defvjp(_fwd, _bwd)
   221	    return _sendrecv
   222	
   223	
   224	# Module-level cache: built lazily on first MPI import.
   225	_sendrecv_vjp_fn = None
   226	
   227	
   228	def get_sendrecv_vjp(mpi4jax_mod):
   229	    """Return the cached AD-safe sendrecv wrapper.
   230	
   231	    This is the single choke point every halo ``sendrecv`` routes through, so
   232	    it is where the mpi4jax GPU-transport preflight runs — on EVERY call, BEFORE
   233	    the (memoised) wrapper is returned. Halo entry points reach mpi4jax by a
   234	    direct ``import mpi4jax`` rather than :func:`require_mpi_stack`, so without
   235	    this call a GPU-direct misconfiguration (``MPI4JAX_USE_CUDA_MPI=1`` against a
   236	    non-CUDA mpi4jax) would slip past the preflight and segfault at the first
   237	    device-buffer exchange.
   238	    """
   239	    check_mpi4jax_transport(mpi4jax_mod)
   240	    global _sendrecv_vjp_fn
   241	    if _sendrecv_vjp_fn is None:
   242	        _sendrecv_vjp_fn = _make_sendrecv_vjp(mpi4jax_mod)
   243	    return _sendrecv_vjp_fn
   244	
   245	
   246	def _place_strip(padded: jax.Array, face: int, edge: int, strip: jax.Array) -> jax.Array:
   247	    """Place a received strip into the correct halo position."""
   248	    if edge == WEST:
   249	        padded = padded.at[face, 0, 1:-1].set(strip)
   250	    elif edge == EAST:
  1200	    halo: int = 1,
  1201	    pole_bc: str = "wall",
  1202	    south_value: float = 0.0,
  1203	    north_value: float = 0.0,
  1204	) -> jax.Array:
  1205	    """Full 2-D halo pad (lat + lon) for ANY pencil row — the increment-3
  1206	    integration of the 2-D building blocks.
  1207	
  1208	    N/S then E/W (corners ride the lat-padded edge columns into the E/W
  1209	    exchange).  N/S: interior cuts MPI-sendrecv with the lat neighbour;
  1210	    pole-touching rows fill the pole side per ``pole_bc``.  Every rank
  1211	    participates in its interior-facing sendrecv (the line terminates at
  1212	    the pole rows' local fill) so there is NO collective-line deadlock —
  1213	    the failure of the earlier "guard-and-skip" interior-only attempt.
  1214	    E/W: the periodic-ring :func:`exchange_halo_lon`.
  1215	
  1216	    ``pole_bc``:
  1217	      * ``"wall"`` (default) — pole ghost rows = constant wall BC
  1218	        (``south_value``/``north_value``).  This is the REGULAR lat-lon
  1219	        case (and ocean, whose poles are closed walls).  Fully LOCAL at
  1220	        the poles ⇒ no longitude transpose, and the whole pad is
  1221	        DEADLOCK-FREE and AD-SAFE (sendrecv-VJP on both axes).
  1222	      * ``"fold"`` / ``"tripole"`` — the atmospheric 180° pole-fold and
  1223	        the ocean tripole north-fold need the FULL longitude axis at the
  1224	        pole row (180° shift / permutation), so they require the
  1225	        lat-pencil transpose (:func:`lon_gather_full`).  NOT YET wired
  1226	        (next increment); raises so a fold deck can't silently get a
  1227	        wall.
  1228	
  1229	    Returns ``(n_lat_local + 2*halo, n_lon_local + 2*halo[, nlev])``.
  1230	    """
  1231	    if pole_bc not in ("wall", "fold", "tripole"):
  1232	        raise ValueError(
  1233	            f"pad_halo_latlon_2d: pole_bc must be 'wall', 'fold', or "
  1234	            f"'tripole', got {pole_bc!r}")
  1235	    if pole_bc in ("fold", "tripole"):
  1236	        raise NotImplementedError(
  1237	            "pad_halo_latlon_2d: pole_bc='fold'/'tripole' needs the "
  1238	            "lat-pencil transpose (lon_gather_full) to do the pole-fold's "
  1239	            "global-longitude shift/permutation on a lon-split row — next "
  1240	            "increment.  Use pole_bc='wall' for regular lat-lon / "
  1241	            "closed-pole ocean."
  1242	        )
  1243	    if halo <= 0:
  1244	        return field
  1245	
  1246	    # halo must fit the SMALLEST local block on BOTH axes — with an
  1247	    # uneven split a neighbour can own fewer than `halo` rows/cols, so
  1248	    # its send/recv would be shorter than this rank expects and the MPI
  1249	    # exchange truncates/aborts/hangs before any reshape (codex MAJOR
  1250	    # 2026-06-13). Smallest block on an even-ish split = floor(n/proc).
  1251	    # _even_split gives the first (n % parts) blocks one extra row/col, so
  1252	    # the SMALLEST block is exactly floor(n_global / parts).
  1253	    min_lat_block = layout.n_lat_global // layout.proc_lat
  1254	    min_lon_block = layout.n_lon_global // layout.proc_lon
  1255	    if halo > min_lat_block or halo > min_lon_block:
  1256	        raise ValueError(
  1257	            f"pad_halo_latlon_2d: halo={halo} exceeds the smallest local "
  1258	            f"block (lat {min_lat_block}=n_lat_global "
  1259	            f"{layout.n_lat_global}//proc_lat {layout.proc_lat}, lon "
  1260	            f"{min_lon_block}=n_lon_global {layout.n_lon_global}//proc_lon "
  1261	            f"{layout.proc_lon}); a neighbour would send/recv a mismatched "
  1262	            f"halo and the MPI exchange would abort/hang.")
  1263	
  1264	    # N/S — lat-axis wall pad (interior cut sendrecv at the AD-safe
  1265	    # rank-as-tag LINE pattern, pole side wall); shared verbatim with
  1266	    # pad_with_pole_bc_lat_2d so the two lat exchanges stay bit-identical.
  1267	    ns = _pad_lat_wall_2d(field, layout, halo, south_value, north_value)
  1268	    # E/W ring on the lat-padded block → fills lon ghosts + corners.
  1269	    return exchange_halo_lon(
  1270	        ns, layout.west_rank, layout.east_rank, layout.rank, halo=halo)
  1271	
  1272	
  1273	# ============================================================================
  1274	# Backend-dispatched pad_halo_latlon implementation
  1275	# ============================================================================
  1276	#
  1277	# Used by :mod:`legoesm.grids.halo_latlon` when the global halo
  1278	# backend is ``"mpi"`` and the active topology is a
  1279	# :class:`LatLonBandLayout`.  Composes the existing
  1280	# :func:`exchange_halo_latlon` (lat MPI sendrecv + boundary pole-fold)
  1281	# with a periodic-lon wrap that every rank performs locally.  Output
  1282	# shape matches the serial :func:`legoesm.grids.halo_latlon.pad_halo_latlon`
  1283	# family — operators stay backend-oblivious.
  1284	#
  1285	# This is the architectural reuse point the user asked for: no
   525	                "initialize_distributed_latlon() called after a "
   526	                "non-lat-lon distributed init; replacing the active "
   527	                "topology with a LatLonBandLayout and re-arming the "
   528	                "MPI halo backend.",
   529	                RuntimeWarning, stacklevel=2,
   530	            )
   531	            _active_topology = None
   532	    if _active_topology is not None:
   533	        _want_n_lon = (2 * global_n_lat if global_n_lon is None
   534	                       else global_n_lon)
   535	        if (_active_topology.n_lat_global != global_n_lat
   536	                or _active_topology.n_lon_global != _want_n_lon):
   537	            # Same hydra, third head (after the cross-grid TYPE reuse
   538	            # above and the test backend leaks): reusing a layout for
   539	            # a DIFFERENT global grid silently mis-slices every band
   540	            # (and may carry a foreign tripolar fold) — the np=2 PCG
   541	            # step-parity 1e-5 drift in merge gate 8460563 was exactly
   542	            # a leaked smaller fold-active layout.  Re-arm fresh.
   543	            warnings.warn(
   544	                "initialize_distributed_latlon() re-called with a "
   545	                f"different global grid ({global_n_lat}x{_want_n_lon} "
   546	                f"vs active {_active_topology.n_lat_global}x"
   547	                f"{_active_topology.n_lon_global}); replacing the "
   548	                "active layout and re-arming the MPI halo backend.",
   549	                RuntimeWarning, stacklevel=2,
   550	            )
   551	            _active_topology = None
   552	    if _active_topology is not None:
   553	        # A changed band-boundary request must not be served by a stale
   554	        # layout (codex): a long-lived process that armed even bands and
   555	        # later opts into wet-cell-balanced boundaries (or vice versa) would
   556	        # silently keep the OLD decomposition — every slicer/scatter reads
   557	        # lat_start/lat_end from the layout. Compare this rank's requested
   558	        # span against the active one; re-arm fresh on mismatch.
   559	        _r = _active_topology.rank
   560	        _n = _active_topology.n_ranks
   561	        if band_boundaries is not None:
   562	            # Validate BEFORE the span comparison: an invalid request
   563	            # (non-integral / overlong / bad span) must raise, never be
   564	            # silently served by a coincidentally-matching stale layout
   565	            # (codex round 2).
   566	            from legoesm.parallel.latlon_mpi import validate_band_boundaries
   567	            _b = validate_band_boundaries(band_boundaries, _n, global_n_lat)
   568	            _want_span = (_b[_r], _b[_r + 1])
   569	        else:
   570	            _base, _rem = divmod(global_n_lat, _n)
   571	            _s = (_r * (_base + 1) if _r < _rem
   572	                  else _rem * (_base + 1) + (_r - _rem) * _base)
   573	            _want_span = (_s, _s + _base + (1 if _r < _rem else 0))
   574	        if (_active_topology.lat_start,
   575	                _active_topology.lat_end) != _want_span:
   576	            warnings.warn(
   577	                "initialize_distributed_latlon() re-called with different "
   578	                f"band boundaries (rank {_r}: requested rows "
   579	                f"[{_want_span[0]}, {_want_span[1]}) vs active "
   580	                f"[{_active_topology.lat_start}, "
   581	                f"{_active_topology.lat_end})); replacing the active layout "
   582	                "and re-arming the MPI halo backend.",
   583	                RuntimeWarning, stacklevel=2,
   584	            )
   585	            _active_topology = None
   586	    if _active_topology is not None:
   587	        active_fold = getattr(_active_topology, "fold", None)
   588	        active_on = (active_fold is not None
   589	                     and getattr(active_fold, "is_active", False))
   590	        requested_on = fold is not None and getattr(fold, "is_active", False)
   591	        if active_on and not requested_on:
   592	            # Mirror of the upgrade case below: a leaked TRIPOLAR layout
   593	            # must not serve a REGULAR-grid request — the foreign north
   594	            # fold permutes/sign-flips the northern band rows of every
   595	            # subsequent pad (merge gate 8460563: 1e-5 step-parity drift
   596	            # from exactly this same-dims stale-fold reuse).  Re-arm
   597	            # without the fold.
   598	            warnings.warn(
   599	                "initialize_distributed_latlon() re-called WITHOUT a "
   600	                "tripolar fold after a fold-active init; replacing the "
   601	                "active layout with a fold-less one and re-arming the "
   602	                "MPI halo backend.",
   603	                RuntimeWarning, stacklevel=2,
   604	            )
   605	            updated = _active_topology._replace(fold=None)
   606	            _active_topology = updated
   607	            from legoesm.grids.halo import set_halo_backend
   608	            set_halo_backend("mpi", updated)
   609	            return updated
   610	        if requested_on and not active_on:
   611	            # A prior fold-less init must NOT mask a later tripolar (ORCA)
   612	            # init — otherwise the ocean run would silently use the
   613	            # geographic pole-fold.  Update the active layout to carry the
   614	            # fold and re-arm the MPI halo backend.
   615	            warnings.warn(
   616	                "initialize_distributed_latlon() re-called with a tripolar "
   617	                "fold after a fold-less init; updating the active layout to "
   618	                "carry the fold.",
   619	                RuntimeWarning, stacklevel=2,
   620	            )
   621	            updated = _active_topology._replace(fold=fold)
   622	            _active_topology = updated
   623	            from legoesm.grids.halo import set_halo_backend
   624	            set_halo_backend("mpi", updated)
   625	            return updated
   626	        warnings.warn(
   627	            "initialize_distributed_latlon() called more than once. "
   628	            "Returning the existing topology.",
   629	            RuntimeWarning,
   630	            stacklevel=2,
   631	        )
   632	        return _active_topology
   633	
   634	    if global_n_lon is None:
   635	        global_n_lon = 2 * global_n_lat
   636	
   637	    # Validate MPI dependencies before touching JAX.
   638	    _mpi4jax, MPI = require_mpi_stack()
   639	    comm = MPI.COMM_WORLD
   640	    rank = comm.Get_rank()
   641	    n_processes = comm.Get_size()
   642	
   643	    from legoesm.parallel.latlon_mpi import make_latlon_band_layout
   644	    layout = make_latlon_band_layout(
   645	        rank=rank, n_ranks=n_processes,
   646	        n_lat=global_n_lat, n_lon=global_n_lon,
   647	        fold=fold,
   648	        boundaries=band_boundaries,
   649	    )
   650	    _active_topology = layout
   651	
   652	    from legoesm.grids.halo import set_halo_backend
   653	    set_halo_backend("mpi", layout)
   654	    return layout
   655	
   656	
   657	def get_active_topology() -> CommTopology | None:
   658	    """Return the active MPI communication topology, if initialized."""
   659	    return _active_topology
   660	
   661	
   662	def get_active_layout():
   663	    """Return the active distributed layout, if initialized."""
   664	    return _active_layout
   665	
   666	
   667	def set_active_layout(layout) -> None:
   668	    """Set the active distributed layout (for deferred construction)."""
   669	    global _active_layout
   670	    _active_layout = layout
   671	
   672	
   673	def reset_distributed_topology() -> None:
   674	    """Disarm the MPI halo backend and forget the active topology/layout.
   675	
2790-    boundary ranks, with no pre-padding of state on the caller side.
2791-
2792-    Architecture
2793-    ------------
2794:    1. ``set_halo_backend("mpi", layout)`` is activated once at
2795-       factory time.  Every subsequent ``pad_halo_latlon*``,
2796-       ``pad_ns_zero``, and ``pad_with_pole_bc_lat`` call inside the
2797-       dycore operators dispatches through the MPI sendrecv path.
2798-       ``is_distributed()`` returns True, which gates the
--
2865-        ``step_fn(local_state, dt, *, target_mass=None) -> local_state``.
2866-
2867-    Side effect
2868-    -----------
2869:    Activates ``set_halo_backend("mpi", layout)`` at the global
2870-    module level.  Caller is responsible for resetting via
2871:    ``set_halo_backend("local")`` once the run is finished — see
2872-    ``ModelDriver._finalize_run`` for the cubed-sphere precedent.
2873-    Tests that share a Python process should reset between cases.
2874-
2875-    Note on ``halo`` kwarg
--
2900-    # pad_with_pole_bc_lat call inside the dycore now dispatches
2901-    # through inter-rank sendrecv at partition cuts and constant /
2902-    # pole-fold pads at boundary ranks.  This also flips
2903-    # ``is_distributed()`` so the mass-fixer reductions allreduce.
2904:    set_halo_backend("mpi", layout)
2905-    from legoesm.grids.polar_filter import (
2906-        get_polar_filter_lon_gather,
2907-        set_polar_filter_lon_gather,
2908-    )
--
3007-
3008-    return step_fn
3009-
3010-
3011:def make_latlon_2d_mpi_step(
3012-    model,
3013-    layout: LatLon2DLayout,
3014-    *,
3015-    physics_fn: Callable | None = None,
--
3104-    from legoesm.timestepping.integration import (
3105-        physics_requires_phys_state,
3106-    )
3107-
3108:    set_halo_backend("mpi", layout)
3109-
3110-    global_total_area = global_sum_mpi(jnp.sum(model.grid.area))
3111-    mpi_grid = model.grid._replace(total_area=global_total_area)
3112-    mpi_config = model.config._replace(
packages/core/legoesm/parallel/__init__.py:206:__all__ = [

exec
/usr/bin/bash -lc "nl -ba tests/distributed/test_latlon_2d_pad_wall_mpi.py | sed -n '1,245p'; nl -ba tests/distributed/conftest.py | sed -n '1,150p'; nl -ba packages/core/legoesm/grids/halo.py | sed -n '1,240p'; rg -n -C 3 \"widen_cgrid_geometry_band\\(|widen_band_cell_fields\\(|widen_band_vface_fields\\(\" packages/core/legoesm | head -600; git diff --check HEAD -- packages/core/legoesm/parallel/latlon_mpi.py packages/core/legoesm/grids/halo_latlon.py tests/distributed/test_latlon_2d_fused_pad.py; git ls-files --error-unmatch tests/distributed/test_latlon_2d_fused_pad.py 2>&1 || true" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 368ms:
     1	"""MPI-collected parity + AD tests for the full 2-D halo pad
     2	(``pad_halo_latlon_2d``, ``pole_bc="wall"`` — regular lat-lon /
     3	closed-pole ocean).  This is the CI home (codex re-review 2026-06-13:
     4	the parallel-lane file ran its MPI checks only under ``__main__`` so
     5	they were never pytest-collected → an AD regression could slip through).
     6	
     7	Run under MPI:  ``mpirun -np {2,3,6} python -m pytest <thisfile>``
     8	(wired into ``.github/workflows/mpi-distributed.yml`` alongside the
     9	1-D halo tests).  Under a single rank (plain pytest) every test skips —
    10	the single-proc forward/grad/raise paths live in the pytest lane
    11	``tests/parallel/test_latlon_2d_pad_wall.py``.
    12	
    13	Size-adaptive: the process grid is factored from the world size so the
    14	SAME file covers np=2 (1×2 ring), np=3 (1×3 ring — the phase-constant
    15	tag reverse-ring case, proc_lon≥3), and np=6 (2×3 — N/S reverse sendrecv
    16	AND the proc_lon=3 reverse ring together).  ``n_lat=2·pr+1`` /
    17	``n_lon=2·pc+1`` give UNEVEN splits, so the remainder path is exercised
    18	too (smallest block = 2 ≥ halo).
    19	"""
    20	from __future__ import annotations
    21	
    22	import jax
    23	
    24	jax.config.update("jax_enable_x64", True)
    25	
    26	import jax.numpy as jnp
    27	import numpy as np
    28	import pytest
    29	
    30	mpi4jax = pytest.importorskip("mpi4jax")
    31	MPI = pytest.importorskip("mpi4py.MPI")
    32	
    33	from legoesm.parallel.latlon_mpi import (
    34	    make_latlon_2d_layout,
    35	    scatter_field_latlon_2d,
    36	    pad_halo_latlon_2d,
    37	)
    38	from legoesm.parallel.reductions import global_sum_mpi
    39	
    40	SV, NV = -1.0, -2.0   # distinct non-zero walls to catch S/N confusion
    41	
    42	
    43	def _pick_grid(n_ranks: int) -> tuple[int, int]:
    44	    """Factor the world size into ``(proc_lat, proc_lon)`` maximising
    45	    halo+VJP coverage: a balanced 2-D split (proc_lat≥2 AND proc_lon≥2)
    46	    when ``n_ranks`` is composite, else a 1×n ring for primes.  Always
    47	    prefers the LARGER factor on proc_lon so the periodic-ring reverse
    48	    VJP (phase-constant tags at proc_lon≥3) gets exercised whenever a
    49	    factor allows it."""
    50	    best = (1, n_ranks)
    51	    for pr in range(2, int(n_ranks ** 0.5) + 1):
    52	        if n_ranks % pr == 0:
    53	            best = (pr, n_ranks // pr)   # pr ≤ pc, so pc holds the larger factor
    54	    return best
    55	
    56	
    57	def _serial_ref(g, halo):
    58	    """lat-wall (axis0) THEN lon-periodic (axis1) — the serial twin of
    59	    the 2-D pad's N/S-then-E/W order (wall rows are constant so their
    60	    lon-wrap is the same constant ⇒ corners = wall)."""
    61	    n_lon = g.shape[1]
    62	    sw = np.full((halo, n_lon) + g.shape[2:], SV, g.dtype)
    63	    nw = np.full((halo, n_lon) + g.shape[2:], NV, g.dtype)
    64	    latw = np.concatenate([sw, g, nw], axis=0)
    65	    pads = [(0, 0)] * latw.ndim
    66	    pads[1] = (halo, halo)
    67	    return np.pad(latw, pads, mode="wrap")
    68	
    69	
    70	def _serial_ref_jax(g, halo):
    71	    """Differentiable twin of :func:`_serial_ref` — pad is a LINEAR
    72	    gather (const wall rows + lon wrap), so jnp.pad(mode='wrap') gives
    73	    the exact serial forward AND its adjoint for the grad reference."""
    74	    n_lon = g.shape[1]
    75	    sw = jnp.full((halo, n_lon) + g.shape[2:], SV, g.dtype)
    76	    nw = jnp.full((halo, n_lon) + g.shape[2:], NV, g.dtype)
    77	    latw = jnp.concatenate([sw, g, nw], axis=0)
    78	    pads = [(0, 0)] * latw.ndim
    79	    pads[1] = (halo, halo)
    80	    return jnp.pad(latw, pads, mode="wrap")
    81	
    82	
    83	def test_mpi_wall_pad_parity():
    84	    """Each rank's wall 2-D pad == the corresponding window of the
    85	    serial lat-wall+lon-periodic global pad."""
    86	    comm = MPI.COMM_WORLD
    87	    rank, n = comm.Get_rank(), comm.Get_size()
    88	    if n < 2:
    89	        pytest.skip("needs mpirun with >=2 ranks (single-proc path is in "
    90	                    "tests/parallel/test_latlon_2d_pad_wall.py)")
    91	    pr, pc = _pick_grid(n)
    92	    h = 1
    93	    n_lat, n_lon = 2 * pr + 1, 2 * pc + 1     # uneven splits, min block 2
    94	    g = np.random.default_rng(5).standard_normal((n_lat, n_lon))
    95	    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    96	    local = scatter_field_latlon_2d(jnp.asarray(g), L)
    97	    out = np.asarray(pad_halo_latlon_2d(
    98	        local, L, halo=h, south_value=SV, north_value=NV))
    99	    ref = _serial_ref(g, h)[
   100	        L.lat_start:L.lat_end + 2 * h, L.lon_start:L.lon_end + 2 * h]
   101	    np.testing.assert_allclose(
   102	        out, ref, atol=1e-12,
   103	        err_msg=f"rank {rank} wall 2-D pad != serial (pr={pr} pc={pc})")
   104	    if rank == 0:
   105	        print(f"WALL_2D_PAD_OK pr={pr} pc={pc} n_lat={n_lat} n_lon={n_lon}",
   106	              flush=True)
   107	
   108	
   109	def test_mpi_wall_grad_parity():
   110	    """Reverse-mode AD through BOTH sendrecv VJPs.  The loss is
   111	    ``global_sum_mpi(sum(pad_local**2))`` (AD-safe allreduce SUM) —
   112	    WITHOUT the global sum, jax.grad of a rank's OWN loss collapses to
   113	    ``2*local`` (ghost cotangents flow AWAY to neighbours and never
   114	    re-enter ``d(own_loss)/d(own_block)``, leaving the VJP untested).
   115	    Summing every rank's loss makes each neighbour's ghost-loss cotangent
   116	    flow BACK through the halo VJP into this rank's owned cells, so the
   117	    distributed grad on a rank's OWNED block == the GLOBAL grad of
   118	    ``sum_r sum(P[window_r]**2)`` (P=_serial_ref_jax) restricted to that
   119	    block.  ``proc_lon≥3`` (np=3, np=6) drives the periodic ring through
   120	    REVERSE with the phase-constant tags it requires; ``proc_lat≥2``
   121	    (np=6) drives the N/S reverse sendrecv."""
   122	    comm = MPI.COMM_WORLD
   123	    rank, n = comm.Get_rank(), comm.Get_size()
   124	    if n < 2:
   125	        pytest.skip("needs mpirun with >=2 ranks")
   126	    pr, pc = _pick_grid(n)
   127	    h = 1
   128	    n_lat, n_lon = 2 * pr + 1, 2 * pc + 1
   129	    g = np.random.default_rng(7).standard_normal((n_lat, n_lon))
   130	    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
   131	    local = scatter_field_latlon_2d(jnp.asarray(g), L)
   132	
   133	    def loss(x):
   134	        local_loss = jnp.sum(pad_halo_latlon_2d(
   135	            x, L, halo=h, south_value=SV, north_value=NV) ** 2)
   136	        return global_sum_mpi(local_loss, comm)
   137	
   138	    grad_local = np.asarray(jax.grad(loss)(local))
   139	    assert np.all(np.isfinite(grad_local)), \
   140	        f"rank {rank} non-finite distributed grad"
   141	
   142	    def total_serial(gg):
   143	        P = _serial_ref_jax(gg, h)
   144	        tot = 0.0
   145	        for r in range(pr * pc):
   146	            Lr = make_latlon_2d_layout(r, pr, pc, n_lat, n_lon)
   147	            tot = tot + jnp.sum(
   148	                P[Lr.lat_start:Lr.lat_end + 2 * h,
   149	                  Lr.lon_start:Lr.lon_end + 2 * h] ** 2)
   150	        return tot
   151	
   152	    ref_grad = np.asarray(jax.grad(total_serial)(jnp.asarray(g)))
   153	    want = ref_grad[L.lat_start:L.lat_end, L.lon_start:L.lon_end]
   154	    np.testing.assert_allclose(
   155	        grad_local, want, atol=1e-9,
   156	        err_msg=f"rank {rank} distributed grad != serial ref (pr={pr} pc={pc})")
   157	    if rank == 0:
   158	        print(f"WALL_2D_GRAD_OK pr={pr} pc={pc} n_lat={n_lat} n_lon={n_lon}",
   159	              flush=True)
     1	"""Distributed test fixtures.
     2	
     3	The MPI face layout is (re-)initialized PER TEST via ``cube_face_layout``
     4	— a once-per-session init cannot work here because the autouse
     5	``_isolate_distributed_state`` teardown resets the process-global
     6	topology/layout/halo-backend after every test.
     7	"""
     8	
     9	import pytest
    10	
    11	mpi4jax = pytest.importorskip("mpi4jax")
    12	MPI = pytest.importorskip("mpi4py.MPI")
    13	
    14	
    15	@pytest.fixture()
    16	def cube_face_layout():
    17	    """(Re-)initialize the active MPI face layout + arm the MPI halo backend.
    18	
    19	    ``_isolate_distributed_state`` (autouse below) resets the
    20	    process-global topology, layout and halo backend AFTER EVERY test —
    21	    correct leak protection, but it also tears down any module- or
    22	    session-scoped initialization after the first test.  Any test that
    23	    relies on the ACTIVE layout (bare ``scatter_to_local`` /
    24	    ``gather_to_global``) or on an armed ``'mpi'`` halo backend must
    25	    therefore re-establish them at test SETUP through this fixture.
    26	    (This mismatch was the long-standing 'broken local MPI stack':
    27	    deterministic ``No layout provided and no active layout is set``
    28	    failures that looked like an mpi4jax/jax version problem — the
    29	    mpi4jax primitives were fine all along.)
    30	
    31	    Safe to call repeatedly: after the reset the topology is None so
    32	    ``initialize_distributed`` runs the clean first-init path;
    33	    ``jax.distributed`` bootstrap is idempotent (single-node skips it).
    34	    """
    35	    from legoesm.parallel.distributed import (
    36	        initialize_distributed,
    37	        get_active_layout,
    38	    )
    39	
    40	    if get_active_layout() is None:
    41	        global_n = max(MPI.COMM_WORLD.Get_size(), 2)
    42	        initialize_distributed(global_n=global_n)
    43	    layout = get_active_layout()
    44	    if layout is None:
    45	        pytest.skip("MPI layout unavailable on this rank configuration")
    46	    return layout
    47	
    48	
    49	@pytest.fixture(autouse=True)
    50	def _mpi_test_isolation():
    51	    """Resynchronize ranks and drain stray messages around every test.
    52	
    53	    Each distributed test builds its own halo-exchange scenario (different
    54	    grids, layouts, and ``sendrecv`` buffer sizes).  Without an explicit
    55	    barrier between tests the ranks can drift out of lock-step, and a
    56	    message posted by one test that is not consumed before the next test
    57	    starts is later matched — by source/tag — against a *different-sized*
    58	    receive buffer, raising ``MPI_ERR_TRUNCATE`` and aborting the whole
    59	    job.  This was reproducible as ``test_latlon_mpi_step`` poisoning
    60	    ``test_latlon_mpi_tripole`` (order-dependent, version-independent).
    61	
    62	    The fix is standard MPI test hygiene: barrier-sync before and after
    63	    each test, and drain any straggler messages (as raw bytes, since
    64	    mpi4jax uses the buffer protocol) so they cannot truncate the next
    65	    test's receive.
    66	    """
    67	    comm = MPI.COMM_WORLD
    68	    if comm.Get_size() < 2:
    69	        # Single-rank runs have no peer messages to drain or sync.
    70	        yield
    71	        return
    72	
    73	    def _drain() -> None:
    74	        status = MPI.Status()
    75	        while comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
    76	            count = status.Get_count(MPI.BYTE)
    77	            buf = bytearray(count)
    78	            comm.Recv([buf, MPI.BYTE],
    79	                      source=status.Get_source(), tag=status.Get_tag())
    80	
    81	    comm.Barrier()
    82	    yield
    83	    comm.Barrier()
    84	    _drain()
    85	    comm.Barrier()
    86	
    87	
    88	@pytest.fixture(autouse=True)
    89	def _isolate_distributed_state():
    90	    """Reset the armed halo backend + active topology AFTER every test.
    91	
    92	    A leftover ``set_halo_backend('mpi', layout)`` is a process global:
    93	    it leaks into every pad of every later test in the same pytest
    94	    process — a later test's 'serial reference' then silently
    95	    dispatches through the stale band layout (merge gate 8460566: a
    96	    deterministic 1e-5 step-parity failure that vanished when the test
    97	    ran alone).  Mirrors tests/ocean/distributed/conftest.py.
    98	    """
    99	    yield
   100	    from legoesm.parallel.distributed import reset_distributed_topology
   101	    reset_distributed_topology()
     1	"""Inter-face halo exchange for the cubed-sphere grid.
     2	
     3	Implements proper communication between the 6 faces of the cubed-sphere,
     4	replacing the incorrect `jnp.roll` (within-face wrapping) with correct
     5	neighbor-face data exchange.
     6	
     7	The connectivity table is derived from the gnomonic projection geometry
     8	in `cubed_sphere.py`. Each face edge maps to a specific edge on a
     9	neighbor face, with possible index reversal and axis swaps.
    10	
    11	All functions are pure and compatible with jax.jit, jax.grad, jax.vmap.
    12	
    13	References
    14	----------
    15	- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
    16	- Ronchi, Iacono, Paolucci (1996): The "Cubed Sphere"
    17	"""
    18	
    19	from __future__ import annotations
    20	
    21	import contextlib
    22	import importlib
    23	import importlib.util
    24	
    25	import jax
    26	import jax.numpy as jnp
    27	import numpy as np
    28	
    29	
    30	# ==============================================================================
    31	# Edge constants
    32	# ==============================================================================
    33	
    34	WEST = 0   # i = 0 boundary
    35	EAST = 1   # i = n-1 boundary
    36	SOUTH = 2  # j = 0 boundary
    37	NORTH = 3  # j = n-1 boundary
    38	
    39	
    40	# ==============================================================================
    41	# Face connectivity table
    42	# ==============================================================================
    43	#
    44	# CONNECTIVITY[face][edge] = (neighbor_face, neighbor_edge, reversed)
    45	#
    46	# Face numbering (from cubed_sphere.py _face_to_cartesian):
    47	#   0: +x (front)    1: +y (right)   2: -x (back)
    48	#   3: -y (left)     4: +z (north)   5: -z (south)
    49	#
    50	# Derived by tracing the gnomonic projection at each face boundary.
    51	# All connections are verified to be symmetric:
    52	#   if A_edge -> (B, B_edge, rev), then B_edge -> (A, A_edge, rev)
    53	
    54	CONNECTIVITY = {
    55	    0: {
    56	        WEST:  (3, EAST, False),
    57	        EAST:  (1, WEST, False),
    58	        SOUTH: (5, NORTH, False),
    59	        NORTH: (4, SOUTH, False),
    60	    },
    61	    1: {
    62	        WEST:  (0, EAST, False),
    63	        EAST:  (2, WEST, False),
    64	        SOUTH: (5, EAST, True),     # reversed
    65	        NORTH: (4, EAST, False),
    66	    },
    67	    2: {
    68	        WEST:  (1, EAST, False),
    69	        EAST:  (3, WEST, False),
    70	        SOUTH: (5, SOUTH, True),    # reversed
    71	        NORTH: (4, NORTH, True),    # reversed
    72	    },
    73	    3: {
    74	        WEST:  (2, EAST, False),
    75	        EAST:  (0, WEST, False),
    76	        SOUTH: (5, WEST, False),
    77	        NORTH: (4, WEST, True),     # reversed
    78	    },
    79	    4: {
    80	        WEST:  (3, NORTH, True),    # reversed
    81	        EAST:  (1, NORTH, False),
    82	        SOUTH: (0, NORTH, False),
    83	        NORTH: (2, NORTH, True),    # reversed
    84	    },
    85	    5: {
    86	        WEST:  (3, SOUTH, False),
    87	        EAST:  (1, SOUTH, True),    # reversed
    88	        SOUTH: (2, SOUTH, True),    # reversed
    89	        NORTH: (0, SOUTH, False),
    90	    },
    91	}
    92	
    93	
    94	# ==============================================================================
    95	# Halo interpolation offset precomputation
    96	# ==============================================================================
    97	
    98	def _face_to_xyz_np(face: int, ax: float, ay: float) -> np.ndarray:
    99	    """Gnomonic coords → unit-sphere Cartesian (numpy, scalar)."""
   100	    tx, ty = np.tan(ax), np.tan(ay)
   101	    if face == 0:   x, y, z = 1.0, tx, ty
   102	    elif face == 1: x, y, z = -tx, 1.0, ty
   103	    elif face == 2: x, y, z = -1.0, -tx, ty
   104	    elif face == 3: x, y, z = tx, -1.0, ty
   105	    elif face == 4: x, y, z = -ty, tx, 1.0
   106	    elif face == 5: x, y, z = ty, tx, -1.0
   107	    else: raise ValueError(face)
   108	    r = np.sqrt(x**2 + y**2 + z**2)
   109	    return np.array([x / r, y / r, z / r])
   110	
   111	
   112	def _xyz_to_gnomonic_np(
   113	    face: int, x: float, y: float, z: float,
   114	) -> tuple[float, float]:
   115	    """Unit-sphere Cartesian → gnomonic coords on *face* (numpy, scalar).
   116	
   117	    Inverts the projection: project from the sphere centre through
   118	    (x, y, z) onto the face plane and recover (alpha_x, alpha_y).
   119	    """
   120	    if face == 0:   return float(np.arctan(y / x)), float(np.arctan(z / x))
   121	    elif face == 1: return float(np.arctan(-x / y)), float(np.arctan(z / y))
   122	    elif face == 2: return float(np.arctan(y / x)), float(np.arctan(-z / x))
   123	    elif face == 3: return float(np.arctan(-x / y)), float(np.arctan(-z / y))
   124	    elif face == 4: return float(np.arctan(y / z)), float(np.arctan(-x / z))
   125	    elif face == 5: return float(np.arctan(-y / z)), float(np.arctan(-x / z))
   126	    else: raise ValueError(face)
   127	
   128	
   129	def extract_edge_strip_at_depth(
   130	    data: jax.Array, face: int, edge: int, depth: int,
   131	) -> jax.Array:
   132	    """Extract strip at given depth from edge (depth=0 is boundary row).
   133	
   134	    Parameters
   135	    ----------
   136	    data : jax.Array, shape (6, n, n)
   137	    face : int
   138	    edge : int
   139	        WEST, EAST, SOUTH, NORTH.
   140	    depth : int
   141	        0 = boundary row, 1 = one row inward, etc.
   142	
   143	    Returns
   144	    -------
   145	    strip : jax.Array, shape (n,)
   146	    """
   147	    if edge == WEST:
   148	        return data[face, depth, :]
   149	    elif edge == EAST:
   150	        return data[face, -(depth + 1), :]
   151	    elif edge == SOUTH:
   152	        return data[face, :, depth]
   153	    elif edge == NORTH:
   154	        return data[face, :, -(depth + 1)]
   155	    else:
   156	        raise ValueError(f"Invalid edge: {edge}")
   157	
   158	
   159	def compute_halo_interp_offsets(n: int) -> jnp.ndarray:
   160	    """Precompute fractional-index offsets for the interpolated halo exchange.
   161	
   162	    The standard halo exchange copies the j-th cell of the neighbour's
   163	    edge strip into the j-th halo slot.  On a gnomonic cubed sphere the
   164	    physical position of the halo slot (from the face's own extended
   165	    gnomonic grid) does NOT coincide with the neighbour's j-th cell —
   166	    the mismatch grows toward cube corners (up to ~0.5 grid cells at C48).
   167	
   168	    This function computes, for every halo cell, the true fractional index
   169	    on the neighbour strip so that ``pad_halo`` can linearly interpolate
   170	    instead of doing a nearest-index copy.
   171	
   172	    Parameters
   173	    ----------
   174	    n : int
   175	        Number of cells per face edge.
   176	
   177	    Returns
   178	    -------
   179	    offsets : jax.Array, shape (6, 4, n)
   180	        ``offsets[face, edge_idx, j]`` is the correction δ such that
   181	        the true fractional index on the (possibly reversed) neighbour
   182	        strip is ``j + δ``.  Edge indices: 0=WEST, 1=EAST, 2=SOUTH, 3=NORTH.
   183	    """
   184	    # halo=1 is the depth-0 slice of the general N-depth precomputation
   185	    # (mirrors compute_halo_interp_offsets_ed → _compute_halo_interp_offsets_ed_hN).
   186	    return _compute_halo_interp_offsets_hN(n, 1)[:, :, 0, :]
   187	
   188	
   189	def _ed_indomain_boundary_strip(arr, edge, n, ext):
   190	    """In-domain boundary strip (length n, ascending array order) of an
   191	    ``(M, M)`` padded gnomonic_ed face, in-domain block ``[ext:ext+n]``."""
   192	    if edge == WEST:
   193	        return arr[ext, ext:ext + n]
   194	    elif edge == EAST:
   195	        return arr[ext + n - 1, ext:ext + n]
   196	    elif edge == SOUTH:
   197	        return arr[ext:ext + n, ext]
   198	    else:  # NORTH
   199	        return arr[ext:ext + n, ext + n - 1]
   200	
   201	
   202	def _ed_halo_strip(arr, edge, n, ext, depth):
   203	    """Halo strip at ``depth`` cells beyond ``edge`` (depth 0 = adjacent to the
   204	    in-domain boundary), length n, ascending array order."""
   205	    if edge == WEST:
   206	        return arr[ext - 1 - depth, ext:ext + n]
   207	    elif edge == EAST:
   208	        return arr[ext + n + depth, ext:ext + n]
   209	    elif edge == SOUTH:
   210	        return arr[ext:ext + n, ext - 1 - depth]
   211	    else:  # NORTH
   212	        return arr[ext:ext + n, ext + n + depth]
   213	
   214	
   215	def _compute_halo_interp_offsets_ed_hN(n: int, halo: int) -> jnp.ndarray:
   216	    """gnomonic_ed cross-face halo interp offsets, ``(6, 4, halo, n)``.
   217	
   218	    Position-matching on the actual extended gnomonic_ed cell centres
   219	    (`gnomonic_ed_padded_centers`, which extend cleanly into the halo): each
   220	    halo cell (depth 0..halo-1 beyond an edge) is matched to its neighbour's
   221	    in-domain edge strip via a parabola-vertex fit on great-circle distance,
   222	    giving the true fractional index; ``δ = frac − j``.  gnomonic_ed-specific
   223	    (the equiangular analytic offsets are the wrong geometry — codex gating
   224	    blocker).
   225	    """
   226	    from legoesm.grids.cubed_sphere import gnomonic_ed_padded_centers
   227	
   228	    ext = halo + 1
   229	    lon, lat = gnomonic_ed_padded_centers(n, halo)  # (6, M, M), M=n+2*halo+2
   230	    lon = np.asarray(lon)
   231	    lat = np.asarray(lat)
   232	    xyz = np.stack([
   233	        np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
   234	    ], axis=-1)
   235	
   236	    edges = [WEST, EAST, SOUTH, NORTH]
   237	    offsets = np.zeros((6, 4, halo, n), dtype=np.float64)
   238	
   239	    for face in range(6):
   240	        for edge_idx, edge in enumerate(edges):
packages/core/legoesm/grids/halo_latlon.py-884-    return jnp.where(north_sel, north_row, out)
packages/core/legoesm/grids/halo_latlon.py-885-
packages/core/legoesm/grids/halo_latlon.py-886-
packages/core/legoesm/grids/halo_latlon.py:887:def widen_band_cell_fields(fields, halo: int, *, clamp_poles: bool = False):
packages/core/legoesm/grids/halo_latlon.py-888-    """Widen cell-row (leading dim ``n_lat``) fields by ``halo`` rows/side.
packages/core/legoesm/grids/halo_latlon.py-889-
packages/core/legoesm/grids/halo_latlon.py-890-    ONE fused exchange under the MPI band backend
--
packages/core/legoesm/grids/halo_latlon.py-903-    )
packages/core/legoesm/grids/halo_latlon.py-904-
packages/core/legoesm/grids/halo_latlon.py-905-
packages/core/legoesm/grids/halo_latlon.py:906:def widen_band_vface_fields(fields, halo: int, *, clamp_poles: bool = False):
packages/core/legoesm/grids/halo_latlon.py-907-    """Widen v-face (leading dim ``n_lat+1``) fields by ``halo`` rows/side.
packages/core/legoesm/grids/halo_latlon.py-908-
packages/core/legoesm/grids/halo_latlon.py-909-    A band's v array carries one duplicated boundary face, so faces cannot
--
packages/core/legoesm/grids/halo_latlon.py-940-    )
packages/core/legoesm/grids/halo_latlon.py-941-
packages/core/legoesm/grids/halo_latlon.py-942-
packages/core/legoesm/grids/halo_latlon.py:943:def widen_cgrid_geometry_band(geom, halo: int):
packages/core/legoesm/grids/halo_latlon.py-944-    """Extended-band twin of ``slice_cgrid_geometry_to_band``: widen a
packages/core/legoesm/grids/halo_latlon.py-945-    band-local :class:`~legoesm.grids.latlon.LatLonCGridGeometry` by
packages/core/legoesm/grids/halo_latlon.py-946-    ``halo`` ghost rows per side via the ACTIVE halo backend.
--
packages/core/legoesm/grids/halo_latlon.py-985-        cell_names = cell_names + ("seam_wall_rows",)
packages/core/legoesm/grids/halo_latlon.py-986-    vface_names = ("dx_v", "dy_v", "area_q", "f_v", "cos_alpha_v",
packages/core/legoesm/grids/halo_latlon.py-987-                   "sin_alpha_v", "cos_lat_v")
packages/core/legoesm/grids/halo_latlon.py:988:    cell_wide = widen_band_cell_fields(
packages/core/legoesm/grids/halo_latlon.py-989-        tuple(getattr(geom, n) for n in cell_names), halo, clamp_poles=True)
packages/core/legoesm/grids/halo_latlon.py:990:    vface_wide = widen_band_vface_fields(
packages/core/legoesm/grids/halo_latlon.py-991-        tuple(getattr(geom, n) for n in vface_names), halo, clamp_poles=True)
packages/core/legoesm/grids/halo_latlon.py-992-    updates = dict(zip(cell_names, cell_wide))
packages/core/legoesm/grids/halo_latlon.py-993-    updates.update(zip(vface_names, vface_wide))
error: pathspec 'tests/distributed/test_latlon_2d_fused_pad.py' did not match any file(s) known to git
Did you forget to 'git add'?

codex
The implementation itself looks sound for the established 2-D contract:

- Lat-only is correct: `pad_with_pole_bc_lat_2d` explicitly leaves longitude untouched, and lon ghosts are separately dispatched.
- The `floor(n_lat_global / proc_lat)` guard is the right 2-D guard; a rank’s local size alone can miss a smaller neighbouring lat block.
- Pole/interior branching matches `_pad_lat_wall_2d`.
- The lon-ring tags do not collide with N/S traffic: N/S and E/W peers are disjoint when `proc_lon > 1`; when `proc_lon == 1`, lon is local. This remains true in the VJP. Rank 512 is also far below the numeric `3000` lon-tag value.

For the fused schedule: current production callers use the same statically ordered field list on all ranks. N/S peers share `proc_col`, hence the same local lon width even for uneven splits, so their dtype groups and packed sizes agree. But the helper’s “trace-deterministic” comment is not a proof across ranks: arbitrary callers could supply differing schemas. Document that as an MPI collective precondition.

VERDICT: FIX-FIRST

1. The new test is currently untracked, so it is absent from `git diff HEAD`; it also is not invoked by [the MPI workflow](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/.github/workflows/mpi-distributed.yml:42). Add it and run it under MPI in CI.

2. The claimed mixed-dtype test is ineffective by default. [The test](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/tests/distributed/test_latlon_2d_fused_pad.py:12) never enables x64, while CI intentionally defaults to fp32. Its requested `float64` field will downcast to float32, leaving one dtype group. Enable `jax_enable_x64` before creating arrays or set `JAX_ENABLE_X64=1` for this MPI test.

3. Coverage is too narrow for a 2-D fused path. [The layout](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/tests/distributed/test_latlon_2d_fused_pad.py:24) forces `proc_lon=1`, so it never exercises a true pencil, a simultaneous N/S-plus-E/W communication program, or the relevant tag argument. Add at least a `2×3` / six-rank case and compose a lon exchange after the fused lat pad.

4. The equivalence test would catch incorrect within-group packing/splitting—its same-dtype fields differ in shape and data—but not dtype-group behavior until issue 2 is fixed, nor a consistent whole-group reordering. The finite-only gradient check is also weak; compare fused and per-field gradients under a global loss, following the stronger existing 2-D wall-pad test.

5. [The existing dispatch test](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/tests/parallel/test_latlon_2d_dispatch_serial.py:137) now has a stale “per-field / band-only” docstring and cannot distinguish fusion from fallback by output. Update it and add a dispatch-level assertion/spying test.

6. No `__all__` work is needed: neither module uses one for this API, and the band sibling is likewise module-local. The new function’s docstring is adequate; add only the cross-rank identical-schema precondition.
tokens used
99,591
The implementation itself looks sound for the established 2-D contract:

- Lat-only is correct: `pad_with_pole_bc_lat_2d` explicitly leaves longitude untouched, and lon ghosts are separately dispatched.
- The `floor(n_lat_global / proc_lat)` guard is the right 2-D guard; a rank’s local size alone can miss a smaller neighbouring lat block.
- Pole/interior branching matches `_pad_lat_wall_2d`.
- The lon-ring tags do not collide with N/S traffic: N/S and E/W peers are disjoint when `proc_lon > 1`; when `proc_lon == 1`, lon is local. This remains true in the VJP. Rank 512 is also far below the numeric `3000` lon-tag value.

For the fused schedule: current production callers use the same statically ordered field list on all ranks. N/S peers share `proc_col`, hence the same local lon width even for uneven splits, so their dtype groups and packed sizes agree. But the helper’s “trace-deterministic” comment is not a proof across ranks: arbitrary callers could supply differing schemas. Document that as an MPI collective precondition.

VERDICT: FIX-FIRST

1. The new test is currently untracked, so it is absent from `git diff HEAD`; it also is not invoked by [the MPI workflow](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/.github/workflows/mpi-distributed.yml:42). Add it and run it under MPI in CI.

2. The claimed mixed-dtype test is ineffective by default. [The test](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/tests/distributed/test_latlon_2d_fused_pad.py:12) never enables x64, while CI intentionally defaults to fp32. Its requested `float64` field will downcast to float32, leaving one dtype group. Enable `jax_enable_x64` before creating arrays or set `JAX_ENABLE_X64=1` for this MPI test.

3. Coverage is too narrow for a 2-D fused path. [The layout](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/tests/distributed/test_latlon_2d_fused_pad.py:24) forces `proc_lon=1`, so it never exercises a true pencil, a simultaneous N/S-plus-E/W communication program, or the relevant tag argument. Add at least a `2×3` / six-rank case and compose a lon exchange after the fused lat pad.

4. The equivalence test would catch incorrect within-group packing/splitting—its same-dtype fields differ in shape and data—but not dtype-group behavior until issue 2 is fixed, nor a consistent whole-group reordering. The finite-only gradient check is also weak; compare fused and per-field gradients under a global loss, following the stronger existing 2-D wall-pad test.

5. [The existing dispatch test](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/tests/parallel/test_latlon_2d_dispatch_serial.py:137) now has a stale “per-field / band-only” docstring and cannot distinguish fusion from fallback by output. Update it and add a dispatch-level assertion/spying test.

6. No `__all__` work is needed: neither module uses one for this API, and the band sibling is likewise module-local. The new function’s docstring is adequate; add only the cross-rank identical-schema precondition.
