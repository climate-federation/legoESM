"""Halo-aware slow-tendency for the plane NH dycore.

Mirror of :func:`compressible_euler_plane.plane_compressible_euler_slow_tendencies`
but operates on a LOCAL per-rank slab + uses the halo-aware
operators from :mod:`plane_operators_halo`. Halo exchange happens
via :func:`packed_exchange_halo_plane_yxz` calls so MPI message
count is independent of stencil count.

JIT compatibility (Codex iter-2 caveat)
---------------------------------------
``packed_exchange_halo_plane_yxz`` is jit-safe on the single-rank
path (uses ``jnp.pad(mode='wrap')``) but the multi-rank
``mpi4jax.sendrecv`` branch contains Python control flow that
can raise ``ConcretizationTypeError`` when traced. Callers
intending to ``jax.jit`` the multi-rank tendency MUST verify
their mpi4jax version supports tracer-input sendrecv, OR call
the slow-tendency in eager mode (no jit) on multi-rank runs.
Single-rank jit is verified by ``test_halo_jit_compilable``.

Coverage (matches the original):

* C-grid pressure gradient (cell→face interp + face gradient)
* Coriolis (if ``use_coriolis``)
* Mass continuity (-div(ρ·u))
* Theta advective transport (1st-order upwind)
* Momentum advection (Arakawa-C upwind, corner averages)
* Vertical advection (column-local, no halo needed)
* Rayleigh sponge (column-local, no halo)
* Biharmonic hyperdiffusion (`laplacian²`)
* Tracer advective transport (vmap over tracer axis)

Smagorinsky LES + vertical-θ diffusion (R4/R5)
----------------------------------------------
* Smagorinsky LES (``smagorinsky_cs > 0``) — full 3D strain tensor
  (S11, S22, S33, S12, S13, S23) on already-halo-padded u/v/w via
  :func:`_compute_smagorinsky_K_m_plane_halo`. K_m is exchanged once
  before driving :func:`oh.variable_K_diffusion_vlast_halo` on each
  prognostic at its native Arakawa-C staggering.
* Vertical θ Laplacian (``vertical_theta_diffusion > 0``) — column-
  local, no halo exchange. Mirrors the serial branch at
  ``compressible_euler_plane.py:1050-1064`` and is applied in both
  the main entry point and the split-trace phase-2 variant.

Higher-order horizontal advection
---------------------------------
* ``config.horizontal_advection_scheme == "weno5"`` switches theta /
  u / v / w horizontal advection to the 5th-order WENO-Z stencil. The
  positive-definite TRACERS are instead kept on the monotone van_leer
  limiter (WENO5 is NOT positivity-preserving — see the POSITIVITY GUARD
  in the slow-tendency), mirroring the serial dycore.
  Requires ``layout.halo >= 3`` (6-point WENO5 reconstruction
  reaches ±3 cells on each axis). Single-rank build with
  ``make_plane_pencil_layout(..., halo=3)`` is bit-identical to the
  serial ``_weno5_advection_x/y`` (tests in
  ``test_weno5_halo_equiv.py``).
* ``config.horizontal_advection_scheme == "van_leer"`` (iter-184)
  switches the same fields to the 2nd-order Van Leer TVD stencil.
  Requires ``layout.halo >= 2`` (4-point reconstruction). Single-rank
  build with ``make_plane_pencil_layout(..., halo=2)`` is bit-identical
  to the serial ``_van_leer_advection_x/y`` (tests in
  ``test_van_leer_halo_equiv.py``). This is the iter-183 production
  default since it gives 3x wall-time speedup over upwind1 at dt=20.

Single-rank equivalence
-----------------------
With ``layout.n_ranks == 1``, the halo exchange devolves to
``jnp.pad(mode='wrap')`` and the halo-aware operators reduce to
the original ``jnp.roll`` stencils via slice arithmetic — the
return value is BIT-IDENTICAL to the non-halo dycore.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.atmosphere.dynamics import (
    plane_operators_halo as oh,
)
from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig, sponge_profile, compute_exner_perturbation,
)
from legoesm.atmosphere.dynamics.compressible_euler_plane import (
    full_level_centred_d_dz, moisture_buoyancy_w_half,
    safe_sqrt_strain, sgs_brunt_vaisala_sq,
    vertical_advection_van_leer_plane,
)
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm.core.state import (
    PlaneNonHydrostaticState, PlaneNonHydrostaticTendencies,
)
from legoesm.grids.plane import PlaneGrid
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.parallel.plane_mpi import (
    PlanePencilLayout, packed_exchange_halo_plane_yxz,
)


def _c_pd_constant():
    from legoesm import constants
    return constants.c_pd


def _vertical_advection_plane(
    field_yxz: jax.Array,
    w_yxz_half: jax.Array,
    hc: HeightCoordinate,
    J: jax.Array,
) -> jax.Array:
    """Vertical advection ``-w · df/dz``. Column-local (no halo)."""
    # Centered-difference on full levels, w averaged to full level.
    w_full = 0.5 * (w_yxz_half[..., :-1] + w_yxz_half[..., 1:])
    nlev = field_yxz.shape[-1]
    if nlev <= 2:
        return jnp.zeros_like(field_yxz)
    dz_half = hc.dz_half
    dz_centered = dz_half[:-1] + dz_half[1:]
    inner_grad = (field_yxz[..., :-2] - field_yxz[..., 2:]) / dz_centered
    pad_axes = ((0, 0),) * (field_yxz.ndim - 1)
    dfield_dz = jnp.pad(inner_grad, (*pad_axes, (1, 1)))
    return -w_full / J[..., None] * dfield_dz


def _compute_smagorinsky_K_m_plane_halo(
    u_pad: jax.Array,
    v_pad: jax.Array,
    w_pad_half: jax.Array,
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    c_s: float,
    halo: int = 1,
    n2_sgs: jax.Array | None = None,
    prandtl: float = 1.0,
    wall_damping: bool = True,
    delta_max: float = 1.0e30,
    stability_length: bool = False,
) -> jax.Array:
    """Halo-aware Smagorinsky-Lilly K_m at interior cell centres.

    Mirror of
    :func:`compressible_euler_plane._compute_smagorinsky_K_m_plane`
    operating on already-halo-padded u, v, w slabs. Vertical gradients
    are column-local — reuses :func:`full_level_centred_d_dz` +
    :func:`safe_sqrt_strain` from the serial module so the inner
    arithmetic stays in one canonical implementation.

    The SAM ``dosmagor`` stratification correction
    ``K_m = (Cs·Δ)²·sqrt(max(0, |S|² − Pr·N²))`` is applied identically
    to the serial kernel.  ``N²`` is COLUMN-LOCAL (no horizontal
    neighbours), so the precomputed sub-grid ``n2_sgs`` slab from the
    shared :func:`compressible_euler_plane.sgs_brunt_vaisala_sq` is
    passed at the rank-local INTERIOR shape ``(ny, nx, nlev)`` — no halo
    exchange of N² needed.  ``n2_sgs=None`` recovers the pure-strain
    form.  Both kernels consuming the one shared N² helper prevents
    serial/MPI divergence (Codex iter-1 adversarial-review HIGH).

    Bit-equivalent to the serial K_m when ``layout.n_ranks == 1``:
    ``jnp.pad(mode='wrap')`` produces the same neighbour values as
    ``jnp.roll(±1)``, and the slice indices below pick the same
    neighbours.

    Inputs are padded as ``(ny+2h, nx+2h, nlev[+1])``; the returned
    K_m is at the interior cell-centre shape ``(ny, nx, nlev)``.
    """
    h = halo

    # u at x-face (i, j); +1 in x: index h+1.
    u_int = u_pad[h:-h, h:-h, :]
    u_xp1 = u_pad[h:-h, h + 1 : (-h + 1) if h > 1 else None, :]
    du_dx_center = (u_xp1 - u_int) / grid.dx

    v_int = v_pad[h:-h, h:-h, :]
    v_yp1 = v_pad[h + 1 : (-h + 1) if h > 1 else None, h:-h, :]
    dv_dy_center = (v_yp1 - v_int) / grid.dy

    # S12 at SW corner (cell (i, j)):
    # du/dy = (u[i, j] - u[i, j-1]) / dy;  dv/dx = (v[i, j] - v[i-1, j]) / dx.
    u_ym1 = u_pad[h - 1 : -h - 1, h:-h, :]
    v_xm1 = v_pad[h:-h, h - 1 : -h - 1, :]
    du_dy_corner = (u_int - u_ym1) / grid.dy
    dv_dx_corner = (v_int - v_xm1) / grid.dx
    S12_corner = 0.5 * (du_dy_corner + dv_dx_corner)

    # S12 at the SE / NW / NE corners (shifted +1 in x, +1 in y, +1 in both).
    u_xp1_ym1 = u_pad[h - 1 : -h - 1, h + 1 : (-h + 1) if h > 1 else None, :]
    v_xp1_int = v_pad[h:-h, h + 1 : (-h + 1) if h > 1 else None, :]
    du_dy_xp1 = (u_xp1 - u_xp1_ym1) / grid.dy
    dv_dx_xp1 = (v_xp1_int - v_int) / grid.dx
    S12_corner_xp1 = 0.5 * (du_dy_xp1 + dv_dx_xp1)

    u_yp1 = u_pad[h + 1 : (-h + 1) if h > 1 else None, h:-h, :]
    v_yp1_xm1 = v_pad[h + 1 : (-h + 1) if h > 1 else None, h - 1 : -h - 1, :]
    du_dy_yp1 = (u_yp1 - u_int) / grid.dy
    dv_dx_yp1 = (v_yp1 - v_yp1_xm1) / grid.dx
    S12_corner_yp1 = 0.5 * (du_dy_yp1 + dv_dx_yp1)

    u_yp1_xp1 = u_pad[h + 1 : (-h + 1) if h > 1 else None,
                      h + 1 : (-h + 1) if h > 1 else None, :]
    v_yp1_xp1 = v_pad[h + 1 : (-h + 1) if h > 1 else None,
                      h + 1 : (-h + 1) if h > 1 else None, :]
    du_dy_yp1_xp1 = (u_yp1_xp1 - u_xp1) / grid.dy
    dv_dx_yp1_xp1 = (v_yp1_xp1 - v_yp1) / grid.dx
    S12_corner_yp1_xp1 = 0.5 * (du_dy_yp1_xp1 + dv_dx_yp1_xp1)

    S12_sq_center = 0.25 * (
        S12_corner ** 2 + S12_corner_xp1 ** 2
        + S12_corner_yp1 ** 2 + S12_corner_yp1_xp1 ** 2
    )

    # ---- Vertical strain components (column-local) ----
    w_int_half = w_pad_half[h:-h, h:-h, :]              # (ny, nx, nlev+1)
    dz_full = height_coord.dz
    dw_dz_center = (
        w_int_half[..., 1:] - w_int_half[..., :-1]
    ) / dz_full
    S33_center = dw_dz_center

    du_dz_int = full_level_centred_d_dz(u_int, height_coord)
    dv_dz_int = full_level_centred_d_dz(v_int, height_coord)

    # ∂w/∂x at x-face (i, j) needs w_full(i-1, j); ∂w/∂y at y-face needs
    # w_full(i, j-1). Reuse the padded w to slice both shifts.
    w_full_pad = 0.5 * (w_pad_half[..., :-1] + w_pad_half[..., 1:])
    w_full_int = w_full_pad[h:-h, h:-h, :]
    w_full_xm1 = w_full_pad[h:-h, h - 1 : -h - 1, :]
    w_full_ym1 = w_full_pad[h - 1 : -h - 1, h:-h, :]
    dw_dx_xface = (w_full_int - w_full_xm1) / grid.dx
    dw_dy_yface = (w_full_int - w_full_ym1) / grid.dy

    S13_xface = 0.5 * (du_dz_int + dw_dx_xface)
    S23_yface = 0.5 * (dv_dz_int + dw_dy_yface)

    # Face → cell-centre via two-point average. Need S13 at (i+1, j)
    # — pull from +1-in-x shifted positions.
    w_full_xp1 = w_full_pad[h:-h, h + 1 : (-h + 1) if h > 1 else None, :]
    u_xp1_int = u_xp1                                   # already sliced
    du_dz_xp1 = full_level_centred_d_dz(u_xp1_int, height_coord)
    dw_dx_xface_xp1 = (w_full_xp1 - w_full_int) / grid.dx
    S13_xface_xp1 = 0.5 * (du_dz_xp1 + dw_dx_xface_xp1)
    S13_sq_center = 0.5 * (S13_xface ** 2 + S13_xface_xp1 ** 2)

    w_full_yp1 = w_full_pad[h + 1 : (-h + 1) if h > 1 else None, h:-h, :]
    v_yp1_int = v_yp1                                   # already sliced
    dv_dz_yp1 = full_level_centred_d_dz(v_yp1_int, height_coord)
    dw_dy_yface_yp1 = (w_full_yp1 - w_full_int) / grid.dy
    S23_yface_yp1 = 0.5 * (dv_dz_yp1 + dw_dy_yface_yp1)
    S23_sq_center = 0.5 * (S23_yface ** 2 + S23_yface_yp1 ** 2)

    strain_mag_sq = 2.0 * (
        du_dx_center ** 2 + dv_dy_center ** 2 + S33_center ** 2
        + 2.0 * S12_sq_center + 2.0 * S13_sq_center + 2.0 * S23_sq_center
    )

    from legoesm import constants
    # SAM-faithful horizontal-spacing cap (tke_full.f90:42 delta_max=1000 m);
    # mirrors the serial _compute_smagorinsky_K_m_plane so serial=MPI parity holds.
    dx_eff = jnp.minimum(delta_max, grid.dx)
    dy_eff = jnp.minimum(delta_max, grid.dy)
    delta = (dx_eff * dy_eff * dz_full) ** (1.0 / 3.0)
    l_smag = c_s * delta
    # ``wall_damping`` mirrors the serial path: SAM dosmagor uses smix=grd (no
    # von-Kármán cap), so the SAM-faithful CRM runs pass False. Kept identical
    # to the serial so serial==halo holds for BOTH values.
    if wall_damping:
        l_wall = constants.kappa_von_karman * height_coord.z_full
        l_m = jnp.minimum(l_smag, l_wall)
    else:
        l_m = l_smag
    l_m_sq = l_m ** 2

    # SAM dosmagor stratification (Lilly) correction — see the serial
    # _compute_smagorinsky_K_m_plane. N² precomputed column-local by the
    # shared sgs_brunt_vaisala_sq helper and passed in as n2_sgs.
    if n2_sgs is not None:
        strain_arg = strain_mag_sq - prandtl * n2_sgs
    else:
        strain_arg = strain_mag_sq
    strain_mag = safe_sqrt_strain(strain_arg)
    if stability_length and n2_sgs is not None:
        # SAM dosmagor stable-layer Deardorff mixing-length limit — mirror of
        # the serial _compute_smagorinsky_K_m_plane (SGS_TKE/tke_full.f90:
        # 285-298): smix shrinks where N²>0, Cee=Ce1+Ce2·(smix/grd),
        # tk=√(Ck³/Cee·(|S|²−Pr·N²))·smix². Reduces to (Cs·grd)²·|S| where N²≤0.
        Ck = 0.1
        Ce = Ck ** 3 / c_s ** 4
        tk_grd = c_s ** 2 * delta ** 2 * strain_mag
        n2_pos = jnp.maximum(n2_sgs, 1.0e-10)
        smix_stable = jnp.minimum(delta, jnp.maximum(
            0.1 * delta,
            jnp.sqrt(0.76 * tk_grd / (Ck * jnp.sqrt(n2_pos)))))
        smix = jnp.where(n2_sgs > 0.0, smix_stable, delta)
        ratio = smix / jnp.clip(delta, 1.0e-12, None)
        Cee = Ce / 0.7 * (0.19 + 0.51 * ratio)
        return jnp.sqrt(Ck ** 3 / Cee) * smix ** 2 * strain_mag
    return l_m_sq * strain_mag


def precompute_coriolis_halo(
    grid: PlaneGrid, layout: PlanePencilLayout,
) -> jax.Array:
    """Pre-exchange grid.f_y for the halo path.

    f_y is static — caller should compute this once at model init
    and pass into plane_compressible_euler_slow_tendencies_halo
    via ``f_pad_cached`` so we avoid one MPI round per slow-tendency
    call (Codex iter-3 perf finding).
    """
    f_3d = grid.f_y[:, :, None]
    f_pad, = packed_exchange_halo_plane_yxz(f_3d, layout=layout)
    return f_pad


def _global_hmean_plane(f_int: jax.Array, layout: PlanePencilLayout):
    """Global horizontal mean of an interior ``(ny, nx, nlev)`` field.

    Returns a ``(1, 1, nlev)`` profile that broadcasts against the
    interior field.  Used for the moist-buoyancy perturbation
    (:func:`moisture_buoyancy_w_half`) which subtracts the SAM ``qv0``
    base state = horizontal mean.

    ``n_ranks == 1`` takes the LOCAL ``jnp.sum`` path (no MPI stack
    required, so single-process pytest works and the result is bit-
    identical to the serial ``jnp.mean``); ``n_ranks > 1`` allreduces the
    rank-local sum via the AD-safe ``global_sum_mpi`` (full VJP) so the
    mean is the true global domain mean.  The Python branch on the static
    ``layout.n_ranks`` is a compile-time constant (no traced control flow).
    """
    local_sum = jnp.sum(f_int, axis=(0, 1))                     # (nlev,)
    n_global = layout.ny_global * layout.nx_global
    if layout.n_ranks > 1:
        from legoesm.parallel.reductions import global_sum_mpi
        total = global_sum_mpi(local_sum)
    else:
        total = local_sum
    return (total / n_global)[None, None, :]


def _require_single_rank_closure(closure: str) -> None:
    """Raise if ``closure`` is a single-rank-only SGS closure on a halo path.

    Vreman and AMD use A-grid centred (roll-±1) gradients that would need a
    halo-2 stencil to stay bit-equal to the serial kernel; like the dynamic
    Smagorinsky closure they have NO MPI-halo K_m kernel.  Both halo entry points
    (:func:`plane_compressible_euler_slow_tendencies_halo` and
    :func:`slow_tendency_jit_split`) call this at ENTRY so a multi-rank config
    fails cleanly BEFORE any halo exchange, not after partial MPI work.
    """
    if closure in ("vreman", "amd"):
        raise NotImplementedError(
            f"turbulence_closure={closure!r} is single-rank only (no MPI-halo "
            "kernel; the dynamic closures are likewise serial-only). Run on one "
            "rank, or use turbulence_closure='smagorinsky' under MPI.")


def plane_compressible_euler_slow_tendencies_halo(
    state: PlaneNonHydrostaticState,
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    config: CompressibleEulerConfig,
    layout: PlanePencilLayout,
    f_pad_cached: jax.Array | None = None,
) -> PlaneNonHydrostaticTendencies:
    """Halo-aware slow-tendency for the plane NH dycore.

    Inputs: rank-local slabs. Output: rank-local interior-shape
    tendencies. Single packed MPI round per call covers all
    horizontally-coupled prognostic fields.

    Parameters
    ----------
    f_pad_cached : optional pre-exchanged f_y halo array (use
        :func:`precompute_coriolis_halo` once at init). When None
        and ``config.use_coriolis``, f_y is exchanged per call
        (correct but adds 1 MPI round).

    JIT safety (Codex iter-2)
    -------------------------
    Hard guard raises ``RuntimeError`` if called under a JIT
    trace with ``layout.n_ranks > 1``. mpi4jax sendrecv branch
    may raise ``ConcretizationTypeError`` under tracing on some
    mpi4jax versions; we prevent silent crashes deep in the
    compiled graph by failing early.
    """
    # Smag + vertical-θ-diffusion supported on the halo path; see
    # _compute_smagorinsky_K_m_plane_halo above. Branches are gated
    # by Python ``if`` on static config values — no traced cost when
    # disabled.
    # Multi-rank under JIT is now SUPPORTED on Linux/MPICH (the original
    # ConcretizationTypeError came from ``int(jnp.prod(jnp.asarray(...)))``
    # on static shapes in ``plane_mpi.exchange_halo_plane_yxz`` — replaced
    # with ``math.prod`` (jit-safe).  mpi4jax ``sendrecv`` traces inside
    # ``@jax.jit`` here exactly as ``voronoi_mpi.make_voronoi_mpi_step``
    # does at scale.  The eager multi-rank path was ~100× slower than np=1
    # (per-op + per-sendrecv host sync); ``step_halo`` now JITs the
    # split-explicit core.  (The macOS shared-mem mpi4jax crash mode that
    # motivated this guard does not occur on the Linux/MPICH stack.)

    # Fail BEFORE any halo exchange / partial-tendency compute below if the
    # closure has no MPI-halo kernel (vreman/amd): a clean early failure, not
    # one raised after several MPI rounds.
    _require_single_rank_closure(
        getattr(config, "turbulence_closure", "smagorinsky"))

    h = layout.halo if hasattr(layout, "halo") else 1

    u = state.u.data
    v = state.v.data
    w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data

    c_p = jnp.asarray(0.0, dtype=u.dtype) + _c_pd_constant()
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    J = terrain_metric.jacobian

    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p, rho_0 + rho_p,
    )
    pi_p = compute_exner_perturbation(rho_p, theta_p, height_coord)

    # ONE packed halo exchange for every horizontally-coupled field
    # used in slow-tendency computations.
    fields_to_exchange = [
        u, v, theta_p, rho_p, pi_p, theta_total, rho_total,
    ]
    (
        u_pad, v_pad, theta_p_pad, rho_p_pad, pi_p_pad,
        theta_total_pad, rho_total_pad,
    ) = packed_exchange_halo_plane_yxz(
        *fields_to_exchange, layout=layout,
    )

    # 2. C-grid pressure gradient.
    grad_pi_x = oh.grad_x_vlast_halo(pi_p_pad, grid, h)
    grad_pi_y = oh.grad_y_vlast_halo(pi_p_pad, grid, h)
    theta_xface = oh.interp_cell_to_xface_vlast_halo(theta_total_pad, grid, h)
    theta_yface = oh.interp_cell_to_yface_vlast_halo(theta_total_pad, grid, h)
    du_pg = -c_p * theta_xface * grad_pi_x
    dv_pg = -c_p * theta_yface * grad_pi_y

    # 3. Coriolis (need v interp to x-face, u interp to y-face).
    if config.use_coriolis:
        # Codex iter-3 (perf): use cached f_pad if caller pre-computed
        # it (preferred — avoids per-step exchange of a static field).
        # Fall back to per-step exchange if not cached.
        if f_pad_cached is not None:
            f_pad = f_pad_cached
        else:
            # Codex iter-2 (correctness): f_y is y-dependent beta-plane
            # field on GLOBAL grid. Local jnp.pad(wrap) would wrap each
            # rank's slab → wrong f on north/south halos under MPI.
            # Use packed_exchange_halo_plane_yxz so the halo carries
            # global-grid f_y from neighbor rank.
            f_3d = grid.f_y[:, :, None]
            f_pad, = packed_exchange_halo_plane_yxz(f_3d, layout=layout)
        f_xface = oh.interp_cell_to_xface_vlast_halo(f_pad, grid, h)
        f_yface = oh.interp_cell_to_yface_vlast_halo(f_pad, grid, h)
        v_xface = oh.interp_yface_to_xface_vlast_halo(v_pad, grid, h)
        u_yface = oh.interp_xface_to_yface_vlast_halo(u_pad, grid, h)
        # SAM coriolis.f90: f acts on the departure from the geostrophic
        # reference (ug0, vg0); None ⇒ 0 (RCE). See the serial
        # compressible_euler_plane Coriolis block for the rationale.
        v_ref = (0.0 if height_coord.v_geo0 is None
                 else height_coord.v_geo0[None, None, :])
        u_ref = (0.0 if height_coord.u_geo0 is None
                 else height_coord.u_geo0[None, None, :])
        du_cor = f_xface * (v_xface - v_ref)
        dv_cor = -f_yface * (u_yface - u_ref)
    else:
        du_cor = jnp.zeros_like(u)
        dv_cor = jnp.zeros_like(v)

    # 4. Mass continuity: -div(ρ·u). rho_xface + rho_yface live at
    #    interior face positions; combine with u_pad (already padded)
    #    by extracting u's interior, multiplying, then re-padding.
    rho_xface_int = oh.interp_cell_to_xface_vlast_halo(
        rho_total_pad, grid, h,
    )
    rho_yface_int = oh.interp_cell_to_yface_vlast_halo(
        rho_total_pad, grid, h,
    )
    u_int = u_pad[h:-h, h:-h, :]
    v_int = v_pad[h:-h, h:-h, :]
    flux_u_int = rho_xface_int * u_int
    flux_v_int = rho_yface_int * v_int
    # Single MPI round to repad both fluxes for divergence.
    flux_u_pad, flux_v_pad = packed_exchange_halo_plane_yxz(
        flux_u_int, flux_v_int, layout=layout,
    )
    drho_p_dt = -oh.divergence_vlast_halo(flux_u_pad, flux_v_pad, grid, h)

    # 5. Theta advection.
    u_center = oh.interp_xface_to_cell_vlast_halo(u_pad, grid, h)
    v_center = oh.interp_yface_to_cell_vlast_halo(v_pad, grid, h)
    # Need u_center, v_center to be padded for use with the halo-aware
    # upwind. Re-pad the interior result.
    u_center_pad = _re_pad_halo(u_center, layout, h)
    v_center_pad = _re_pad_halo(v_center, layout, h)
    # Choose horizontal advection scheme. iter-187 consults the
    # public HORIZONTAL_ADVECTION_HALO_REQUIREMENT map from
    # compressible_euler_plane so the halo-width contract lives in
    # ONE place; the driver script (run_rce_mpi_long.py) reads the
    # same map at layout construction.
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        HORIZONTAL_ADVECTION_HALO_REQUIREMENT,
    )
    scheme = getattr(config, "horizontal_advection_scheme", "upwind1")
    if scheme not in HORIZONTAL_ADVECTION_HALO_REQUIREMENT:
        raise ValueError(
            f"Unknown horizontal_advection_scheme: {scheme!r}. "
            f"Expected one of "
            f"{sorted(HORIZONTAL_ADVECTION_HALO_REQUIREMENT)}."
        )
    # ADV-SPLIT (#86): the per-field momentum/scalar advection split is wired in
    # the SERIAL slow-tendency path; the MPI halo path here still applies ONE
    # scheme to both legs. Fail LOUD rather than silently diverge serial≠MPI
    # (CLAUDE.md: serial/MPI bit-identical) until the split is wired here too.
    _mscheme = getattr(config, "horizontal_momentum_advection_scheme", None)
    if _mscheme is not None and _mscheme != scheme:
        raise NotImplementedError(
            "horizontal_momentum_advection_scheme (ADV-SPLIT #86) is not yet "
            "wired into the MPI halo dycore path — it would silently diverge "
            "from the serial split. Use the serial (single-rank) path, or set "
            "horizontal_momentum_advection_scheme=None for MPI runs."
        )
    # SGS-VERT (#81): the VERTICAL SGS flux ∂_z(K ∂_z φ) is wired in the serial
    # path but NOT here. Fail LOUD rather than silently diverge serial≠MPI (the
    # horizontal-only #80 tracer SGS IS mirrored below). Wiring needs a
    # column-local vertical-flux call on u/v/θ'/tracers/w + a parity test.
    if getattr(config, "sgs_vertical_diffusion", False):
        raise NotImplementedError(
            "sgs_vertical_diffusion (SGS-VERT #81) is not yet wired into the "
            "MPI halo dycore path (vertical SGS flux is serial-only) — it would "
            "silently diverge from the serial path. Use the serial (single-rank) "
            "path, or set sgs_vertical_diffusion=False for MPI runs."
        )
    _required_halo = HORIZONTAL_ADVECTION_HALO_REQUIREMENT[scheme]
    if h < _required_halo:
        raise ValueError(
            f"horizontal_advection_scheme={scheme!r} requires "
            f"layout.halo >= {_required_halo}; got halo={h}. "
            f"Construct the layout with "
            f"make_plane_pencil_layout(..., halo={_required_halo}) "
            f"or rely on the driver's iter-187 auto-selection."
        )
    if scheme == "weno5":
        adv_x = lambda f_pad, u_pad_, dx_, h_: (
            oh.weno5_advection_x_halo(f_pad, u_pad_, dx_, h_)
        )
        adv_y = lambda f_pad, v_pad_, dy_, h_: (
            oh.weno5_advection_y_halo(f_pad, v_pad_, dy_, h_)
        )
    elif scheme == "van_leer":
        adv_x = lambda f_pad, u_pad_, dx_, h_: (
            oh.van_leer_advection_x_halo(f_pad, u_pad_, dx_, h_)
        )
        adv_y = lambda f_pad, v_pad_, dy_, h_: (
            oh.van_leer_advection_y_halo(f_pad, v_pad_, dy_, h_)
        )
    elif scheme == "upwind1":
        adv_x = oh.upwind_advection_x_halo
        adv_y = oh.upwind_advection_y_halo
    else:
        # iter-192 Codex MEDIUM: a fourth scheme added to
        # HORIZONTAL_ADVECTION_HALO_REQUIREMENT but NOT wired into
        # this dispatch would silently fall through to upwind on the
        # halo path. Refuse loudly so the failure mode matches the
        # serial path (compressible_euler_plane.py raises identically).
        raise NotImplementedError(
            f"horizontal_advection_scheme={scheme!r} is in "
            f"HORIZONTAL_ADVECTION_HALO_REQUIREMENT but has no halo-path "
            f"wiring in compressible_euler_plane_halo.py. Add the "
            f"adv_x / adv_y branch alongside upwind1 / van_leer / weno5."
        )
    # POSITIVITY GUARD (codex CRM-dycore review): mirror the serial dycore —
    # positive-definite TRACERS use the monotone van_leer limiter under weno5
    # (θ′ below keeps weno5). van_leer_*_halo accepts halo>=2 and, given
    # weno5's halo=3 wrap-padded inputs, picks the SAME neighbours as the
    # serial van_leer ⇒ serial==MPI stays bit-identical. Non-weno5 schemes are
    # already monotone (van_leer/upwind1) so tadv_* == adv_*.
    if scheme == "weno5":
        tadv_x = lambda f_pad, u_pad_, dx_, h_: (
            oh.van_leer_advection_x_halo(f_pad, u_pad_, dx_, h_)
        )
        tadv_y = lambda f_pad, v_pad_, dy_, h_: (
            oh.van_leer_advection_y_halo(f_pad, v_pad_, dy_, h_)
        )
    else:
        tadv_x, tadv_y = adv_x, adv_y
    # VERTICAL tracer scheme (codex CRM-dycore review): mirror the serial
    # dispatch so vertical_tracer_advection="van_leer" is honored on the MPI
    # path too — previously the halo tracer leg was hardcoded to centered,
    # silently diverging from serial (van_leer is the RCE runner default).
    # Vertical advection is column-local (no halo), so the serial van_leer
    # kernel applies bit-identically to the interior tracer q_int.
    _vert_tracer_scheme = getattr(
        config, "vertical_tracer_advection", "centered")
    if _vert_tracer_scheme == "van_leer":
        _vertical_tracer_adv = vertical_advection_van_leer_plane
    elif _vert_tracer_scheme == "centered":
        _vertical_tracer_adv = _vertical_advection_plane
    else:
        raise ValueError(
            f"Unknown vertical_tracer_advection: {_vert_tracer_scheme!r}. "
            f"Expected 'centered' or 'van_leer'."
        )
    dtheta_p_dt = (
        adv_x(theta_total_pad, u_center_pad, grid.dx, h)
        + adv_y(theta_total_pad, v_center_pad, grid.dy, h)
    )

    # 6. Horizontal momentum advection — u advected by (u, v_at_xface),
    #    v advected by (u_at_yface, v).
    v_at_xface = oh.interp_yface_to_xface_vlast_halo(v_pad, grid, h)
    u_at_yface = oh.interp_xface_to_yface_vlast_halo(u_pad, grid, h)
    v_at_xface_pad = _re_pad_halo(v_at_xface, layout, h)
    u_at_yface_pad = _re_pad_halo(u_at_yface, layout, h)
    du_adv = (
        adv_x(u_pad, u_pad, grid.dx, h)
        + adv_y(u_pad, v_at_xface_pad, grid.dy, h)
    )
    dv_adv = (
        adv_x(v_pad, u_at_yface_pad, grid.dx, h)
        + adv_y(v_pad, v_pad, grid.dy, h)
    )

    # 7. Vertical advection (column-local).
    du_vert = _vertical_advection_plane(u, w, height_coord, J)
    dv_vert = _vertical_advection_plane(v, w, height_coord, J)

    du_dt = du_adv + du_vert + du_pg + du_cor
    dv_dt = dv_adv + dv_vert + dv_pg + dv_cor

    # 8. w slow part — same scheme as theta + momentum (set above).
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])
    w_full_pad = _re_pad_halo(w_full, layout, h)
    dw_full = (
        adv_x(w_full_pad, u_center_pad, grid.dx, h)
        + adv_y(w_full_pad, v_center_pad, grid.dy, h)
    )
    pad_axes = ((0, 0),) * (dw_full.ndim - 1)
    dw_dt = jnp.pad(
        0.5 * (dw_full[..., :-1] + dw_full[..., 1:]),
        (*pad_axes, (1, 1)),
    )

    # 8b. SAM moist buoyancy on w (shared serial helper). Perturbation
    #     from the GLOBAL horizontal mean (= SAM qv0/qn0/qp0), so it
    #     matches the serial jnp.mean at n_ranks==1 and is the true domain
    #     mean for n_ranks>1. Dry θ' buoyancy stays in the acoustic substep.
    # Gated like the serial path: when acoustic_moist_buoyancy=True (default) the
    # moist buoyancy is added INSIDE the acoustic substeps (the shared plane
    # wrappers), so adding it here too would DOUBLE-COUNT. (At n_ranks==1 the
    # acoustic helper's local mean equals this global mean ⇒ serial==halo parity.)
    if config.moist_buoyancy and not getattr(
            config, "acoustic_moist_buoyancy", True):
        dw_dt = dw_dt + moisture_buoyancy_w_half(
            state.tracers.data, theta_p, height_coord,
            lambda f: _global_hmean_plane(f, layout),
        )

    # 9. Rayleigh sponge (column-local).
    _sponge_shape = getattr(config, "sponge_profile_shape", "sin2")
    sponge_full = sponge_profile(
        height_coord.z_full, height_coord.H,
        config.sponge_width, config.sponge_coeff, shape=_sponge_shape,
    )
    sponge_half = sponge_profile(
        height_coord.z_half, height_coord.H,
        config.sponge_width, config.sponge_coeff, shape=_sponge_shape,
    )
    # ``sponge_w_only`` mirrors the serial path (SAM damping.f90 = w only) so
    # serial==halo holds for BOTH values (iter-47 D4).
    if not getattr(config, "sponge_w_only", False):
        du_dt = du_dt - sponge_full * u
        dv_dt = dv_dt - sponge_full * v
        dtheta_p_dt = dtheta_p_dt - sponge_full * theta_p
        # Codex iter-2026-05 finding: serial path applies rho' sponge too
        # (compressible_euler_plane.py:1008, commit aa0a8d75); halo path
        # was missing it which caused drho_p_dt to diverge by O(1e-4) from
        # the serial reference.
        drho_p_dt = drho_p_dt - sponge_full * rho_p
    dw_dt = dw_dt - sponge_half * w

    # 10. Biharmonic hyperdiffusion. Two laplacian passes →
    #     need re-exchange between passes since the first lap output
    #     loses halo freshness.
    if config.hyperdiff_coeff > 0.0:
        lap_u_int = oh.laplacian_vlast_halo(u_pad, grid, h)
        lap_v_int = oh.laplacian_vlast_halo(v_pad, grid, h)
        lap_theta_int = oh.laplacian_vlast_halo(theta_p_pad, grid, h)
        lap_u_pad, lap_v_pad, lap_theta_pad = packed_exchange_halo_plane_yxz(
            lap_u_int, lap_v_int, lap_theta_int, layout=layout,
        )
        du_dt = du_dt - config.hyperdiff_coeff * oh.laplacian_vlast_halo(
            lap_u_pad, grid, h,
        )
        dv_dt = dv_dt - config.hyperdiff_coeff * oh.laplacian_vlast_halo(
            lap_v_pad, grid, h,
        )
        dtheta_p_dt = dtheta_p_dt - config.hyperdiff_coeff * (
            oh.laplacian_vlast_halo(lap_theta_pad, grid, h)
        )
    if config.hyperdiff_rho_coeff > 0.0:
        lap_rho_int = oh.laplacian_vlast_halo(rho_p_pad, grid, h)
        lap_rho_pad, = packed_exchange_halo_plane_yxz(
            lap_rho_int, layout=layout,
        )
        drho_p_dt = drho_p_dt - config.hyperdiff_rho_coeff * (
            oh.laplacian_vlast_halo(lap_rho_pad, grid, h)
        )
    if config.hyperdiff_w_coeff > 0.0:
        w_pad = _re_pad_halo(w, layout, h)
        lap_w_int = oh.laplacian_vlast_halo(w_pad, grid, h)
        lap_w_pad, = packed_exchange_halo_plane_yxz(
            lap_w_int, layout=layout,
        )
        dw_dt = dw_dt - config.hyperdiff_w_coeff * (
            oh.laplacian_vlast_halo(lap_w_pad, grid, h)
        )

    # 10b. Explicit vertical θ Laplacian dissipation (column-local,
    #      no halo needed). Mirrors the serial branch at
    #      compressible_euler_plane.py:1050-1064.
    nu_v = float(getattr(config, "vertical_theta_diffusion", 0.0))
    if nu_v > 0.0 and theta_p.shape[-1] > 2:
        dz_half = height_coord.dz_half
        dz_avg = 0.5 * (dz_half[:-1] + dz_half[1:])
        d2_inner = (
            theta_p[..., 2:] - 2.0 * theta_p[..., 1:-1] + theta_p[..., :-2]
        ) / (dz_avg ** 2)
        pad_axes_v = ((0, 0),) * (theta_p.ndim - 1)
        d2_theta = jnp.pad(d2_inner, (*pad_axes_v, (1, 1)))
        dtheta_p_dt = dtheta_p_dt + nu_v * d2_theta

    # 11. Smagorinsky-Lilly LES eddy viscosity (R4 — halo port).
    #     Computes K_m on the interior using
    #     :func:`_compute_smagorinsky_K_m_plane_halo`, exchanges K_m
    #     once, then drives the existing
    #     :func:`oh.variable_K_diffusion_vlast_halo` on each prognostic
    #     at its native Arakawa-C staggering.
    # Turbulence-closure mode (DNS-LES, iter-177) — mirrors the serial path so
    # closure="molecular" (DNS) and "smagorinsky" (CRM/LES) stay bit-identical
    # serial vs MPI at n_ranks==1.
    # vreman/amd already rejected at entry (_require_single_rank_closure);
    # here only the halo-supported closures (smagorinsky / molecular) remain.
    _closure = getattr(config, "turbulence_closure", "smagorinsky")
    _use_smag = _closure == "smagorinsky" and config.smagorinsky_cs > 0.0
    _use_mol = (_closure == "molecular"
                and getattr(config, "molecular_viscosity", 0.0) > 0.0)
    if _use_smag or _use_mol:
        w_pad_half, = packed_exchange_halo_plane_yxz(w, layout=layout)
        if _use_mol:
            # DNS: CONSTANT molecular ν on the interior; the SAME halo exchange
            # + operators below then apply (a constant field is trivially
            # halo-consistent, so no strain stencil / N² is needed).
            K_m_int = jnp.full(
                theta_total.shape, config.molecular_viscosity,
                dtype=theta_total.dtype,
            )
            sgs_prandtl = config.molecular_prandtl
        else:
            # Interior N² for the stratification term (column-local — no halo
            # exchange needed). Built by the SAME shared helper the serial call
            # site uses, on the rank-local interior θ/tracers, so the clear↔
            # moist switch and serial/MPI parity hold bit-for-bit at n_ranks==1.
            n2_sgs_h = sgs_brunt_vaisala_sq(
                theta_total, state.tracers.data, height_coord,
            )
            K_m_int = _compute_smagorinsky_K_m_plane_halo(
                u_pad, v_pad, w_pad_half, grid, height_coord,
                config.smagorinsky_cs, halo=h,
                n2_sgs=n2_sgs_h, prandtl=config.smagorinsky_prandtl,
                wall_damping=getattr(config, "smagorinsky_wall_damping", True),
                delta_max=config.smagorinsky_delta_max,
                stability_length=getattr(
                    config, "smagorinsky_stability_length", False),
            )
            sgs_prandtl = config.smagorinsky_prandtl
        K_m_pad, = packed_exchange_halo_plane_yxz(K_m_int, layout=layout)
        K_xface_int = oh.interp_cell_to_xface_vlast_halo(K_m_pad, grid, h)
        K_yface_int = oh.interp_cell_to_yface_vlast_halo(K_m_pad, grid, h)
        K_xface_pad, K_yface_pad = packed_exchange_halo_plane_yxz(
            K_xface_int, K_yface_int, layout=layout,
        )
        du_dt = du_dt + oh.variable_K_diffusion_vlast_halo(
            u_pad, K_xface_pad, grid, h,
        )
        dv_dt = dv_dt + oh.variable_K_diffusion_vlast_halo(
            v_pad, K_yface_pad, grid, h,
        )
        K_h_pad = K_m_pad / sgs_prandtl
        dtheta_p_dt = dtheta_p_dt + oh.variable_K_diffusion_vlast_halo(
            theta_p_pad, K_h_pad, grid, h,
        )
        # K_m at half level for w: column-local mean + rigid zero BC.
        K_m_half_interior = 0.5 * (K_m_pad[..., :-1] + K_m_pad[..., 1:])
        pad_axes_K = ((0, 0),) * (K_m_half_interior.ndim - 1)
        K_m_half_pad = jnp.pad(
            K_m_half_interior, (*pad_axes_K, (1, 1)),
        )
        dw_dt = dw_dt + oh.variable_K_diffusion_vlast_halo(
            w_pad_half, K_m_half_pad, grid, h,
        )

    # 12. Tracer advection.
    tracers = state.tracers.data
    if tracers.shape[-1] > 0:
        # Pad each tracer slot via packed exchange (one MPI round).
        # tracers.shape = (ny_local, nx_local, nlev, n_tracers)
        n_tr = tracers.shape[-1]
        # Treat trailing tracer axis as additional channels by
        # stacking into the trailing dim of packed exchange.
        # tracers already (ny, nx, nlev, n_tr); pack into single
        # exchange directly by reshape into (ny, nx, nlev*n_tr).
        ny_l, nx_l, nlev, _ = tracers.shape
        tracers_flat = tracers.reshape(ny_l, nx_l, nlev * n_tr)
        tracers_flat_pad, = packed_exchange_halo_plane_yxz(
            tracers_flat, layout=layout,
        )
        ny_p, nx_p = tracers_flat_pad.shape[:2]
        tracers_pad = tracers_flat_pad.reshape(ny_p, nx_p, nlev, n_tr)

        def _tracer_tend_one(q_pad, q_int):
            # tadv_* = van_leer under weno5 (positivity guard); == adv_* else.
            return (
                tadv_x(q_pad, u_center_pad, grid.dx, h)
                + tadv_y(q_pad, v_center_pad, grid.dy, h)
                + _vertical_tracer_adv(q_int, w, height_coord, J)
            )
        dtracers_dt = jax.vmap(
            _tracer_tend_one, in_axes=(-1, -1), out_axes=-1,
        )(tracers_pad, tracers)
        # SGS-SCALAR (#80): SAM `sgs.f90:664-675` SGS-diffuses EVERY scalar
        # (q_v + all hydrometeors) with K_h, like θ'. The serial dycore does
        # this; the MPI halo path previously diffused θ'/u/v/w but NOT the
        # tracers ⇒ silent serial≠MPI divergence (caught by
        # test_halo_equiv_with_smagorinsky_and_tracers). Mirror the serial
        # leg here so the two stay bit-identical (K_h_pad is in scope only
        # inside the closure block above — Smagorinsky OR molecular/DNS).
        if _use_smag or _use_mol:
            dtracers_dt = dtracers_dt + jax.vmap(
                lambda q_pad: oh.variable_K_diffusion_vlast_halo(
                    q_pad, K_h_pad, grid, h),
                in_axes=-1, out_axes=-1,
            )(tracers_pad)
    else:
        dtracers_dt = jnp.zeros_like(tracers)
    zero_phis = jnp.zeros_like(state.phis.data)

    return PlaneNonHydrostaticTendencies(
        du_dt=state.u.replace(data=du_dt),
        dv_dt=state.v.replace(data=dv_dt),
        dw_dt=state.w.replace(data=dw_dt),
        dtheta_prime_dt=state.theta_prime.replace(data=dtheta_p_dt),
        drho_prime_dt=state.rho_prime.replace(data=drho_p_dt),
        dphis_dt=state.phis.replace(data=zero_phis),
        dtracers_dt=state.tracers.replace(data=dtracers_dt),
    )


def _re_pad_halo(arr_int, layout, halo):
    """Helper: re-exchange halo for an interior-shape result.

    Used when an operator output is interior-shape but a downstream
    operator needs it padded. Single-rank uses jnp.pad(mode='wrap');
    multi-rank goes through packed_exchange_halo_plane_yxz which
    triggers MPI sendrecv.
    """
    arr_pad, = packed_exchange_halo_plane_yxz(arr_int, layout=layout)
    return arr_pad


# ====================================================================== #
# Two-phase JIT-friendly inner kernels                                   #
#                                                                        #
# The single-call `plane_compressible_euler_slow_tendencies_halo`        #
# above issues 8+ MPI exchanges inside its body — convenient but         #
# forces eager execution under multi-rank (since                         #
# packed_exchange_halo_plane_yxz is not jit-safe on multi-rank).         #
#                                                                        #
# The functions below split the same computation into 2 pure-JAX         #
# kernels that JIT cleanly + can run on the GPU/TPU, plus a              #
# Python-driven `slow_tendency_jit_split` orchestrator that does the     #
# halo exchanges OUTSIDE jit. This is the canonical CRM pattern.         #
# ====================================================================== #


def _slow_tendency_phase1_jit(
    u_pad, v_pad, theta_p_pad, rho_p_pad, pi_p_pad,
    theta_total_pad, rho_total_pad,
    grid, height_coord, terrain_metric, config,
    halo,
):
    """Pure-JAX phase 1: from padded state, compute INTERIOR
    intermediates needing re-exchange + partial tendencies.

    Returns (partial_tend_dict, intermediates_dict).
    intermediates: face fluxes, cell-centred winds, w_full, laplacians.
    partial: PG + Coriolis + sponge + vertical adv + lap-1 contributions.
    """
    h = halo
    c_p = jnp.asarray(0.0, dtype=u_pad.dtype) + _c_pd_constant()
    u_int = u_pad[h:-h, h:-h, :]
    v_int = v_pad[h:-h, h:-h, :]

    # Pressure gradient.
    grad_pi_x = oh.grad_x_vlast_halo(pi_p_pad, grid, h)
    grad_pi_y = oh.grad_y_vlast_halo(pi_p_pad, grid, h)
    theta_xface = oh.interp_cell_to_xface_vlast_halo(
        theta_total_pad, grid, h,
    )
    theta_yface = oh.interp_cell_to_yface_vlast_halo(
        theta_total_pad, grid, h,
    )
    du_pg = -c_p * theta_xface * grad_pi_x
    dv_pg = -c_p * theta_yface * grad_pi_y

    # Mass flux interior (needs re-exchange for divergence).
    rho_xface_int = oh.interp_cell_to_xface_vlast_halo(
        rho_total_pad, grid, h,
    )
    rho_yface_int = oh.interp_cell_to_yface_vlast_halo(
        rho_total_pad, grid, h,
    )
    flux_u_int = rho_xface_int * u_int
    flux_v_int = rho_yface_int * v_int

    # Cell-centred winds (needs re-exchange for upwind advection).
    u_center_int = oh.interp_xface_to_cell_vlast_halo(u_pad, grid, h)
    v_center_int = oh.interp_yface_to_cell_vlast_halo(v_pad, grid, h)

    # Face-staggered winds for momentum advection (needs re-exchange).
    v_at_xface_int = oh.interp_yface_to_xface_vlast_halo(v_pad, grid, h)
    u_at_yface_int = oh.interp_xface_to_yface_vlast_halo(u_pad, grid, h)

    # First-pass laplacians for hyperdiffusion (needs re-exchange).
    lap_u_int = oh.laplacian_vlast_halo(u_pad, grid, h)
    lap_v_int = oh.laplacian_vlast_halo(v_pad, grid, h)
    lap_theta_int = oh.laplacian_vlast_halo(theta_p_pad, grid, h)
    lap_rho_int = oh.laplacian_vlast_halo(rho_p_pad, grid, h)

    partial = {
        "du_pg": du_pg, "dv_pg": dv_pg,
    }
    intermediates = {
        "flux_u_int": flux_u_int, "flux_v_int": flux_v_int,
        "u_center_int": u_center_int, "v_center_int": v_center_int,
        "v_at_xface_int": v_at_xface_int,
        "u_at_yface_int": u_at_yface_int,
        "lap_u_int": lap_u_int, "lap_v_int": lap_v_int,
        "lap_theta_int": lap_theta_int, "lap_rho_int": lap_rho_int,
    }
    return partial, intermediates


def _slow_tendency_phase2_jit(
    state_pad,    # dict of padded state fields
    inter_pad,    # dict of padded intermediates from phase 1
    partial,      # partial tendencies from phase 1
    f_pad,        # padded Coriolis f (or None)
    grid, height_coord, terrain_metric, config,
    halo, original_state,
):
    """Pure-JAX phase 2: combine padded state + padded intermediates
    into final tendencies. JIT-safe (no MPI calls)."""
    h = halo
    cfg = config
    u_pad = state_pad["u_pad"]
    v_pad = state_pad["v_pad"]
    theta_total_pad = state_pad["theta_total_pad"]
    w = state_pad["w"]
    u = state_pad["u"]
    v = state_pad["v"]
    theta_p = state_pad["theta_p"]
    rho_p = state_pad["rho_p"]
    J = terrain_metric.jacobian

    # Mass continuity divergence.
    drho_p_dt = -oh.divergence_vlast_halo(
        inter_pad["flux_u_pad"], inter_pad["flux_v_pad"], grid, h,
    )

    # Theta advection (cell-centred).
    dtheta_p_dt = (
        oh.upwind_advection_x_halo(
            theta_total_pad, inter_pad["u_center_pad"], grid.dx, h,
        )
        + oh.upwind_advection_y_halo(
            theta_total_pad, inter_pad["v_center_pad"], grid.dy, h,
        )
    )

    # Momentum advection (Arakawa-C upwind).
    du_adv = (
        oh.upwind_advection_x_halo(u_pad, u_pad, grid.dx, h)
        + oh.upwind_advection_y_halo(
            u_pad, inter_pad["v_at_xface_pad"], grid.dy, h,
        )
    )
    dv_adv = (
        oh.upwind_advection_x_halo(
            v_pad, inter_pad["u_at_yface_pad"], grid.dx, h,
        )
        + oh.upwind_advection_y_halo(v_pad, v_pad, grid.dy, h)
    )

    # Vertical advection (column-local).
    du_vert = _vertical_advection_plane(u, w, height_coord, J)
    dv_vert = _vertical_advection_plane(v, w, height_coord, J)

    du_dt = du_adv + du_vert + partial["du_pg"]
    dv_dt = dv_adv + dv_vert + partial["dv_pg"]

    # Coriolis.
    if cfg.use_coriolis and f_pad is not None:
        f_xface = oh.interp_cell_to_xface_vlast_halo(f_pad, grid, h)
        f_yface = oh.interp_cell_to_yface_vlast_halo(f_pad, grid, h)
        v_xface_pad = inter_pad["v_at_xface_pad"]
        u_yface_pad = inter_pad["u_at_yface_pad"]
        # Slice to interior for tendency placement.
        v_xface_int = v_xface_pad[h:-h, h:-h, :]
        u_yface_int = u_yface_pad[h:-h, h:-h, :]
        f_xface_int = f_xface  # already interior-shape from interp
        f_yface_int = f_yface
        du_dt = du_dt + f_xface_int * v_xface_int
        dv_dt = dv_dt - f_yface_int * u_yface_int

    # w slow.
    dw_full = (
        oh.upwind_advection_x_halo(
            inter_pad["w_full_pad"], inter_pad["u_center_pad"], grid.dx, h,
        )
        + oh.upwind_advection_y_halo(
            inter_pad["w_full_pad"], inter_pad["v_center_pad"], grid.dy, h,
        )
    )
    pad_axes = ((0, 0),) * (dw_full.ndim - 1)
    dw_dt = jnp.pad(
        0.5 * (dw_full[..., :-1] + dw_full[..., 1:]),
        (*pad_axes, (1, 1)),
    )

    # Sponge.
    sponge_full = sponge_profile(
        height_coord.z_full, height_coord.H,
        cfg.sponge_width, cfg.sponge_coeff,
    )
    sponge_half = sponge_profile(
        height_coord.z_half, height_coord.H,
        cfg.sponge_width, cfg.sponge_coeff,
    )
    du_dt = du_dt - sponge_full * u
    dv_dt = dv_dt - sponge_full * v
    dtheta_p_dt = dtheta_p_dt - sponge_full * theta_p
    drho_p_dt = drho_p_dt - sponge_full * rho_p
    dw_dt = dw_dt - sponge_half * w

    # Hyperdiffusion (2nd laplacian pass on padded inter outputs).
    if cfg.hyperdiff_coeff > 0.0:
        du_dt = du_dt - cfg.hyperdiff_coeff * oh.laplacian_vlast_halo(
            inter_pad["lap_u_pad"], grid, h,
        )
        dv_dt = dv_dt - cfg.hyperdiff_coeff * oh.laplacian_vlast_halo(
            inter_pad["lap_v_pad"], grid, h,
        )
        dtheta_p_dt = dtheta_p_dt - cfg.hyperdiff_coeff * (
            oh.laplacian_vlast_halo(inter_pad["lap_theta_pad"], grid, h)
        )
    if cfg.hyperdiff_rho_coeff > 0.0:
        drho_p_dt = drho_p_dt - cfg.hyperdiff_rho_coeff * (
            oh.laplacian_vlast_halo(inter_pad["lap_rho_pad"], grid, h)
        )

    # Vertical θ Laplacian dissipation (column-local; matches the
    # serial branch in compressible_euler_plane.py:1050-1064 and the
    # main halo entry point — keep both phases in sync).
    nu_v = float(getattr(cfg, "vertical_theta_diffusion", 0.0))
    if nu_v > 0.0 and theta_p.shape[-1] > 2:
        dz_half = height_coord.dz_half
        dz_avg = 0.5 * (dz_half[:-1] + dz_half[1:])
        d2_inner = (
            theta_p[..., 2:] - 2.0 * theta_p[..., 1:-1] + theta_p[..., :-2]
        ) / (dz_avg ** 2)
        pad_axes_v = ((0, 0),) * (theta_p.ndim - 1)
        d2_theta = jnp.pad(d2_inner, (*pad_axes_v, (1, 1)))
        dtheta_p_dt = dtheta_p_dt + nu_v * d2_theta

    zero_phis = jnp.zeros_like(original_state.phis.data)
    dtracers_dt = jnp.zeros_like(original_state.tracers.data)

    return PlaneNonHydrostaticTendencies(
        du_dt=original_state.u.replace(data=du_dt),
        dv_dt=original_state.v.replace(data=dv_dt),
        dw_dt=original_state.w.replace(data=dw_dt),
        dtheta_prime_dt=original_state.theta_prime.replace(data=dtheta_p_dt),
        drho_prime_dt=original_state.rho_prime.replace(data=drho_p_dt),
        dphis_dt=original_state.phis.replace(data=zero_phis),
        dtracers_dt=original_state.tracers.replace(data=dtracers_dt),
    )


def slow_tendency_jit_split(
    state: PlaneNonHydrostaticState,
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    config: CompressibleEulerConfig,
    layout: PlanePencilLayout,
    f_pad_cached: jax.Array | None = None,
) -> PlaneNonHydrostaticTendencies:
    """JIT-friendly halo-aware slow tendency.

    Drives 2 MPI exchanges from PYTHON + 2 jit'd compute kernels:
      1. Exchange initial state padded fields (7 fields, 1 MPI round)
      2. JIT kernel 1: pressure-grad + mass flux + face/cell interps
         + first-pass laplacians (all interior-shape outputs)
      3. Exchange intermediates (8 fields, 1 MPI round)
      4. JIT kernel 2: divergence + advection + Coriolis + sponge +
         hyperdiff (2nd laplacian pass) → final tendencies

    Avoids the per-call mpi4jax sendrecv-inside-jit limitation that
    forces the single-call `plane_compressible_euler_slow_tendencies_halo`
    into eager mode under multi-rank.

    Tracers + Smag NOT yet supported in this fast path; falls back
    to the eager call when needed.

    iter-56 Codex review: the kernel-2 trace below also lacks branches
    for Coriolis (HIGH#1), WENO5 advection (HIGH#2), and
    ``hyperdiff_w_coeff`` (MEDIUM#1). Without an early-fallback those
    features silently no-op on this fast path while running correctly
    on ``step_halo``. Extended the fallback set so any production-
    style config falls through to the eager path, leaving this fast
    path correct in its narrower regime (no-Smag, no-tracer,
    no-Coriolis, upwind1, no-hyperdiff_w bench-style runs).
    Production driver always trips at least one of these gates
    (Smag c_s=0.2 + 3 tracers) so behaviour is unchanged.
    """
    _w_hyperdiff = getattr(config, "hyperdiff_w_coeff", 0.0)
    _advection = getattr(config, "horizontal_advection_scheme", "upwind1")
    _closure = getattr(config, "turbulence_closure", "smagorinsky")
    # Single-rank-only closures ("vreman", "amd") have NO MPI-halo K_m kernel.
    # Fail at ENTRY — before any dispatch — so a multi-rank config raises
    # immediately instead of first doing an MPI halo exchange inside the eager
    # fallback (whose own guard sits after several exchanges).  amd_c /
    # smagorinsky_cs both default 0.0, so without this by-NAME check an 'amd'
    # config would trip none of the fall-back conditions below and run INVISCID.
    _require_single_rank_closure(_closure)
    # The remaining non-default closure supported by the eager kernel is
    # "molecular" (constant-nu DNS): fall back so its SGS block is not skipped.
    if (
        config.smagorinsky_cs > 0.0
        or _closure == "molecular"
        or state.tracers.data.shape[-1] > 0
        or config.use_coriolis
        or _advection != "upwind1"
        or _w_hyperdiff > 0.0
    ):
        return plane_compressible_euler_slow_tendencies_halo(
            state, grid, height_coord, terrain_metric, config, layout,
            f_pad_cached=f_pad_cached,
        )

    h = layout.halo if hasattr(layout, "halo") else 1
    u = state.u.data; v = state.v.data; w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref

    # Column-local pre-compute (jit-safe).
    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p, rho_0 + rho_p,
    )
    pi_p = compute_exner_perturbation(rho_p, theta_p, height_coord)
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])

    # === MPI exchange 1: padded state ===
    (
        u_pad, v_pad, theta_p_pad, rho_p_pad, pi_p_pad,
        theta_total_pad, rho_total_pad,
    ) = packed_exchange_halo_plane_yxz(
        u, v, theta_p, rho_p, pi_p, theta_total, rho_total,
        layout=layout,
    )

    # === JIT kernel 1: intermediates ===
    phase1, phase2 = _build_phase_kernels(
        grid, height_coord, terrain_metric, config, h,
    )
    partial, inter_int = phase1(
        u_pad, v_pad, theta_p_pad, rho_p_pad, pi_p_pad,
        theta_total_pad, rho_total_pad,
    )

    # === MPI exchange 2: intermediates + w_full ===
    (
        flux_u_pad, flux_v_pad, u_center_pad, v_center_pad,
        v_at_xface_pad, u_at_yface_pad, w_full_pad,
        lap_u_pad, lap_v_pad, lap_theta_pad, lap_rho_pad,
    ) = packed_exchange_halo_plane_yxz(
        inter_int["flux_u_int"], inter_int["flux_v_int"],
        inter_int["u_center_int"], inter_int["v_center_int"],
        inter_int["v_at_xface_int"], inter_int["u_at_yface_int"],
        w_full,
        inter_int["lap_u_int"], inter_int["lap_v_int"],
        inter_int["lap_theta_int"], inter_int["lap_rho_int"],
        layout=layout,
    )

    # === JIT kernel 2: final tendencies ===
    state_pad = {
        "u_pad": u_pad, "v_pad": v_pad,
        "theta_p_pad": theta_p_pad, "theta_total_pad": theta_total_pad,
        "rho_p_pad": rho_p_pad, "w": w, "u": u, "v": v,
        "theta_p": theta_p, "rho_p": rho_p,
    }
    inter_pad = {
        "flux_u_pad": flux_u_pad, "flux_v_pad": flux_v_pad,
        "u_center_pad": u_center_pad, "v_center_pad": v_center_pad,
        "v_at_xface_pad": v_at_xface_pad,
        "u_at_yface_pad": u_at_yface_pad,
        "w_full_pad": w_full_pad,
        "lap_u_pad": lap_u_pad, "lap_v_pad": lap_v_pad,
        "lap_theta_pad": lap_theta_pad, "lap_rho_pad": lap_rho_pad,
    }
    tend = phase2(state_pad, inter_pad, partial, f_pad_cached, state)

    # SAM moist buoyancy on w (post-phase2). The sponge / hyperdiff / Smag
    # legs phase2 added to dw_dt are functions of the STATE w (not dw_dt),
    # so this extra additive forcing is bit-identical to adding it inside
    # the monolithic kernel before those legs. Kept here (not in the jitted
    # phase2) because the GLOBAL horizontal mean needs ``layout``, which the
    # phase2 signature does not carry. Skipped when moisture is off.
    if config.moist_buoyancy and not getattr(
            config, "acoustic_moist_buoyancy", True):  # else added in acoustic loop
        b_half = moisture_buoyancy_w_half(
            state.tracers.data, state.theta_prime.data, height_coord,
            lambda f: _global_hmean_plane(f, layout),
        )
        tend = tend._replace(
            dw_dt=tend.dw_dt.replace(data=tend.dw_dt.data + b_half),
        )
    return tend


# Per-(grid id, hc id, tm id, cfg, h) closure cache for jit'd kernels.
# PlaneGrid + HeightCoordinate + TerrainMetric are NamedTuples
# containing JAX arrays (unhashable) → cannot be static_argnums
# directly. Build closure that captures them; jit by function
# identity. Subsequent calls with the SAME grid/hc/tm/cfg reuse
# the compiled kernel.
_PHASE_KERNEL_CACHE = {}


def _build_phase_kernels(grid, height_coord, terrain_metric, config, halo):
    """Return (phase1_jit, phase2_jit) closures keyed in the cache."""
    key = (id(grid), id(height_coord), id(terrain_metric), id(config), halo)
    if key in _PHASE_KERNEL_CACHE:
        return _PHASE_KERNEL_CACHE[key]

    @jax.jit
    def phase1(u_pad, v_pad, theta_p_pad, rho_p_pad, pi_p_pad,
               theta_total_pad, rho_total_pad):
        return _slow_tendency_phase1_jit(
            u_pad, v_pad, theta_p_pad, rho_p_pad, pi_p_pad,
            theta_total_pad, rho_total_pad,
            grid, height_coord, terrain_metric, config, halo,
        )

    @jax.jit
    def phase2(state_pad, inter_pad, partial, f_pad, original_state):
        return _slow_tendency_phase2_jit(
            state_pad, inter_pad, partial, f_pad,
            grid, height_coord, terrain_metric, config, halo,
            original_state,
        )

    _PHASE_KERNEL_CACHE[key] = (phase1, phase2)
    return phase1, phase2


