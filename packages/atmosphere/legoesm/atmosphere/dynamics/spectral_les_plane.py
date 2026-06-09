"""Pseudo-spectral INCOMPRESSIBLE boundary-layer LES on the doubly-periodic plane.

Faithful in algorithm to the **jax-alfa** LES oracle (Basu; the validation
reference for legoESM's ABL LES): horizontal derivatives are exact in Fourier
space (``rfft2``), the vertical is second-order finite difference on a grid
STAGGERED in ``w``, advection is the rotational (vorticity) form ``C = ω × u``
(de-aliased by the 3/2 rule), and incompressibility is enforced each step by a
fractional-step pressure PROJECTION (spectral-horizontal + tridiagonal-vertical
Poisson). Time stepping is Adams–Bashforth-2.

Why this core (vs the compressible plane dycore)
-----------------------------------------------
The compressible plane dycore needs acoustic off-centring + biharmonic hyperdiff
for stability; that numerical dissipation caps the effective Reynolds number
below the turbulence-sustaining threshold, so resolved ABL turbulence laminarises
(``les_plane_turbulence_notes.md``). A pseudo-spectral incompressible solver
has NO acoustic mode and NO numerical hyperdiffusion — the ONLY dissipation is
the physical SGS model — so the effective Re is high and turbulence sustains,
exactly as in the oracle. This module is the faithful path to a quantitative
oracle match; it hosts the same wall-damped Smagorinsky family as the oracle
(constant-coefficient here; the Bou-Zeid scale-dependent dynamic coefficient
``_compute_scale_dependent_dynamic_smag_cs_plane`` is wired in a follow-up).

Grid / staggering
-----------------
Physical layout ``(ny, nx, nz)`` (``y`` axis 0, ``x`` axis 1), matching the rest
of the plane code. ``u, v`` live at cell CENTRES ``z_c = (k+½)Δz`` (k=0..nz-1);
``w`` lives at FACES ``z_f = kΔz`` (k=0..nz) with the rigid-lid / ground BC
``w[0] = w[nz] = 0``. Uniform ``Δz`` (oracle convention). Horizontal is periodic.

Status: NEUTRAL ABL (no buoyancy) — the cleanest oracle-comparison case. θ /
buoyancy for SBL/CBL is a follow-up; the projection, advection and wall model are
buoyancy-independent.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.lasd_core import lasd_cs2
from legoesm.atmosphere.physics.turbulence.vreman import vreman_nu_t as _vreman_core
from legoesm.timestepping.split_explicit import (
    SplitExplicitConfig, split_explicit_step)


# --------------------------------------------------------------------------- #
# Config + grid + state                                                        #
# --------------------------------------------------------------------------- #
class SpectralLESConfig(NamedTuple):
    nx: int
    ny: int
    nz: int
    Lx: float
    Ly: float
    Lz: float
    z0: float = 0.1                 # aerodynamic roughness [m]
    c_s: float = 0.16              # Smagorinsky coefficient (static; wall-damped)
    wall_damping: bool = True      # cap l_m at κz near the surface (Mason 1989)
    dealias: bool = True           # 3/2-rule zero-padding de-aliasing
    nu_molecular: float = 0.0      # optional explicit viscosity (usually 0)
    smagorinsky_dynamic: bool = False  # Bou-Zeid LASD scale-dependent dynamic C_s(x,y,z)
    cs_max: float = 1.0            # upper clip on the dynamic C_s² (oracle mask)
    buoyancy: bool = False         # Boussinesq buoyancy in w (θ scalar required)
    theta_ref0: float = 290.0      # reference θ for the buoyancy term [K]
    pr_sgs: float = 1.0            # turbulent Prandtl number (K_h = ν_t / Pr)
    nu_floor: float = 0.0          # background eddy-viscosity floor [m²/s] — keeps
    #                                strongly-stable layers (where the dynamic SGS
    #                                shuts off) from going fully inviscid and
    #                                growing the 2Δ gravity-wave/KH mode (SBL).
    spectral_filter: bool = True   # SHARP high-wavenumber cutoff applied to the
    #                                prognostic fields each step. In quiescent layers
    #                                above the turbulent BL the Smagorinsky ν_t→0, so
    #                                the forward energy cascade piles up at the 2Δx
    #                                grid scale with nothing to dissipate it (spectral
    #                                "thermalisation" → grid-scale w noise that grows
    #                                with height). A SHARP cutoff (σ=1 below k_c, 0
    #                                above) is used rather than a smooth exp filter:
    #                                applied EVERY step, any σ<1 compounds over the
    #                                ~10^4-10^5 steps of a run and would erode the
    #                                resolved scales too; a sharp cutoff leaves the
    #                                kept modes EXACTLY unchanged (σ=1, no compounding)
    #                                and only discards the top noise band. Being a
    #                                horizontal multiplier uniform in z, applying it
    #                                equally to u, v, w preserves the discrete
    #                                divergence-free condition (filtered div = σ·div
    #                                = 0) — no re-projection needed.
    filter_cutoff_frac: float = 0.67  # keep radial |k|/k_Nyquist ≤ frac (2/3 rule);
    #                                   resolved turbulence lives well below this, so
    #                                   the cutoff removes the grid-scale noise band
    #                                   with no measurable effect on u_*/σ_w.
    sgs_model: str = "smagorinsky"  # static-SGS closure: "smagorinsky" (Mason-capped
    #                                 |S|-Smagorinsky) or "vreman" (Vreman 2004). Only
    #                                 used when smagorinsky_dynamic=False. Vreman ν_t
    #                                 VANISHES for well-resolved laminar/2D shear and
    #                                 activates only on genuine 3D (under-resolved)
    #                                 structure → less spurious dissipation in the
    #                                 surface layer & quiescent air aloft, and it uses
    #                                 the per-direction filter widths so it behaves on
    #                                 ANISOTROPIC dx≠dz grids where Smagorinsky/LASD
    #                                 over- or under-dissipate.
    c_vreman: float = 0.07          # Vreman model constant (≈ 2.5·C_s²; Vreman 2004)
    time_scheme: str = "rk3"        # "rk3" (SSP-RK3, Shu-Osher; projection each
    #                                 stage) or "ab2" (Adams–Bashforth-2). RK3 is
    #                                 self-starting, 3rd-order, and stable to a ~2-3×
    #                                 larger advective CFL than AB2 → supports the
    #                                 larger time steps a CFL-adaptive controller picks.


class SpectralLESLayout(NamedTuple):
    """y-slab MPI decomposition for the distributed horizontal FFT.

    Rank owns ``(ny_local, nx, nz)`` physically; in spectral space it owns the
    full ``ny`` (ky) and a kx-column slab of width ``nkx_local`` (see
    ``parallel/distributed_fft.py``). ``n_ranks_x`` is implicitly 1 (slab).
    ``comm`` is the MPI communicator (non-array; kept out of any AD path).
    """
    rank: int
    n_ranks: int
    ny_global: int
    nx: int
    comm: Any = None


class SpectralLESGrid(NamedTuple):
    cfg: SpectralLESConfig
    dx: float
    dy: float
    dz: float
    z_c: jax.Array                 # (nz,) centre heights
    z_f: jax.Array                 # (nz+1,) face heights
    kx: jax.Array                  # (ny, nkx) rad/m, x-wavenumber (kx-local if MPI)
    ky: jax.Array                  # (ny, nkx) rad/m, y-wavenumber
    k2: jax.Array                  # kx²+ky²
    dealias_mask: jax.Array        # (ny, nkx) 2/3 truncation mask
    filter_mask: jax.Array         # (ny, nkx) smooth high-k low-pass σ(k)
    layout: SpectralLESLayout | None = None  # None ⇒ serial (global rfft2)


class SpectralLESState(NamedTuple):
    u: jax.Array                   # (ny, nx, nz)   centres
    v: jax.Array                   # (ny, nx, nz)   centres
    w: jax.Array                   # (ny, nx, nz+1) faces, w[..,0]=w[..,nz]=0
    rhs_u_prev: jax.Array          # AB2 previous tendencies
    rhs_v_prev: jax.Array
    rhs_w_prev: jax.Array
    theta: jax.Array | None = None      # (ny,nx,nz) potential temperature [K]
    rhs_theta_prev: jax.Array | None = None


def make_grid(cfg: SpectralLESConfig, dtype=jnp.float64,
              layout: SpectralLESLayout | None = None) -> SpectralLESGrid:
    # SGS constants must be non-negative or ν_t can go negative (anti-diffusion,
    # blow-up). The shared vreman/Smagorinsky cores trust these — validate here,
    # the single point where a config becomes a runnable grid.
    if cfg.c_s < 0.0 or cfg.c_vreman < 0.0 or cfg.nu_floor < 0.0:
        raise ValueError(
            f"SGS constants must be >= 0: c_s={cfg.c_s}, c_vreman={cfg.c_vreman}, "
            f"nu_floor={cfg.nu_floor} (negative ν_t is anti-diffusive).")
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    # The rfft Nyquist-zeroing and the 3/2-rule de-aliasing (drop the single
    # Nyquist row/column) assume EVEN nx, ny. Odd sizes would silently use a
    # different, wrong truncation. Validate here (codex 2026-06-09).
    if nx % 2 or ny % 2:
        raise ValueError(
            f"spectral LES needs EVEN nx, ny (rfft Nyquist + 3/2-rule de-aliasing "
            f"assume it); got nx={nx}, ny={ny}.")
    dx, dy, dz = cfg.Lx / nx, cfg.Ly / ny, cfg.Lz / nz
    z_c = (jnp.arange(nz, dtype=dtype) + 0.5) * dz
    z_f = jnp.arange(nz + 1, dtype=dtype) * dz
    # rfft2 over (y, x): axis-0 (y) full length ny, axis-1 (x) reduced to nx//2+1.
    kx_1d = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=dx).astype(dtype)      # (nx//2+1,)
    ky_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dy).astype(dtype)       # (ny,)
    # Zero the Nyquist wavenumbers: the spectral derivative of the Nyquist mode
    # is not representable on the grid (it would alias), so it must be excluded
    # from BOTH the gradient/divergence AND the pressure Laplacian k² to keep the
    # projection EXACTLY consistent (div∘grad = Laplacian). Without this the
    # corrected field retains an O(10%) Nyquist divergence.
    if nx % 2 == 0:
        kx_1d = kx_1d.at[-1].set(0.0)
    if ny % 2 == 0:
        ky_1d = ky_1d.at[ny // 2].set(0.0)
    kx = jnp.broadcast_to(kx_1d[None, :], (ny, kx_1d.size))
    ky = jnp.broadcast_to(ky_1d[:, None], (ny, kx_1d.size))
    k2 = kx ** 2 + ky ** 2
    # 2/3-rule de-alias mask (truncate |k| above 2/3 Nyquist on each axis).
    cut_x = nx // 3
    cut_y = ny // 3
    iy = jnp.arange(ny)
    fold_y = jnp.minimum(iy, ny - iy)
    mask = ((fold_y[:, None] <= cut_y) & (jnp.arange(kx_1d.size)[None, :] <= cut_x)
            ).astype(dtype)
    # Sharp high-wavenumber cutoff: keep |kx|,|ky| ≤ frac·k_Nyquist on each axis,
    # zero above. Built from the TRUE wavenumber indices (INCLUDING the Nyquist
    # column, where the derivative kx/ky were zeroed) so the grid-scale noise that
    # lives at/near Nyquist is actually removed. σ∈{0,1} ⇒ kept modes unchanged
    # (no per-step compounding) and divergence-free preserving (z-uniform).
    if cfg.spectral_filter:
        # RADIAL cutoff in normalised wavenumber: keep modes whose isotropic
        # |k|/k_Nyquist ≤ frac (removes the diagonal near-Nyquist modes a per-axis
        # box would retain — those carry the grid-scale noise in the quiescent
        # layers). Normalise each axis by its own Nyquist index so dx≠dz/Lx≠Ly
        # anisotropy is handled, then take the radial magnitude.
        ix = jnp.arange(kx_1d.size)                        # 0..nx//2 (rfft x-axis)
        kxn = (ix[None, :] / (nx // 2))                    # |kx|/kx_Nyq ∈ [0,1]
        kyn = (jnp.minimum(iy, ny - iy)[:, None] / (ny // 2))
        rn = jnp.sqrt(kxn ** 2 + kyn ** 2)                 # radial, corner = √2
        fmask = (rn <= cfg.filter_cutoff_frac).astype(dtype)
    else:
        fmask = jnp.ones_like(k2)
    if layout is not None:
        # Distributed FFT: each rank holds a kx-column slab (full ky). Pad the
        # reduced-kx axis to P·ceil(nkx/P) and slice this rank's columns so every
        # pointwise spectral multiply (kx, ky, k2, masks) lines up with the
        # (ny, nkx_local, nz) output of distributed_rfft2. Padding columns carry
        # no energy (zeros), so the sliced wavenumbers there are harmless.
        from legoesm.parallel.distributed_fft import (
            kx_local_size, local_kx_slice)
        nkx = nx // 2 + 1
        nkx_pad = layout.n_ranks * kx_local_size(nx, layout.n_ranks)
        lo, hi = local_kx_slice(nx, layout.n_ranks, layout.rank)

        def _slice(arr):
            return jnp.pad(arr, ((0, 0), (0, nkx_pad - nkx)))[:, lo:hi]

        kx, ky, k2 = _slice(kx), _slice(ky), _slice(k2)
        mask, fmask = _slice(mask), _slice(fmask)
    return SpectralLESGrid(cfg=cfg, dx=dx, dy=dy, dz=dz, z_c=z_c, z_f=z_f,
                           kx=kx, ky=ky, k2=k2, dealias_mask=mask,
                           filter_mask=fmask, layout=layout)


# --------------------------------------------------------------------------- #
# Horizontal spectral derivatives (exact)                                      #
# --------------------------------------------------------------------------- #
def _fft(f, g: SpectralLESGrid):
    """Forward horizontal rfft2 over (y, x). Serial ``jnp.fft.rfft2`` when
    ``g.layout is None``; otherwise the y-slab distributed FFT (output is
    ``(ny_global, nkx_local, nz)``)."""
    if g.layout is None:
        return jnp.fft.rfft2(f, axes=(0, 1))
    from legoesm.parallel.distributed_fft import distributed_rfft2
    L = g.layout
    return distributed_rfft2(f, ny_global=L.ny_global, nx=L.nx,
                             n_ranks=L.n_ranks, comm=L.comm)


def _ifft(fh, g: SpectralLESGrid):
    """Inverse of :func:`_fft` (returns the physical y-slab ``(ny_local,nx,nz)``
    under MPI, the global ``(ny,nx,nz)`` serially)."""
    if g.layout is None:
        return jnp.fft.irfft2(fh, axes=(0, 1), s=(g.cfg.ny, g.cfg.nx))
    from legoesm.parallel.distributed_fft import distributed_irfft2
    L = g.layout
    return distributed_irfft2(fh, ny_global=L.ny_global, nx=L.nx,
                              n_ranks=L.n_ranks, comm=L.comm)


def _planar_mean(f, g: SpectralLESGrid, keepdims=False):
    """Horizontal (y, x) mean. Serial: ``jnp.mean`` over axes (0,1). Under the
    y-slab MPI layout each rank holds only a y-slab, so the true planar mean is a
    global reduction: ``global_sum_mpi(Σ_slab) / (ny_global·nx)`` (AD-safe SUM
    collective — keeps the wall model / buoyancy differentiable)."""
    if g.layout is None:
        return jnp.mean(f, axis=(0, 1), keepdims=keepdims)
    from legoesm.parallel.reductions import global_sum_mpi
    local_sum = jnp.sum(f, axis=(0, 1), keepdims=keepdims)
    # Reduce over the LAYOUT's communicator (same one the distributed FFT uses) —
    # not COMM_WORLD — so sub-communicator runs stay consistent (codex 2026-06-09).
    return (global_sum_mpi(local_sum, comm=g.layout.comm)
            / (g.layout.ny_global * g.layout.nx))


def ddx(f, g: SpectralLESGrid):
    """∂/∂x via spectral (exact); Nyquist x-mode killed for a real derivative."""
    fh = _fft(f, g) * (1j * g.kx[..., None])
    return _ifft(fh, g)


def ddy(f, g: SpectralLESGrid):
    fh = _fft(f, g) * (1j * g.ky[..., None])
    return _ifft(fh, g)


def _apply_filter(f, g: SpectralLESGrid):
    """Smooth horizontal high-wavenumber low-pass (per z-level). Multiplies the
    rfft2 spectrum by the precomputed σ(k) mask; uniform in z ⇒ divergence-free
    preserving when applied equally to u, v, w."""
    return _ifft(_fft(f, g) * g.filter_mask[..., None], g)


def _dealias(f, g: SpectralLESGrid):
    if not g.cfg.dealias:
        return f
    return _ifft(_fft(f, g) * g.dealias_mask[..., None], g)


# --------------------------------------------------------------------------- #
# 3/2-rule (zero-padding) de-aliasing — verbatim port of jax-alfa Dealias1/2.  #
# Pad each factor to the 3/2 grid, multiply alias-free in physical space, then  #
# truncate the product back. Removes the quadratic-interaction aliasing that    #
# 2/3-truncation leaves in the chained rotational advection. Layout (ny,nx):    #
# the rfft2 FULL axis is y (axis 0), the REDUCED axis is x (axis 1). Both       #
# Nyquist modes are dropped (unrepresentable derivative; the oracle does too).  #
# --------------------------------------------------------------------------- #
def _pad_to_fine(f_yxz, g: SpectralLESGrid | None = None):
    """Coarse ``(ny,nx,nz)`` → fine ``(3ny/2, 3nx/2, nz)`` physical (zero-pad).

    Under the y-slab ``g.layout`` the pad is separable: x is done locally (x
    undecomposed) and y via one all-to-all transpose — see
    ``parallel/distributed_fft.distributed_pad_to_fine``."""
    if g is not None and g.layout is not None:
        from legoesm.parallel.distributed_fft import distributed_pad_to_fine
        L = g.layout
        return distributed_pad_to_fine(f_yxz, ny_global=L.ny_global, nx=L.nx,
                                       n_ranks=L.n_ranks, comm=L.comm)
    ny, nx, nz = f_yxz.shape
    nyf, nxf = 3 * ny // 2, 3 * nx // 2
    nyh, nxr = ny // 2, nx // 2                 # half full-axis; drop x-Nyquist
    fh = jnp.fft.rfft2(f_yxz, axes=(0, 1))       # (ny, nx//2+1, nz)
    pad = jnp.zeros((nyf, nxf // 2 + 1, nz), dtype=fh.dtype)
    pad = pad.at[:nyh, :nxr, :].set(fh[:nyh, :nxr, :])              # +ky, +kx
    pad = pad.at[nyf - nyh + 1:, :nxr, :].set(fh[nyh + 1:ny, :nxr, :])  # −ky
    return jnp.fft.irfft2(pad, axes=(0, 1), s=(nyf, nxf))


def _truncate_from_fine(f_fine, ny, nx, g: SpectralLESGrid | None = None):
    """Fine ``(3ny/2,3nx/2,nz)`` → coarse ``(ny,nx,nz)`` physical (+9/4 scaling)."""
    if g is not None and g.layout is not None:
        from legoesm.parallel.distributed_fft import distributed_truncate_from_fine
        L = g.layout
        return distributed_truncate_from_fine(f_fine, ny_global=L.ny_global,
                                              nx=L.nx, n_ranks=L.n_ranks,
                                              comm=L.comm)
    nyf, _nxf = f_fine.shape[0], f_fine.shape[1]
    nz = f_fine.shape[2]
    nyh, nxr = ny // 2, nx // 2
    fh = jnp.fft.rfft2(f_fine, axes=(0, 1))      # (nyf, nxf//2+1, nz)
    out = jnp.zeros((ny, nx // 2 + 1, nz), dtype=fh.dtype)
    out = out.at[:nyh, :nxr, :].set(fh[:nyh, :nxr, :])
    out = out.at[ny - nyh + 1:, :nxr, :].set(fh[nyf - nyh + 1:, :nxr, :])
    return (9.0 / 4.0) * jnp.fft.irfft2(out, axes=(0, 1), s=(ny, nx))


# --------------------------------------------------------------------------- #
# Vertical staggering helpers (uniform Δz)                                     #
# --------------------------------------------------------------------------- #
def c2f(fc):
    """Centre (nz) → interior face (nz-1) average."""
    return 0.5 * (fc[..., :-1] + fc[..., 1:])


def f2c(ff):
    """Face (nz+1) → centre (nz) average."""
    return 0.5 * (ff[..., :-1] + ff[..., 1:])


def ddz_c2f(fc, dz):
    """∂/∂z of a CENTRE field at interior FACES (nz-1) = (f[k]-f[k-1])/Δz."""
    return (fc[..., 1:] - fc[..., :-1]) / dz


def ddz_f2c(ff, dz):
    """∂/∂z of a FACE field at CENTRES (nz) = (f[k+1]-f[k])/Δz."""
    return (ff[..., 1:] - ff[..., :-1]) / dz


# --------------------------------------------------------------------------- #
# Strain rate + Smagorinsky eddy viscosity                                     #
# --------------------------------------------------------------------------- #
def _velocity_gradients(u, v, w, g: SpectralLESGrid):
    """All nine resolved velocity gradients ``a_cd = ∂u_c/∂x_d`` at CENTRES
    (c=component u,v,w; d=direction x,y,z). Shared by :func:`_strain` and
    :func:`_vreman_nu_t`."""
    dz = g.dz
    dudx, dvdx, dwdx = ddx(u, g), ddx(v, g), ddx(w, g)   # dwdx at faces
    dudy, dvdy, dwdy = ddy(u, g), ddy(v, g), ddy(w, g)
    dwdz_c = ddz_f2c(w, dz)                               # ∂w/∂z at centres
    # ∂u/∂z, ∂v/∂z at faces → centre.
    dudz_f = jnp.pad(ddz_c2f(u, dz), ((0, 0), (0, 0), (1, 1)), mode="edge")
    dvdz_f = jnp.pad(ddz_c2f(v, dz), ((0, 0), (0, 0), (1, 1)), mode="edge")
    return dict(
        a11=dudx, a12=dudy, a13=f2c(dudz_f),
        a21=dvdx, a22=dvdy, a23=f2c(dvdz_f),
        a31=f2c(dwdx), a32=f2c(dwdy), a33=dwdz_c)


def _strain(u, v, w, g: SpectralLESGrid):
    """Full resolved strain ``S_ij`` and ``|S|=√(2 S_ij S_ij)`` at CENTRES."""
    a = _velocity_gradients(u, v, w, g)
    dudx, dudy, dudz_c = a["a11"], a["a12"], a["a13"]
    dvdx, dvdy, dvdz_c = a["a21"], a["a22"], a["a23"]
    dwdx_c, dwdy_c, dwdz_c = a["a31"], a["a32"], a["a33"]
    S11, S22, S33 = dudx, dvdy, dwdz_c
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz_c + dwdx_c)
    S23 = 0.5 * (dvdz_c + dwdy_c)
    Smag = jnp.sqrt(jnp.maximum(
        2.0 * (S11 ** 2 + S22 ** 2 + S33 ** 2
               + 2.0 * (S12 ** 2 + S13 ** 2 + S23 ** 2)), 1e-30))
    return (S11, S22, S33, S12, S13, S23), Smag


def _vreman_nu_t(u, v, w, g: SpectralLESGrid):
    """Vreman (2004) eddy viscosity at CENTRES — computes the nine resolved
    gradients on the spectral grid and defers the algebra to the shared
    :func:`legoesm.atmosphere.physics.turbulence.vreman.vreman_nu_t` (the SAME
    implementation the compressible CRM uses)."""
    a = _velocity_gradients(u, v, w, g)
    return _vreman_core(
        a["a11"], a["a12"], a["a13"], a["a21"], a["a22"], a["a23"],
        a["a31"], a["a32"], a["a33"],
        g.dx, g.dy, g.dz, g.cfg.c_vreman, nu_floor=g.cfg.nu_floor)


def eddy_viscosity(u, v, w, g: SpectralLESGrid):
    """Wall-damped constant-coefficient Smagorinsky ``ν_t=(C_s l)²|S|`` at centres.

    ``l = min(C_s Δ, κ z)`` (Mason 1989) with ``Δ=(Δx Δy Δz)^⅓``. This is the
    oracle's static-SGS option; the Bou-Zeid scale-dependent dynamic coefficient
    (``smagorinsky_dynamic``) replaces the constant ``C_s`` + κz cap with the
    per-(x,y,z) ``C_s²`` from :func:`lasd_core.lasd_cs2` (β IS the near-wall
    scale correction, so no Mason cap)."""
    S_tuple, Smag = _strain(u, v, w, g)
    delta = (g.dx * g.dy * g.dz) ** (1.0 / 3.0)
    if g.cfg.smagorinsky_dynamic:
        nz = u.shape[-1]
        wc = f2c(w)
        cs2 = lasd_cs2(u, v, wc, *S_tuple, Smag,
                       jnp.full(nz, delta, dtype=u.dtype), cs_max=g.cfg.cs_max,
                       layout=g.layout)
        return cs2 * (delta ** 2) * Smag + g.cfg.nu_floor   # ν_t = C_s²·Δ²·|S|
    if g.cfg.sgs_model == "vreman":
        return _vreman_nu_t(u, v, w, g)
    if g.cfg.sgs_model != "smagorinsky":
        raise ValueError(
            f"unknown sgs_model {g.cfg.sgs_model!r}; "
            "expected 'smagorinsky' or 'vreman'")
    l_smag = g.cfg.c_s * delta
    if g.cfg.wall_damping:
        kappa = constants.kappa_von_karman
        l_m = jnp.minimum(l_smag, kappa * g.z_c)            # (nz,)
    else:
        l_m = l_smag
    return (l_m ** 2) * Smag + g.cfg.nu_floor               # (ny,nx,nz)


# --------------------------------------------------------------------------- #
# Rotational-form advection (de-aliased)                                        #
# --------------------------------------------------------------------------- #
def advection(u, v, w, g: SpectralLESGrid):
    """Rotational (vorticity) form ``C = ω × u`` — the energy-conserving form the
    oracle uses (the Bernoulli ½|u|² term is absorbed by the pressure). Returns
    ``(Cu, Cv, Cw)`` as MINUS the advective tendency (i.e. the RHS contribution
    ``-C``). ``Cu, Cv`` at centres, ``Cw`` at interior faces."""
    dz = g.dz
    ny, nx = u.shape[0], u.shape[1]
    # Vorticity from spectral/staggered derivatives on the COARSE grid.
    dudy, dvdx = ddy(u, g), ddx(v, g)
    omega_z = dvdx - dudy                                   # centres
    dudz_f = jnp.pad(ddz_c2f(u, dz), ((0, 0), (0, 0), (1, 1)), mode="edge")
    dvdz_f = jnp.pad(ddz_c2f(v, dz), ((0, 0), (0, 0), (1, 1)), mode="edge")
    dwdx_f, dwdy_f = ddx(w, g), ddy(w, g)                   # faces
    omega_x_f = dvdz_f - dwdy_f                             # faces
    omega_y_f = dwdx_f - dudz_f                             # faces

    # RHS advection = +(u × ω): (u×ω)_x = v ω_z − w ω_y; (u×ω)_y = w ω_x − u ω_z;
    # (u×ω)_z = u ω_y − v ω_x. The Bernoulli ½|u|² is absorbed by the pressure.
    # Each quadratic product is formed ALIAS-FREE by 3/2 zero-padding (verbatim
    # jax-alfa Dealias1/2): pad both factors to the fine grid, multiply (with the
    # vertical f2c/c2f staggering done on the fine grid), truncate back. This
    # replaces the 2/3-truncation, which leaves quadratic-interaction aliasing in
    # the chained rotational form. ``w·ω`` products are formed at the FACES then
    # averaged (StagGridAvg-on-the-product).
    if g.cfg.dealias:
        pf = lambda x: _pad_to_fine(x, g)                  # noqa: E731
        tf = lambda x: _truncate_from_fine(x, ny, nx, g)   # noqa: E731
        u_F, v_F, w_F = pf(u), pf(v), pf(w)
        omz_F, omx_F, omy_F = pf(omega_z), pf(omega_x_f), pf(omega_y_f)
        Cu = tf(omz_F * v_F + f2c(w_F * omy_F))             # centres
        Cv = tf(-omz_F * u_F - f2c(w_F * omx_F))
        uf_F = jnp.pad(c2f(u_F), ((0, 0), (0, 0), (1, 1)))  # faces, 0 at walls
        vf_F = jnp.pad(c2f(v_F), ((0, 0), (0, 0), (1, 1)))
        Cw = tf(vf_F * omx_F - uf_F * omy_F)               # faces (nz+1)
    else:
        Cu = omega_z * v + f2c(w * omega_y_f)
        Cv = -omega_z * u - f2c(w * omega_x_f)
        uf = jnp.pad(c2f(u), ((0, 0), (0, 0), (1, 1)))
        vf = jnp.pad(c2f(v), ((0, 0), (0, 0), (1, 1)))
        Cw = vf * omega_x_f - uf * omega_y_f
    return Cu, Cv, Cw


# --------------------------------------------------------------------------- #
# SGS stress divergence + MOST wall model                                      #
# --------------------------------------------------------------------------- #
def sgs_and_wall(u, v, w, nu_t, g: SpectralLESGrid, u_geo):
    """SGS force ``∂_j(2 ν_t S_ij)`` with the surface vertical momentum flux
    replaced by the Monin–Obukhov (neutral) wall stress.

    Returns ``(Fu, Fv, Fw)`` (centres, centres, faces) and the surface friction
    velocity ``u_*`` (diagnostic)."""
    dz = g.dz
    kappa = constants.kappa_von_karman
    (S11, S22, S33, S12, S13, S23), _ = _strain(u, v, w, g)
    tau11, tau22, tau33 = 2 * nu_t * S11, 2 * nu_t * S22, 2 * nu_t * S33
    tau12 = 2 * nu_t * S12
    _tau13, _tau23 = 2 * nu_t * S13, 2 * nu_t * S23           # centres
    # Horizontal divergence of the stress (spectral, exact).
    Fu = ddx(tau11, g) + ddy(tau12, g)
    Fv = ddx(tau12, g) + ddy(tau22, g)
    # Vertical SGS momentum flux on FACES: τ_i3 = 2 ν_t S_i3, interpolated to
    # faces; the SURFACE face value is the MOST wall stress (not the interior
    # closure), the rigid-lid top is no-stress.
    nu_t_f = c2f(nu_t)
    dudz_f = ddz_c2f(u, dz)
    dvdz_f = ddz_c2f(v, dz)
    # FULL vertical SGS stress τ_i3 = 2 ν_t S_i3 = ν_t(∂u_i/∂z + ∂w/∂x_i) on the
    # interior faces (codex faithfulness: the ∂w/∂x, ∂w/∂y terms were missing).
    dwdx_i = ddx(w, g)[..., 1:-1]                           # faces → interior (nz-1)
    dwdy_i = ddy(w, g)[..., 1:-1]
    tau13_f = nu_t_f * (dudz_f + dwdx_i)                    # interior faces (nz-1)
    tau23_f = nu_t_f * (dvdz_f + dwdy_i)
    # MOST neutral wall stress at the first centre level z_c[0], Moeng (1984)
    # formulation: the drag uses the PLANAR-MEAN speed ⟨|u₁|⟩, not the local
    # instantaneous |u₁|. Using the local speed makes τ_w ∝ u₁² over-respond to
    # near-wall fluctuations, a positive feedback that pumps the resolved
    # turbulence to ~5× its physical level. With the mean speed the stress
    # MAGNITUDE is set by the mean wind (∝ u_*²) and only its DIRECTION follows
    # the local wind — the standard, well-behaved ABL-LES wall model.
    u1, v1 = u[..., 0], v[..., 0]
    spd1 = jnp.sqrt(u1 ** 2 + v1 ** 2 + 1e-12)
    spd1_mean = _planar_mean(spd1, g)                       # planar mean ⟨|u₁|⟩
    Cd = (kappa / jnp.log(g.z_c[0] / g.cfg.z0)) ** 2
    tau_w_x = -Cd * spd1_mean * u1                          # ∝ ⟨U⟩·u₁ (kinematic)
    tau_w_y = -Cd * spd1_mean * v1
    u_star = (Cd ** 0.5) * spd1_mean
    # Assemble full-face momentum flux: [surface, interior, top=0]. The surface
    # face carries the wall stress τ_w = -Cd⟨U⟩u₁ (a momentum SINK: the
    # divergence ∂_z(flux) then decelerates the near-surface wind). Interior
    # faces carry the down-gradient SGS flux ν_t ∂u/∂z; the rigid lid is no-flux.
    z = jnp.zeros_like(u1)[..., None]
    tau13_full = jnp.concatenate([(-tau_w_x)[..., None], tau13_f, z], axis=-1)
    tau23_full = jnp.concatenate([(-tau_w_y)[..., None], tau23_f, z], axis=-1)
    # ∂τ_i3/∂z at centres.
    Fu = Fu + ddz_f2c(tau13_full, dz)
    Fv = Fv + ddz_f2c(tau23_full, dz)
    # Vertical SGS force on interior FACES: ∂_x τ13 + ∂_y τ23 + ∂_z τ33.
    #   τ13_f, τ23_f live on interior faces (nz-1); τ33 at centres (nz).
    Fw_int = (ddx(tau13_f, g) + ddy(tau23_f, g)
              + ddz_c2f(tau33, dz))                        # (ny, nx, nz-1)
    Fw = jnp.pad(Fw_int, ((0, 0), (0, 0), (1, 1)))         # 0 at the walls → nz+1
    return Fu, Fv, Fw, u_star


# --------------------------------------------------------------------------- #
# Potential-temperature scalar transport + Boussinesq buoyancy                 #
# --------------------------------------------------------------------------- #
def scalar_rhs(theta, u, v, w, nu_t, g: SpectralLESGrid, sfc_flux):
    """RHS of the θ equation: ``−u·∇θ + ∂_j(K_h ∂_j θ)`` with the SURFACE vertical
    flux replaced by the prescribed kinematic heat flux ``sfc_flux = ⟨w'θ'⟩_0``
    (CBL: +ve heating; SBL: −ve cooling), no-flux lid. ``K_h = ν_t / Pr``.

    Advection in flux/skew form (∇·u=0 ⇒ −u·∇θ); horizontal derivatives spectral
    (de-aliased by 3/2 padding), the vertical ``w ∂θ/∂z`` product formed at faces
    then averaged. θ at centres, w at faces."""
    dz = g.dz
    dthdx, dthdy = ddx(theta, g), ddy(theta, g)            # centres
    dthdz_f = jnp.pad(ddz_c2f(theta, dz), ((0, 0), (0, 0), (1, 1)))  # faces, 0 walls
    if g.cfg.dealias:
        # Pass g so the 3/2 pad/truncate use the DISTRIBUTED global-y transform
        # under MPI (theta.shape[0] is ny_LOCAL on a slab). (codex 2026-06-09.)
        pf = lambda x: _pad_to_fine(x, g)                  # noqa: E731
        tf = lambda x: _truncate_from_fine(x, g.cfg.ny, g.cfg.nx, g)  # noqa: E731
        adv = tf(pf(u) * pf(dthdx) + pf(v) * pf(dthdy)
                 + f2c(pf(w) * pf(dthdz_f)))
    else:
        adv = u * dthdx + v * dthdy + f2c(w * dthdz_f)
    # Horizontal SGS scalar flux divergence (spectral).
    Kh = nu_t / g.cfg.pr_sgs
    Hsgs = ddx(Kh * dthdx, g) + ddy(Kh * dthdy, g)
    # Vertical SGS flux on interior faces + surface flux BC + no-flux lid.
    Kh_f = c2f(Kh)
    flux_int = Kh_f * ddz_c2f(theta, dz)                   # (ny,nx,nz-1), J=K∂θ/∂z
    z = jnp.zeros_like(theta[..., :1])
    # Surface face carries J = −⟨w'θ'⟩_0 (the diffusive flux is −w'θ'): a +ve
    # kinematic heat flux ``sfc_flux`` (heating) ⇒ ∂_z(J) WARMS the lowest cell.
    flux_full = jnp.concatenate([-sfc_flux + z, flux_int, z], axis=-1)  # nz+1
    Vsgs = ddz_f2c(flux_full, dz)
    return -adv + Hsgs + Vsgs


def buoyancy_w(theta, g: SpectralLESGrid):
    """Boussinesq buoyancy on the w-faces: ``b = (g/θ_ref0)·(θ − ⟨θ⟩_xy)`` — only
    the deviation from the horizontal-mean profile drives the eddies (the mean is
    in hydrostatic balance, absorbed by the pressure)."""
    g_over_th = constants.g / g.cfg.theta_ref0
    b_c = g_over_th * (theta - _planar_mean(theta, g, keepdims=True))
    bf = jnp.pad(c2f(b_c), ((0, 0), (0, 0), (1, 1)))        # faces, 0 at walls
    return bf


# --------------------------------------------------------------------------- #
# Pressure projection (fractional step)                                        #
# --------------------------------------------------------------------------- #
def project(u_s, v_s, w_s, dt, g: SpectralLESGrid):
    """Enforce ``∇·u=0`` by a pressure projection. Solve per horizontal
    wavenumber a tridiagonal vertical Poisson ``φ''−k²φ = div(u*)/dt`` with
    Neumann (``w=0``) walls, then ``u = u* − dt ∇φ``."""
    nz = u_s.shape[-1]
    # Tridiagonal arrays live in SPECTRAL space: (ny_global, nkx_local, nz) under
    # MPI, (ny, nxr, nz) serially — take the horizontal shape from g.kx (already
    # sliced to this rank's kx columns), NOT from the physical slab u_s.
    nyk, nxr = g.kx.shape
    dz = g.dz
    uh = _fft(u_s, g) * (1j * g.kx[..., None])
    vh = _fft(v_s, g) * (1j * g.ky[..., None])
    # divergence at centres: i kx û + i ky v̂ + ∂w/∂z|_c
    wzh = _fft(ddz_f2c(w_s, dz), g)
    div_h = uh + vh + wzh                                   # (nyk, nxr, nz) complex
    rhs = div_h / dt
    # Tridiagonal in z (centres), Neumann walls: φ[-1]=φ[0], φ[nz]=φ[nz-1].
    k2 = g.k2[..., None]                                    # (nyk, nxr, 1)
    inv_dz2 = 1.0 / dz ** 2
    a = jnp.full((nyk, nxr, nz), inv_dz2, dtype=rhs.dtype)   # sub
    c = jnp.full((nyk, nxr, nz), inv_dz2, dtype=rhs.dtype)   # super
    b = -(2.0 * inv_dz2 + k2) * jnp.ones_like(a)
    # Neumann: top/bottom diagonal loses one neighbour coupling.
    b = b.at[..., 0].set(-(inv_dz2 + k2[..., 0]))
    b = b.at[..., -1].set(-(inv_dz2 + k2[..., 0]))
    a = a.at[..., 0].set(0.0)
    c = c.at[..., -1].set(0.0)
    # k²=0 modes (the global mean + the Nyquist-zeroed columns) still need their
    # VERTICAL Poisson solved (dw/dz ≠ 0 there); the pure-Neumann vertical matrix
    # is singular by one constant, so pin only the GAUGE — φ[0]=0 — and keep the
    # interior vertical Laplacian. (Pinning the whole mode to 0 would leave the
    # vertical divergence of these columns uncorrected — an O(5%) leak.)
    sing = (g.k2 == 0.0)                                   # (ny, nxr)
    s0 = sing[..., None]
    b = b.at[..., 0].set(jnp.where(sing, 1.0, b[..., 0]))
    c = c.at[..., 0].set(jnp.where(sing, 0.0, c[..., 0]))
    rhs = rhs.at[..., 0].set(jnp.where(sing, 0.0 + 0.0j, rhs[..., 0]))
    del s0
    phi_h = _thomas_complex(a, b, c, rhs)
    # u = u* - dt ∇φ.  Horizontal grad spectral; vertical grad to faces.
    phi = _ifft(phi_h, g)
    u_new = u_s - dt * ddx(phi, g)
    v_new = v_s - dt * ddy(phi, g)
    dphidz_f = jnp.pad(ddz_c2f(phi, dz), ((0, 0), (0, 0), (1, 1)))    # 0 at walls
    w_new = w_s - dt * dphidz_f
    w_new = w_new.at[..., 0].set(0.0).at[..., -1].set(0.0)
    return u_new, v_new, w_new


def _thomas_complex(a, b, c, d):
    """Thomas tridiagonal solve along the LAST axis (complex RHS, real coeffs).

    Static Python loop over ``n`` (= nz, small) so it fully unrolls under JIT —
    robust (no scan-reconstruction indexing) and exact for the projection."""
    n = d.shape[-1]
    cp = [None] * n
    dp = [None] * n
    cp[0] = c[..., 0] / b[..., 0]
    dp[0] = d[..., 0] / b[..., 0]
    for k in range(1, n):
        m = b[..., k] - a[..., k] * cp[k - 1]
        cp[k] = c[..., k] / m
        dp[k] = (d[..., k] - a[..., k] * dp[k - 1]) / m
    x = [None] * n
    x[n - 1] = dp[n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return jnp.stack(x, axis=-1)


# --------------------------------------------------------------------------- #
# One AB2 time step                                                            #
# --------------------------------------------------------------------------- #
def rhs(u, v, w, g: SpectralLESGrid, u_geo, f_cor, force=(0.0, 0.0),
        theta=None, sfc_theta_flux=0.0):
    """Momentum RHS = -advection + SGS force + Coriolis + a constant body force.

    ``f_cor``≠0 drives a geostrophic/Ekman balance toward ``u_geo=(ug,vg)``; a
    constant ``force=(fx,fy)`` drives a pressure-gradient channel (``fx=u_*²/Lz``
    gives a target ``u_*`` and a log-law equilibrium in a few eddy turnovers —
    the clean Monin–Obukhov validation case)."""
    Cu, Cv, Cw = advection(u, v, w, g)
    nu_t = eddy_viscosity(u, v, w, g)
    Fu, Fv, Fw, u_star = sgs_and_wall(u, v, w, nu_t, g, u_geo)
    ug, vg = u_geo
    Ru = Cu + Fu + f_cor * (v - vg) + force[0]
    Rv = Cv + Fv - f_cor * (u - ug) + force[1]
    Rw = Cw + Fw
    Rtheta = None
    if theta is not None:
        Rtheta = scalar_rhs(theta, u, v, w, nu_t, g, sfc_theta_flux)
        if g.cfg.buoyancy:
            Rw = Rw + buoyancy_w(theta, g)
    Rw = Rw.at[..., 0].set(0.0).at[..., -1].set(0.0)
    return Ru, Rv, Rw, u_star, Rtheta


def _surface_ustar(u, v, g: SpectralLESGrid):
    """Surface friction velocity u_* = √Cd · ⟨|u₁|⟩ (the neutral MOST wall model,
    same formula as :func:`sgs_and_wall`) — a cheap diagnostic recomputed from the
    lowest-level (u, v) after a step (the shared RK integrator does not thread it
    through)."""
    kappa = constants.kappa_von_karman
    u1, v1 = u[..., 0], v[..., 0]
    Cd = (kappa / jnp.log(g.z_c[0] / g.cfg.z0)) ** 2
    return (Cd ** 0.5) * _planar_mean(jnp.sqrt(u1 ** 2 + v1 ** 2 + 1e-12), g)


def _filt_state(u, v, w, th, g):
    """Apply the high-k cutoff to a state (no-op if disabled). Divergence-free
    preserving (mask uniform in z); re-zeros the w walls."""
    if not g.cfg.spectral_filter:
        return u, v, w, th
    u = _apply_filter(u, g)
    v = _apply_filter(v, g)
    w = _apply_filter(w, g).at[..., 0].set(0.0).at[..., -1].set(0.0)
    th = None if th is None else _apply_filter(th, g)
    return u, v, w, th


def step(state: SpectralLESState, g: SpectralLESGrid, dt,
         u_geo, f_cor: float, first: bool = False, force=(0.0, 0.0),
         sfc_theta_flux=0.0):
    """One time step + pressure projection.

    ``time_scheme`` selects the integrator. The RK options ("rk3"=SSP-RK3,
    "ssp_rk34", "ssp_rk54") REUSE the shared ``timestepping.split_explicit`` SSP-RK
    drivers (pytree-generic, AD-safe), with the incompressible pressure PROJECTION
    supplied as the per-stage ``acoustic_update_fn`` (this core has no acoustic
    substep). "ab2" is a self-contained Adams–Bashforth-2 fallback (``first`` uses
    forward Euler). When ``state.theta`` is set the scalar is advanced with the same
    scheme and Boussinesq buoyancy is added to w. A high-k spectral cutoff is applied
    once at the end. ``dt`` may be a Python float or a JAX scalar."""
    u, v, w, th = state.u, state.v, state.w, state.theta
    if g.cfg.time_scheme == "ab2":
        Ru, Rv, Rw, u_star, Rth = rhs(u, v, w, g, u_geo, f_cor, force=force,
                                      theta=th, sfc_theta_flux=sfc_theta_flux)
        if first:
            au, av, aw, ath = Ru, Rv, Rw, Rth
        else:
            au = 1.5 * Ru - 0.5 * state.rhs_u_prev
            av = 1.5 * Rv - 0.5 * state.rhs_v_prev
            aw = 1.5 * Rw - 0.5 * state.rhs_w_prev
            ath = None if Rth is None else 1.5 * Rth - 0.5 * state.rhs_theta_prev
        w_s = (w + dt * aw).at[..., 0].set(0.0).at[..., -1].set(0.0)
        u_n, v_n, w_n = project(u + dt * au, v + dt * av, w_s, dt, g)
        th_n = None if th is None else th + dt * ath
        u_n, v_n, w_n, th_n = _filt_state(u_n, v_n, w_n, th_n, g)
        return SpectralLESState(u=u_n, v=v_n, w=w_n, theta=th_n,
                                rhs_theta_prev=Rth, rhs_u_prev=Ru,
                                rhs_v_prev=Rv, rhs_w_prev=Rw), u_star

    # --- SSP-RK via the shared integrator -------------------------------------
    # slow_fn returns the RHS packed as a state pytree (history fields zeroed so the
    # integrator's axpy/linear-combination leave them at 0); the per-stage "fast"
    # update is the incompressible pressure projection (no acoustic substep here).
    def slow_fn(s):
        Ru, Rv, Rw, _u, Rth = rhs(s.u, s.v, s.w, g, u_geo, f_cor, force=force,
                                  theta=s.theta, sfc_theta_flux=sfc_theta_flux)
        return SpectralLESState(
            u=Ru, v=Rv, w=Rw, theta=Rth,
            rhs_u_prev=jnp.zeros_like(Ru), rhs_v_prev=jnp.zeros_like(Rv),
            rhs_w_prev=jnp.zeros_like(Rw),
            rhs_theta_prev=None if Rth is None else jnp.zeros_like(Rth))

    def proj_fn(s_slow, slow_tend, dt_sub, n_sub, cfg):    # acoustic_update_fn slot
        wz = s_slow.w.at[..., 0].set(0.0).at[..., -1].set(0.0)
        un, vn, wn = project(s_slow.u, s_slow.v, wz, dt_sub, g)
        return s_slow._replace(u=un, v=vn, w=wn)           # θ untouched by projection

    se_cfg = SplitExplicitConfig(n_substeps=1, outer_integrator=g.cfg.time_scheme)
    out = split_explicit_step(state, slow_fn, proj_fn, dt, se_cfg)
    u_n, v_n, w_n, th_n = _filt_state(out.u, out.v, out.w, out.theta, g)
    # u_* diagnostic from the RETURNED (filtered) state so it matches what the
    # caller sees (the filter is a horizontal low-pass; the surface-layer effect is
    # tiny, but keep them consistent).
    u_star = _surface_ustar(u_n, v_n, g)
    return SpectralLESState(
        u=u_n, v=v_n, w=w_n, theta=th_n,
        rhs_u_prev=jnp.zeros_like(u_n), rhs_v_prev=jnp.zeros_like(v_n),
        rhs_w_prev=jnp.zeros_like(w_n),
        rhs_theta_prev=None if th_n is None else jnp.zeros_like(th_n)), u_star
