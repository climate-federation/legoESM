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
from legoesm.core.bulk_flux import psi_h, psi_m   # canonical MOST stability functions
from legoesm.atmosphere.physics._shared import (
    brunt_vaisala_n_squared_from_gradient,
    lilly_buoyancy_factor,
)
from legoesm.atmosphere.physics.turbulence.lasd_core import lasd_cs2
from legoesm.atmosphere.physics.turbulence.vreman import vreman_nu_t as _vreman_core
from legoesm.timestepping.split_explicit import (
    SplitExplicitConfig, split_explicit_step)
from legoesm.timestepping.tridiagonal import thomas_solve


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
    sgs_buoyancy: bool = False     # multiply the strain-based ν_t by the Lilly (1962)
    #                                buoyancy factor √(max(0, 1 − Ri/Pr_t)), Ri =
    #                                N²/|S|² from the (virtual) θ gradient — suppresses
    #                                SGS mixing at a stable inversion. REQUIRED for
    #                                stratocumulus (DYCOMS): the strain-only ν_t
    #                                over-entrains the cloud-top jump → thin cloud.
    #                                Uses pr_sgs as the critical Ri_c. Default off keeps
    #                                the neutral ABL byte-identical (N²≈0 ⇒ f≈1 but not
    #                                EXACTLY 1 at fp; gate preserves reproducibility).
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
    moist: bool = False             # moist θ_v buoyancy from the water tracers:
    #                                 b = (g/θ_ref0)·(θ_v − ⟨θ_v⟩) with
    #                                 θ_v = θ·(1 + (1/ε−1)·q_v − q_c − q_r) (vapour
    #                                 buoyancy + liquid condensate loading). Requires
    #                                 buoyancy=True and a state with tracers. θ stays
    #                                 the FULL potential temperature (the microphysics
    #                                 coupling applies the latent heating to it).
    scalar_advection: str = "van_leer"  # face reconstruction for the monotone
    #                                 flux-form scalar transport: "van_leer" (2nd
    #                                 order TVD, robust but diffusive) or "weno5"
    #                                 (5th-order WENO-Z, reuses core.weno — far
    #                                 less numerical diffusion ⇒ preserves the
    #                                 cloud-layer moisture van-Leer erodes). Only
    #                                 used when monotone_scalars.
    w_hyperdiff_coeff: float = 0.0  # OPT-IN horizontal biharmonic hyperdiffusion on
    #                                 w: dw/dt −= ν₄·(∇²_h)²w = −ν₄·k⁴·ŵ (spectral,
    #                                 horizontal-only). This core has NO numerical
    #                                 hyperdiffusion; the sharp latent heating of a
    #                                 sharp (WENO5) cloud field drives a grid-scale w
    #                                 mode the near-inviscid momentum dynamics cannot
    #                                 dissipate → the MOMENTUM-side lever that lets
    #                                 WENO5 scalars run stable WITHOUT broad scalar
    #                                 diffusion (which would erase WENO5's sharpness).
    #                                 Start ≈ 0.25·dx⁴/(π⁴·dt) (≈1e5 m⁴/s, dx=100,
    #                                 dt=2). Explicit-stability radius λ_max·dt
    #                                 with λ_max=ν₄·max(k⁴): ≈2.51 (RK3), ≈1.0
    #                                 (AB2), 2.0 (FE start). 0 ⇒ off.
    theta_hyperdiff_coeff: float = 0.0  # OPT-IN SCALE-SELECTIVE horizontal k⁴
    #                                 hyperdiffusion on θ ONLY: dθ/dt −= ν₄θ·k⁴·θ̂.
    #                                 NOT broad scalar diffusion — k⁴ damps the 2Δ
    #                                 θ' noise that the sharp (WENO5) latent heating
    #                                 injects (the actual instability SOURCE: grid-
    #                                 scale θ' → buoyancy → grid-scale w) while
    #                                 barely touching the resolved θ and leaving the
    #                                 moisture q fully WENO5-sharp. Same scaling/
    #                                 CFL as w_hyperdiff. NOTE (codex 2026-06-13):
    #                                 the k⁴ uses g.k2 whose Nyquist row/col are
    #                                 zeroed (derivative consistency) ⇒ this damps
    #                                 the RETAINED non-Nyquist modes; the exact 2Δ
    #                                 Nyquist mode is removed by ``spectral_filter``
    #                                 (the 2/3 cutoff), not here.
    div_damping_coeff: float = 0.0  # OPT-IN momentum divergence damping
    #                                 du/dt += α·∇(∇·u) (⇒ ∂δ/∂t += α·∇²δ, α>0 DAMPS
    #                                 divergence). On this INCOMPRESSIBLE core the
    #                                 pressure projection already removes ∇·u each
    #                                 step, so this is LARGELY REDUNDANT (kept for
    #                                 completeness; prefer w_hyperdiff). Units m²/s;
    #                                 start ≈ 0.05·dx²/dt. 0 ⇒ off.
    monotone_scalars: bool = False  # use the conservative van-Leer (TVD) flux-form
    #                                 scalar transport for θ + tracers instead of the
    #                                 spectral/centred advection. REQUIRED for moist
    #                                 runs: the spectral scalar advection is
    #                                 NON-MONOTONE and rings (Gibbs over/undershoot)
    #                                 at the sharp moisture inversion, which the
    #                                 saturation-adjustment latent heating amplifies
    #                                 into a grid-scale instability (whole-column
    #                                 runaway condensation). van-Leer is monotone +
    #                                 conservative + positivity-friendly. Dry θ runs
    #                                 keep the validated spectral advection (False).
    n_tracers: int = 0              # trailing water-tracer fields on the state, the
    #                                 STANDARD microphysics slot layout ([0]=q_v,
    #                                 [1]=q_c, [2]=q_r, [3]=q_i, [4]=q_s, [5]=q_g,
    #                                 [6]=N_c, [7]=N_r, [8]=N_i — see microphysics/
    #                                 integration._PLANE_MIN_TRACER_SLOTS) so ANY
    #                                 scheme swaps in unchanged. 0 ⇒ dry (unchanged).
    filter_monotone_theta: bool = True  # filtering θ is useful buoyancy-noise
    #                                 control and has no positivity contract.
    filter_monotone_qv: bool = False  # optional vapor-only smoothing; hydrometeors
    #                                 stay unfiltered to preserve monotonicity at
    #                                 cloud edges.
    filter_monotone_scalars: bool = False  # diagnostic/back-compat switch.  The
    #                                 sharp spectral cutoff is non-monotone, so the
    #                                 moist monotone scalar path leaves water
    #                                 tracers unfiltered by default; set True only
    #                                 to reproduce the old filtered-tracer behavior.


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
    tracers: jax.Array | None = None    # (ny,nx,nz,n_tracers) water tracers, the
    #                                     standard microphysics slot layout (see
    #                                     SpectralLESConfig.n_tracers)
    rhs_tracers_prev: jax.Array | None = None


def make_grid(cfg: SpectralLESConfig, dtype=jnp.float64,
              layout: SpectralLESLayout | None = None) -> SpectralLESGrid:
    # SGS constants must be non-negative or ν_t can go negative (anti-diffusion,
    # blow-up). The shared vreman/Smagorinsky cores trust these — validate here,
    # the single point where a config becomes a runnable grid.
    if cfg.c_s < 0.0 or cfg.c_vreman < 0.0 or cfg.nu_floor < 0.0:
        raise ValueError(
            f"SGS constants must be >= 0: c_s={cfg.c_s}, c_vreman={cfg.c_vreman}, "
            f"nu_floor={cfg.nu_floor} (negative ν_t is anti-diffusive).")
    if cfg.scalar_advection not in ("van_leer", "weno5", "weno5_hv"):
        raise ValueError(
            f"scalar_advection must be 'van_leer', 'weno5' or 'weno5_hv', got "
            f"{cfg.scalar_advection!r}.")
    if (cfg.scalar_advection in ("weno5", "weno5_hv")
            and jnp.dtype(dtype) == jnp.float32):
        # WENO-Z float32 hazard (codex 2026-06-13): the smoothness indicators β
        # are formed from RAW-VALUE squared differences; for fields with a large
        # mean and small fluctuation (θ≈300 K + O(0.1 K) eddies, the moisture
        # inversion) float32 cancellation makes β noisy/negative → the run NaNs
        # at cloud onset. float64 WENO5 is stable (this, NOT a dynamical
        # instability, was the earlier blow-up). The robust f32 fix is a common-
        # shift of the stencil before β (reconstruct the original values) in
        # core.weno — follow-up; for now WARN + recommend float64.
        import warnings
        warnings.warn(
            "WENO5 scalar advection in float32: the WENO-Z smoothness indicators "
            "suffer raw-value cancellation (large mean θ + small fluctuations) "
            "and can NaN at cloud onset. Run WENO5 moist cases in float64 "
            "(omit --f32 / JAX_ENABLE_X64=1), or add a common-shifted-β WENO.",
            stacklevel=2)
    if layout is not None and cfg.monotone_scalars:
        # The monotone flux operators use a LOCAL jnp.roll in y, which wraps
        # within each MPI slab instead of exchanging halos at the slab boundary
        # (codex 2026-06-12; serial-vs-2-slab probe diverged by ~46). Reject
        # until a y-halo exchange is added for the face stencils + velocities.
        raise NotImplementedError(
            "monotone_scalars is not yet MPI-safe (the y-roll wraps per slab; "
            "needs a halo exchange). Run monotone moist cases single-rank, or "
            "use the spectral scalar advection under MPI.")
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


def sgs_buoyancy_factor(theta, tracers, u, v, w, g: SpectralLESGrid):
    """Lilly (1962) stable-stratification suppression factor for the SGS eddy
    viscosity, at CENTRES: ``f = √(max(0, 1 − Ri/Pr_t))`` with ``Ri = N²/|S|²``.

    ``N² = (g/θ_v)·∂θ_v/∂z`` from the resolved virtual potential temperature
    (:func:`virtual_theta`; dry ``θ`` when ``tracers`` is None), ``|S|`` the same
    strain magnitude the Smagorinsky/Vreman/LASD ``ν_t`` uses. Multiplying ``ν_t``
    by ``f`` shuts SGS mixing off across a stable inversion (``Ri ≥ Pr_t``) — the
    missing physics that makes the strain-only closure over-entrain the
    stratocumulus cloud top. Reuses the shared
    :func:`~legoesm.atmosphere.physics._shared.lilly_buoyancy_factor` /
    ``brunt_vaisala_n_squared_from_gradient`` (no re-derived Ri form).

    ponytail: recomputes ``_strain`` (also done in :func:`eddy_viscosity`) rather
    than threading ``Smag`` out — only runs when ``sgs_buoyancy`` is on; make
    ``eddy_viscosity`` return ``Smag`` if this doubling ever shows up in a profile.
    """
    (S11, S22, S33, S12, S13, S23), _Smag = _strain(u, v, w, g)
    theta_v = virtual_theta(theta, tracers) if tracers is not None else theta
    theta_v = jnp.clip(theta_v, 1.0, None)                  # keep g/θ_v finite
    # ∂θ_v/∂z at centres: 2nd-order central interior + one-sided edges (uniform
    # dz on this plane core), matching the centre layout of ν_t.
    dtheta_v_dz = jnp.gradient(theta_v, g.dz, axis=-1)
    n2 = brunt_vaisala_n_squared_from_gradient(theta_v, dtheta_v_dz)
    # |S|² = 2 S_ij S_ij built DIRECTLY from the strain components, NOT Smag**2.
    # Same value as Smag² (pre the 1e-30 clamp inside _strain, negligible vs the
    # 1e-10 floor) but self-contained AD-safe: a differentiable LES must not rely
    # on _strain's INTERNAL √-clamp (there for a different purpose) to keep THIS
    # Ri denominator's adjoint finite at an exact zero-strain (rest/uniform)
    # state, and it skips a wasteful √→square roundtrip. (codex 2026-07-16 flagged
    # the √→square as a NaN risk; the clamp happens to save the forward, this
    # removes the fragile coupling.)
    s2 = 2.0 * (S11 ** 2 + S22 ** 2 + S33 ** 2
                + 2.0 * (S12 ** 2 + S13 ** 2 + S23 ** 2)) + 1e-10
    return lilly_buoyancy_factor(n2 / s2, g.cfg.pr_sgs)


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
# Coupled stable Monin–Obukhov surface layer (GABLS1 prescribed-cooling BC)     #
# --------------------------------------------------------------------------- #
def most_surface_flux(spd_mean, th_air_mean, t_sfc, z1, z0, theta_ref, n_iter=10):
    """Coupled Monin–Obukhov surface layer (planar-mean, GABLS1-faithful).

    Given the planar-mean first-level wind speed ``spd_mean=⟨|u₁|⟩`` and potential
    temperature ``th_air_mean=⟨θ₁⟩`` at height ``z1`` and the (cooled) surface
    temperature ``t_sfc``, solve the MOST profile relations by a fixed-point
    iteration on the Obukhov length::

        u_*  = κ ⟨U⟩       / (ln(z₁/z0) − ψ_m(ζ))
        θ_*  = κ (⟨θ₁⟩−T_s) / (ln(z₁/z0) − ψ_h(ζ))
        1/L  = κ g θ_* / (u_*² θ_ref),   ζ = z₁/L

    The stability functions ``ψ_m, ψ_h`` are the SHARED canonical Businger–Dyer
    implementation (:func:`legoesm.core.bulk_flux.psi_m`/``psi_h``; stable branch
    Dyer 1974 ``−5ζ``, unstable Businger–Dyer, both internally clipped to
    ``ζ∈[−10,10]`` with safe-branch gradients) rather than a re-derived form.
    Thermal roughness ``z0h=z0`` (Beare et al. 2006 specify z0m=z0h=0.1 m).

    Returns ``(u_star, theta_star, q0, cd_eff)`` with the kinematic surface heat
    flux ``q0 = ⟨w'θ'⟩₀ = −u_* θ_*`` (negative ⇒ surface cooling) and the
    stability-corrected drag ``cd_eff = (u_*/⟨U⟩)²``. As the stable stratification
    strengthens both denominators grow, so ``u_*, θ_*, q0 → 0`` (turbulence shuts
    itself off) — the physical, self-limiting SBL behaviour the neutral drag law
    cannot reproduce.

    Differentiable / JIT-safe: a fixed Python-unrolled iteration on scalar
    (planar-mean) quantities, no data-dependent control flow."""
    kappa = constants.kappa_von_karman
    gacc = constants.g
    lnz = jnp.log(z1 / z0)
    dth = th_air_mean - t_sfc                     # >0 when air warmer than surface
    u_star = kappa * spd_mean / lnz               # neutral first guess
    th_star = kappa * dth / lnz
    for _ in range(n_iter):
        inv_L = kappa * gacc * th_star / (u_star ** 2 * theta_ref + 1e-12)
        zeta = z1 * inv_L                         # +ve stable, −ve unstable
        u_star = kappa * spd_mean / (lnz - psi_m(zeta))
        th_star = kappa * dth / (lnz - psi_h(zeta))
    q0 = -u_star * th_star
    cd_eff = (u_star / (spd_mean + 1e-12)) ** 2
    return u_star, th_star, q0, cd_eff


# --------------------------------------------------------------------------- #
# SGS stress divergence + MOST wall model                                      #
# --------------------------------------------------------------------------- #
def sgs_and_wall(u, v, w, nu_t, g: SpectralLESGrid, u_geo, cd_surf=None):
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
    # Neutral log-law drag, OR the stability-corrected drag ``cd_surf`` supplied by
    # the coupled MOST surface layer (GABLS1 prescribed-cooling BC). cd_surf=None
    # ⇒ neutral (unchanged default path, no regression).
    Cd = ((kappa / jnp.log(g.z_c[0] / g.cfg.z0)) ** 2
          if cd_surf is None else cd_surf)
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


def _vanleer_flux_div(phi, u, v, w, g: SpectralLESGrid):
    """Conservative van-Leer (TVD, monotone) flux-form advection divergence
    ``∇·(u φ)`` of a cell-centred scalar. u, v at centres, w at faces, ∇·u=0
    (post-projection) so the flux form equals ``u·∇φ``. Periodic in x, y; rigid
    walls in z (``w=0`` at the surface + lid ⇒ zero advective wall flux). Reuses
    the shared 4-cell reconstruction ``core.flux_limiters.van_leer_face_values``
    (single source of truth, same limiter as the plane CRM)."""
    from legoesm.core.flux_limiters import van_leer_face_values
    dx, dy, dz = g.dx, g.dy, g.dz
    nz = phi.shape[-1]

    # --- x faces (axis=1, periodic). Face i+1/2 from the 4-cell x-stencil. ---
    u_xf = 0.5 * (u + jnp.roll(u, -1, axis=1))             # u at i+1/2
    pp, pn = van_leer_face_values(
        jnp.roll(phi, 1, axis=1), phi,
        jnp.roll(phi, -1, axis=1), jnp.roll(phi, -2, axis=1))
    phi_xf = jnp.where(u_xf >= 0.0, pp, pn)
    Fx = u_xf * phi_xf
    divx = (Fx - jnp.roll(Fx, 1, axis=1)) / dx            # F[i+1/2]-F[i-1/2]

    # --- y faces (axis=0, periodic). ---
    v_yf = 0.5 * (v + jnp.roll(v, -1, axis=0))
    pp, pn = van_leer_face_values(
        jnp.roll(phi, 1, axis=0), phi,
        jnp.roll(phi, -1, axis=0), jnp.roll(phi, -2, axis=0))
    phi_yf = jnp.where(v_yf >= 0.0, pp, pn)
    Fy = v_yf * phi_yf
    divy = (Fy - jnp.roll(Fy, 1, axis=0)) / dy

    # --- z faces (axis=2, walls). Edge-pad φ by 2 for the boundary faces. ---
    phip = jnp.pad(phi, ((0, 0), (0, 0), (2, 2)), mode="edge")
    pp, pn = van_leer_face_values(
        phip[..., 0:nz + 1], phip[..., 1:nz + 2],
        phip[..., 2:nz + 3], phip[..., 3:nz + 4])         # faces 0..nz
    phi_zf = jnp.where(w >= 0.0, pp, pn)                   # w upward ⇒ from below
    Fz = w * phi_zf                                        # w=0 at walls ⇒ Fz=0
    divz = (Fz[..., 1:nz + 1] - Fz[..., 0:nz]) / dz
    return divx + divy + divz


def _vanleer_fv_velocity_divergence(u, v, w, g: SpectralLESGrid):
    """Finite-volume divergence seen by :func:`_vanleer_flux_div`.

    The pressure projection enforces the SPECTRAL divergence
    ``ddx(u)+ddy(v)+ddz(w)=0``.  The monotone flux operator uses arithmetic
    horizontal face velocities instead, so its finite-volume divergence is not
    exactly zero.  Scalar advection must subtract ``φ·div_fv(u)`` to preserve a
    constant scalar under the spectral projection.
    """
    dx, dy, dz = g.dx, g.dy, g.dz
    nz = u.shape[-1]
    u_xf = 0.5 * (u + jnp.roll(u, -1, axis=1))
    divx = (u_xf - jnp.roll(u_xf, 1, axis=1)) / dx
    v_yf = 0.5 * (v + jnp.roll(v, -1, axis=0))
    divy = (v_yf - jnp.roll(v_yf, 1, axis=0)) / dy
    divz = (w[..., 1:nz + 1] - w[..., 0:nz]) / dz
    return divx + divy + divz


def _weno5_flux_div(phi, u, v, w, g: SpectralLESGrid):
    """Conservative WENO5-Z flux-form advection divergence ``∇·(uφ)``.

    Identical flux-form structure to :func:`_vanleer_flux_div` (same arithmetic
    face velocities, same conservative differencing, same free-stream caveat —
    pair with ``_vanleer_fv_velocity_divergence``), but the face value is the
    6-cell WENO5-Z upwind reconstruction from the SHARED ``core.weno`` kernels
    (the oracle: identical to the ocean's vertical-tracer WENO). WENO5 is ~5th-
    order in smooth flow and only steepens near discontinuities ⇒ far less
    numerical diffusion than van-Leer, so it preserves the cloud-layer moisture
    that van-Leer over-mixes, while staying essentially non-oscillatory.

    Stencil for face i+1/2: ``[φ_{i-2},φ_{i-1},φ_i,φ_{i+1},φ_{i+2},φ_{i+3}]``
    (``weno5_z`` convention). Periodic x, y via roll; z edge-padded by 3 with
    the rigid-wall zero advective flux (w=0 at surface + lid)."""
    from legoesm.core.weno import weno5_z, weno_upwind
    dx, dy, dz = g.dx, g.dy, g.dz
    nz = phi.shape[-1]

    def faces_periodic(ax):
        st = [jnp.roll(phi, s, axis=ax) for s in (2, 1, 0, -1, -2, -3)]
        return weno5_z(st)                                 # (f_plus, f_minus)

    # x faces (axis=1)
    u_xf = 0.5 * (u + jnp.roll(u, -1, axis=1))
    fp, fm = faces_periodic(1)
    Fx = u_xf * weno_upwind(fp, fm, u_xf)
    divx = (Fx - jnp.roll(Fx, 1, axis=1)) / dx

    # y faces (axis=0)
    v_yf = 0.5 * (v + jnp.roll(v, -1, axis=0))
    fp, fm = faces_periodic(0)
    Fy = v_yf * weno_upwind(fp, fm, v_yf)
    divy = (Fy - jnp.roll(Fy, 1, axis=0)) / dy

    # z faces (axis=2, walls). Edge-pad by 3 for the 6-cell boundary stencils.
    phip = jnp.pad(phi, ((0, 0), (0, 0), (3, 3)), mode="edge")
    st = [phip[..., j:j + nz + 1] for j in range(6)]       # φ_{k-3..k+2}, faces 0..nz
    fp, fm = weno5_z(st)
    Fz = w * weno_upwind(fp, fm, w)                        # w=0 at walls ⇒ Fz=0
    divz = (Fz[..., 1:nz + 1] - Fz[..., 0:nz]) / dz
    return divx + divy + divz


def _weno5_hv_flux_div(phi, u, v, w, g: SpectralLESGrid):
    """HYBRID scalar advection divergence: 5th-order WENO-Z in the HORIZONTAL
    (x, y) + 2nd-order van-Leer in the VERTICAL (z).

    The moist instability lives at the sharp VERTICAL moisture inversion
    (∂q/∂z) — WENO5 there is too weakly dissipative and the latent-heat feedback
    blows up; van-Leer's diffusion at the inversion is stabilising. But the
    HORIZONTAL cloud field (cover, plume structure) wants WENO5's low diffusion
    — pure van-Leer over-mixes it and under-predicts cloud cover. This hybrid
    keeps each direction's right scheme. Conservative + free-stream-paired with
    ``_vanleer_fv_velocity_divergence`` (same arithmetic face velocities)."""
    from legoesm.core.weno import weno5_z, weno_upwind
    from legoesm.core.flux_limiters import van_leer_face_values
    dx, dy, dz = g.dx, g.dy, g.dz
    nz = phi.shape[-1]
    # x, y: WENO5
    u_xf = 0.5 * (u + jnp.roll(u, -1, axis=1))
    fp, fm = weno5_z([jnp.roll(phi, s, axis=1) for s in (2, 1, 0, -1, -2, -3)])
    Fx = u_xf * weno_upwind(fp, fm, u_xf)
    divx = (Fx - jnp.roll(Fx, 1, axis=1)) / dx
    v_yf = 0.5 * (v + jnp.roll(v, -1, axis=0))
    fp, fm = weno5_z([jnp.roll(phi, s, axis=0) for s in (2, 1, 0, -1, -2, -3)])
    Fy = v_yf * weno_upwind(fp, fm, v_yf)
    divy = (Fy - jnp.roll(Fy, 1, axis=0)) / dy
    # z: van-Leer (stable at the inversion)
    phip = jnp.pad(phi, ((0, 0), (0, 0), (2, 2)), mode="edge")
    pp, pn = van_leer_face_values(
        phip[..., 0:nz + 1], phip[..., 1:nz + 2],
        phip[..., 2:nz + 3], phip[..., 3:nz + 4])
    Fz = w * jnp.where(w >= 0.0, pp, pn)
    divz = (Fz[..., 1:nz + 1] - Fz[..., 0:nz]) / dz
    return divx + divy + divz


def scalar_rhs_monotone(phi, u, v, w, nu_t, g: SpectralLESGrid, sfc_flux):
    """Scalar tendency with MONOTONE (van-Leer TVD) advection + the SAME SGS
    diffusion + surface-flux BC as :func:`scalar_rhs`.

    ``_vanleer_flux_div`` is globally conservative, but its finite-volume face
    velocities are not exactly divergence-free after the SPECTRAL pressure
    projection.  Use the advective correction ``∇·(uφ) − φ∇_fv·u`` here so a
    uniform θ/q field remains uniform.  Without this free-stream-preserving
    correction, moist runs generate artificial saturation anomalies that the
    microphysics/buoyancy feedback explosively amplifies.
    """
    dz = g.dz
    flux_div = {"weno5": _weno5_flux_div, "weno5_hv": _weno5_hv_flux_div,
                "van_leer": _vanleer_flux_div}[g.cfg.scalar_advection]
    adv = (flux_div(phi, u, v, w, g)
           - phi * _vanleer_fv_velocity_divergence(u, v, w, g))
    dthdx, dthdy = ddx(phi, g), ddy(phi, g)
    Kh = nu_t / g.cfg.pr_sgs
    Hsgs = ddx(Kh * dthdx, g) + ddy(Kh * dthdy, g)
    Kh_f = c2f(Kh)
    flux_int = Kh_f * ddz_c2f(phi, dz)
    z = jnp.zeros_like(phi[..., :1])
    flux_full = jnp.concatenate([-sfc_flux + z, flux_int, z], axis=-1)
    Vsgs = ddz_f2c(flux_full, dz)
    return -adv + Hsgs + Vsgs


def virtual_theta(theta, tracers):
    """Virtual potential temperature with liquid-water loading:
    ``θ_v = θ·(1 + (1/ε−1)·q_v − q_c − q_r)``.

    ``(1/ε−1) ≈ 0.608`` from the shared ``constants.epsilon`` (R_d/R_v) — no
    re-derived 0.61 literal. Ice slots are ignored (warm-cloud LES); extend with
    ``− q_i − q_s − q_g`` if a mixed-phase case is ever run on this core."""
    eps_v = 1.0 / constants.epsilon - 1.0
    q_v = tracers[..., 0]
    q_c = tracers[..., 1] if tracers.shape[-1] > 1 else 0.0
    q_r = tracers[..., 2] if tracers.shape[-1] > 2 else 0.0
    return theta * (1.0 + eps_v * q_v - q_c - q_r)


def buoyancy_w_moist(theta, tracers, g: SpectralLESGrid):
    """Moist Boussinesq buoyancy on the w-faces:
    ``b = (g/θ_ref0)·(θ_v − ⟨θ_v⟩_xy)`` (same anomaly form as :func:`buoyancy_w`
    — the planar mean is absorbed by the pressure projection)."""
    th_v = virtual_theta(theta, tracers)
    g_over_th = constants.g / g.cfg.theta_ref0
    b_c = g_over_th * (th_v - _planar_mean(th_v, g, keepdims=True))
    return jnp.pad(c2f(b_c), ((0, 0), (0, 0), (1, 1)))      # faces, 0 at walls


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
    phi_h = thomas_solve(a, b, c, rhs)
    # u = u* - dt ∇φ.  Horizontal grad spectral; vertical grad to faces.
    phi = _ifft(phi_h, g)
    u_new = u_s - dt * ddx(phi, g)
    v_new = v_s - dt * ddy(phi, g)
    dphidz_f = jnp.pad(ddz_c2f(phi, dz), ((0, 0), (0, 0), (1, 1)))    # 0 at walls
    w_new = w_s - dt * dphidz_f
    w_new = w_new.at[..., 0].set(0.0).at[..., -1].set(0.0)
    return u_new, v_new, w_new


# --------------------------------------------------------------------------- #
# One AB2 time step                                                            #
# --------------------------------------------------------------------------- #
def _horizontal_hyperdiff(f, nu4, g: SpectralLESGrid):
    """Scale-selective horizontal biharmonic hyperdiffusion tendency
    ``−ν₄·(∇²_h)²f = −ν₄·k⁴·f̂`` (k⁴=(kx²+ky²)²), applied per level in spectral
    space. Works for cell-centred (nz) or face (nz+1) fields. k⁴ makes it damp
    the 2Δ grid mode hard while barely touching the resolved scales — so it does
    NOT act as broad diffusion (the WENO5 sharpness in the energy-containing
    scales is preserved)."""
    k4 = (g.k2 ** 2)[..., None]                            # (nyk, nxr, 1)
    return _ifft(-nu4 * k4 * _fft(f, g), g)


def _w_hyperdiffusion(w, g: SpectralLESGrid):
    """OPT-IN momentum-side w-hyperdiffusion (see :func:`_horizontal_hyperdiff`).
    Targets the grid-scale w mode; w on faces, walls re-zeroed by the caller."""
    return _horizontal_hyperdiff(w, g.cfg.w_hyperdiff_coeff, g)


def _divergence_damping(u, v, w, g: SpectralLESGrid):
    """Momentum divergence damping tendency ``α·∇(∇·u)`` (OPT-IN). ``∇·u`` is
    formed at centres (spectral horizontal + face→centre vertical); its gradient
    is added back to u, v (centres) and w (faces). α>0 damps divergence
    (∂δ/∂t += α∇²δ). On the incompressible projection core ∇·u≈0 already, so this
    is mostly redundant — see the config note; ``_w_hyperdiffusion`` is the
    operative momentum lever here."""
    a = g.cfg.div_damping_coeff
    dz = g.dz
    div_c = ddx(u, g) + ddy(v, g) + ddz_f2c(w, dz)        # (ny,nx,nz) at centres
    dRu = a * ddx(div_c, g)
    dRv = a * ddy(div_c, g)
    dRw = a * jnp.pad(ddz_c2f(div_c, dz), ((0, 0), (0, 0), (1, 1)))  # faces, 0 walls
    return dRu, dRv, dRw


def rhs(u, v, w, g: SpectralLESGrid, u_geo, f_cor, force=(0.0, 0.0),
        theta=None, sfc_theta_flux=0.0, t_sfc=None,
        tracers=None, sfc_qv_flux=0.0):
    """Momentum RHS = -advection + SGS force + Coriolis + a constant body force.

    ``f_cor``≠0 drives a geostrophic/Ekman balance toward ``u_geo=(ug,vg)``; a
    constant ``force=(fx,fy)`` drives a pressure-gradient channel (``fx=u_*²/Lz``
    gives a target ``u_*`` and a log-law equilibrium in a few eddy turnovers —
    the clean Monin–Obukhov validation case).

    Surface scalar BC: when ``t_sfc`` is None the prescribed kinematic heat flux
    ``sfc_theta_flux`` is used with the NEUTRAL drag law (default path). When
    ``t_sfc`` (a prescribed/cooled surface temperature) is supplied AND ``theta``
    is active, the surface heat flux AND a stability-corrected drag are derived
    from the coupled stable MOST surface layer (:func:`most_surface_flux`) — the
    GABLS1-faithful prescribed-cooling boundary condition."""
    Cu, Cv, Cw = advection(u, v, w, g)
    nu_t = eddy_viscosity(u, v, w, g)
    if g.cfg.sgs_buoyancy and theta is not None:
        # Suppress the strain-based ν_t across stable stratification (Lilly 1962);
        # the SAME factor damps momentum (sgs_and_wall) and scalar (K_h=ν_t/Pr)
        # mixing, so the cloud-top inversion stops over-entraining. Neutral runs
        # leave this off ⇒ byte-identical.
        #   f·(strain + nu_floor) + (1−f)·nu_floor = f·strain + nu_floor
        # keeps the background floor UNsuppressed — a molecular-like minimum that
        # must survive f→0, else the inviscid inversion grows the 2Δ KH/gravity-
        # wave mode nu_floor exists to damp. No-op when nu_floor=0 (the default).
        f_buoy = sgs_buoyancy_factor(theta, tracers, u, v, w, g)
        nu_t = f_buoy * nu_t + (1.0 - f_buoy) * g.cfg.nu_floor
    cd_surf, sfc_flux = None, sfc_theta_flux
    if t_sfc is not None and theta is not None:
        u1, v1 = u[..., 0], v[..., 0]
        spd_mean = _planar_mean(jnp.sqrt(u1 ** 2 + v1 ** 2 + 1e-12), g)
        th1_mean = _planar_mean(theta[..., 0], g)
        _us, _ths, sfc_flux, cd_surf = most_surface_flux(
            spd_mean, th1_mean, t_sfc, g.z_c[0], g.cfg.z0, g.cfg.theta_ref0)
    Fu, Fv, Fw, u_star = sgs_and_wall(u, v, w, nu_t, g, u_geo, cd_surf=cd_surf)
    ug, vg = u_geo
    Ru = Cu + Fu + f_cor * (v - vg) + force[0]
    Rv = Cv + Fv - f_cor * (u - ug) + force[1]
    Rw = Cw + Fw
    # OPT-IN momentum-side dissipation (default off): the lever that lets the
    # less-diffusive WENO5 scalars run stable on this near-inviscid core without
    # diffusing the scalars themselves.
    if g.cfg.w_hyperdiff_coeff > 0.0:
        Rw = Rw + _w_hyperdiffusion(w, g)
    if g.cfg.div_damping_coeff > 0.0:
        dRu, dRv, dRw = _divergence_damping(u, v, w, g)
        Ru = Ru + dRu; Rv = Rv + dRv; Rw = Rw + dRw
    scalar_fn = scalar_rhs_monotone if g.cfg.monotone_scalars else scalar_rhs
    Rtheta = None
    if theta is not None:
        Rtheta = scalar_fn(theta, u, v, w, nu_t, g, sfc_flux)
        if g.cfg.theta_hyperdiff_coeff > 0.0:
            # SCALE-SELECTIVE k⁴ damping of the 2Δ θ' noise the sharp latent
            # heating injects — the instability SOURCE. Not broad diffusion: it
            # leaves the resolved θ + the WENO5 moisture sharp.
            Rtheta = Rtheta + _horizontal_hyperdiff(
                theta, g.cfg.theta_hyperdiff_coeff, g)
        if g.cfg.buoyancy:
            if g.cfg.moist:
                # moist=True with no tracers would silently fall back to DRY
                # buoyancy — a mis-assembled moist case must fail loudly
                # (codex 2026-06-11 #2).
                if tracers is None or tracers.shape[-1] < 1:
                    raise ValueError(
                        "cfg.moist=True needs a state with >=1 water tracer "
                        "(slot 0 = q_v) for the theta_v buoyancy; got "
                        "tracers=None/empty.")
                Rw = Rw + buoyancy_w_moist(theta, tracers, g)
            else:
                Rw = Rw + buoyancy_w(theta, g)
    Rtracers = None
    if tracers is not None and tracers.shape[-1] == 0:
        # Degenerate (…,0) array: no transport, but keep the pytree leaf shape
        # (codex 2026-06-11 #3 — jnp.stack on an empty list would crash).
        Rtracers = jnp.zeros_like(tracers)
    elif tracers is not None:
        # Each water tracer is advected + SGS-diffused exactly like θ (the same
        # scalar operator ⇒ same numerics, no re-derivation). Surface flux: the
        # prescribed kinematic moisture flux enters slot 0 (q_v); all other
        # slots have zero surface flux. Static Python loop over the (small,
        # compile-time-constant) slot count — unrolled at trace time.
        cols = []
        for k in range(tracers.shape[-1]):
            flx = sfc_qv_flux if k == 0 else 0.0
            cols.append(scalar_fn(tracers[..., k], u, v, w, nu_t, g, flx))
        Rtracers = jnp.stack(cols, axis=-1)
    Rw = Rw.at[..., 0].set(0.0).at[..., -1].set(0.0)
    return Ru, Rv, Rw, u_star, Rtheta, Rtracers


def _surface_ustar(u, v, g: SpectralLESGrid):
    """Surface friction velocity u_* = √Cd · ⟨|u₁|⟩ (the neutral MOST wall model,
    same formula as :func:`sgs_and_wall`) — a cheap diagnostic recomputed from the
    lowest-level (u, v) after a step (the shared RK integrator does not thread it
    through)."""
    kappa = constants.kappa_von_karman
    u1, v1 = u[..., 0], v[..., 0]
    Cd = (kappa / jnp.log(g.z_c[0] / g.cfg.z0)) ** 2
    return (Cd ** 0.5) * _planar_mean(jnp.sqrt(u1 ** 2 + v1 ** 2 + 1e-12), g)


def _filt_state(u, v, w, th, tr, g):
    """Apply the high-k cutoff to a state (no-op if disabled). Divergence-free
    preserving (mask uniform in z); re-zeros the w walls."""
    if not g.cfg.spectral_filter:
        return u, v, w, th, tr
    u = _apply_filter(u, g)
    v = _apply_filter(v, g)
    w = _apply_filter(w, g).at[..., 0].set(0.0).at[..., -1].set(0.0)
    # When scalars are intentionally transported by the monotone FV path,
    # do not follow that update with a non-monotone spectral cutoff: the
    # cutoff rings at q_v/q_c edges, creates negatives, and hands the
    # microphysics/positivity fixer spurious moisture anomalies to amplify.
    filter_tracers = (
        (not g.cfg.monotone_scalars) or g.cfg.filter_monotone_scalars)
    filter_theta = (
        (not g.cfg.monotone_scalars) or g.cfg.filter_monotone_scalars
        or g.cfg.filter_monotone_theta)
    th = None if th is None else (_apply_filter(th, g) if filter_theta else th)

    if tr is not None and tr.shape[-1] > 0 and (
        filter_tracers or (g.cfg.monotone_scalars and g.cfg.filter_monotone_qv)
    ):
        # _apply_filter contracts over the horizontal axes; map it over the
        # trailing tracer axis (static unroll, small slot count).
        cols = []
        for k in range(tr.shape[-1]):
            do_filter = filter_tracers or (k == 0 and g.cfg.filter_monotone_qv)
            col = _apply_filter(tr[..., k], g) if do_filter else tr[..., k]
            cols.append(col)
        tr = jnp.stack(cols, axis=-1)
    return u, v, w, th, tr


def step(state: SpectralLESState, g: SpectralLESGrid, dt,
         u_geo, f_cor: float, first: bool = False, force=(0.0, 0.0),
         sfc_theta_flux=0.0, t_sfc=None, sfc_qv_flux=0.0):
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
    tr = state.tracers
    if g.cfg.time_scheme == "ab2":
        Ru, Rv, Rw, u_star, Rth, Rtr = rhs(u, v, w, g, u_geo, f_cor, force=force,
                                           theta=th, sfc_theta_flux=sfc_theta_flux,
                                           t_sfc=t_sfc, tracers=tr,
                                           sfc_qv_flux=sfc_qv_flux)
        if first:
            au, av, aw, ath, atr = Ru, Rv, Rw, Rth, Rtr
        else:
            au = 1.5 * Ru - 0.5 * state.rhs_u_prev
            av = 1.5 * Rv - 0.5 * state.rhs_v_prev
            aw = 1.5 * Rw - 0.5 * state.rhs_w_prev
            ath = None if Rth is None else 1.5 * Rth - 0.5 * state.rhs_theta_prev
            atr = None if Rtr is None else 1.5 * Rtr - 0.5 * state.rhs_tracers_prev
        w_s = (w + dt * aw).at[..., 0].set(0.0).at[..., -1].set(0.0)
        u_n, v_n, w_n = project(u + dt * au, v + dt * av, w_s, dt, g)
        th_n = None if th is None else th + dt * ath
        tr_n = None if tr is None else tr + dt * atr
        u_n, v_n, w_n, th_n, tr_n = _filt_state(u_n, v_n, w_n, th_n, tr_n, g)
        return SpectralLESState(u=u_n, v=v_n, w=w_n, theta=th_n, tracers=tr_n,
                                rhs_theta_prev=Rth, rhs_tracers_prev=Rtr,
                                rhs_u_prev=Ru,
                                rhs_v_prev=Rv, rhs_w_prev=Rw), u_star

    # --- SSP-RK via the shared integrator -------------------------------------
    # slow_fn returns the RHS packed as a state pytree (history fields zeroed so the
    # integrator's axpy/linear-combination leave them at 0); the per-stage "fast"
    # update is the incompressible pressure projection (no acoustic substep here).
    def slow_fn(s):
        Ru, Rv, Rw, _u, Rth, Rtr = rhs(s.u, s.v, s.w, g, u_geo, f_cor,
                                       force=force, theta=s.theta,
                                       sfc_theta_flux=sfc_theta_flux,
                                       t_sfc=t_sfc, tracers=s.tracers,
                                       sfc_qv_flux=sfc_qv_flux)
        # History leaves zeroed from the INPUT state's structure (not the RHS):
        # if the caller passed rhs_*_prev=None with an active field, mirroring
        # Rth/Rtr here would change the pytree structure mid-integration and
        # jax.tree.map would reject the mismatch (codex 2026-06-11 #1).
        return SpectralLESState(
            u=Ru, v=Rv, w=Rw, theta=Rth, tracers=Rtr,
            rhs_u_prev=jnp.zeros_like(Ru), rhs_v_prev=jnp.zeros_like(Rv),
            rhs_w_prev=jnp.zeros_like(Rw),
            rhs_theta_prev=(None if s.rhs_theta_prev is None
                            else jnp.zeros_like(s.rhs_theta_prev)),
            rhs_tracers_prev=(None if s.rhs_tracers_prev is None
                              else jnp.zeros_like(s.rhs_tracers_prev)))

    def proj_fn(s_slow, slow_tend, dt_sub, n_sub, cfg):    # acoustic_update_fn slot
        wz = s_slow.w.at[..., 0].set(0.0).at[..., -1].set(0.0)
        un, vn, wn = project(s_slow.u, s_slow.v, wz, dt_sub, g)
        return s_slow._replace(u=un, v=vn, w=wn)           # θ untouched by projection

    se_cfg = SplitExplicitConfig(n_substeps=1, outer_integrator=g.cfg.time_scheme)
    out = split_explicit_step(state, slow_fn, proj_fn, dt, se_cfg)
    u_n, v_n, w_n, th_n, tr_n = _filt_state(out.u, out.v, out.w, out.theta,
                                            out.tracers, g)
    # u_* diagnostic from the RETURNED (filtered) state so it matches what the
    # caller sees (the filter is a horizontal low-pass; the surface-layer effect is
    # tiny, but keep them consistent).
    u_star = _surface_ustar(u_n, v_n, g)
    return SpectralLESState(
        u=u_n, v=v_n, w=w_n, theta=th_n, tracers=tr_n,
        rhs_u_prev=jnp.zeros_like(u_n), rhs_v_prev=jnp.zeros_like(v_n),
        rhs_w_prev=jnp.zeros_like(w_n),
        rhs_theta_prev=None if th_n is None else jnp.zeros_like(th_n),
        rhs_tracers_prev=None if tr_n is None else jnp.zeros_like(tr_n)), u_star
