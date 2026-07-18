"""Pseudo-incompressible boundary-layer / deep-convection LES on the doubly-periodic
plane — the GPU/TPU mesh-scalable plane dycore.

Faithful in algorithm to the **LEX** model (MetLab-HKUST/LEX, GMD 2026,
``solver_opt=1``): the anelastic-type constraint ``∇·(ρ0θ0 u) = 0`` is enforced each
RK stage by a fractional-step pressure PROJECTION whose elliptic solve is the
variable-coefficient, **matrix-free Jacobi-preconditioned BiCGSTAB** Poisson in
:mod:`pseudo_incompressible_poisson` (a nearest-neighbour 7-point stencil — one halo
exchange per matvec, NOT the spectral core's global all-to-all FFT). Advection is the
shared flux-form FD transport (:mod:`plane_fd_advection`, WENO5/van-Leer/upwind on the
public ``core.weno``/``core.flux_limiters`` kernels). Time stepping is SSP-RK3 with the
projection applied per stage.

Why this core (vs the two existing plane dycores) — see ``docs/physics-notes/pseudo_incompressible_les.md``.
The spectral core is exact but FFT ⇒ all-to-all (poor mesh scaling, periodic-only); the
compressible core scales (FD + halo) but acoustic CFL + hyperdiff cap the effective Re.
This core has no acoustic mode and only nearest-neighbour elliptic comms.

Grid / staggering — full Arakawa C-grid
---------------------------------------
Physical ``(ny, nx, nz)``. Scalars (θ, tracers) at cell CENTRES. ``u`` at x-faces
(``u[..,i]=u_{i+½}``), ``v`` at y-faces (``v[..,j]=v_{j+½}``) — periodic, same array
shape ``(ny,nx,nz)``. ``w`` at z-faces ``(ny,nx,nz+1)`` with rigid walls
``w[..,0]=w[..,nz]=0``. Uniform ``Δz``. The C-grid makes the discrete divergence,
pressure gradient and the compact Poisson Laplacian an EXACT ``D·G=L`` triple, so the
projection drives the ρ-weighted divergence to machine zero (no collocated checkerboard;
better-conditioned ⇒ fewer BiCGSTAB iterations ⇒ fewer halo exchanges on a device mesh).
Hydrostatic base state ``(θ0, π0, ρ0)`` per LEX (``dπ0/dz=−g/(Cp θ0)``,
``ρ0θ0=π0^{Cv/Rd} p00/Rd``).

Moisture / lego
---------------
``cfg.moist`` switches the buoyancy to the density potential temperature
``θ_ρ=θ·(1+reps·q_v)/(1+q_v)`` using tracer slot 0 (``q_v``). Water tracers (standard
microphysics slot layout ``[0]=q_v,[1]=q_c,…``) are transported by the same shared scalar
operator, so any microphysics scheme swaps in unchanged via the existing factory (latent
heating enters ``θ`` through the physics hook — wired with the moist cases). Dry runs
(``n_tracers=0``) are unchanged.

Pure-pytree, JIT/``jax.grad``-safe. MPI: swap the ``jnp.roll`` halos in
:mod:`plane_fd_advection` / :mod:`pseudo_incompressible_poisson` for
``parallel.halo_exchange`` strips (follow-up; the stencils are unchanged).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.dynamics.les import plane_fd_advection as _adv
from legoesm.atmosphere.dynamics.les import pseudo_incompressible_poisson as _poisson
from legoesm.atmosphere.dynamics.les.spectral_les_plane import most_surface_flux
from legoesm.atmosphere.physics.turbulence.vreman import vreman_nu_t
from legoesm.atmosphere.physics.turbulence.lasd_core import lasd_cs2

_AY, _AX, _AZ = 0, 1, 2
_REPS = constants.R_v / constants.R_d            # 1/epsilon ≈ 1.608
_SGS = ("none", "smagorinsky", "vreman", "lasd")
_SURFACE = ("free", "flux", "most_cooling")


class PseudoIncompressibleConfig(NamedTuple):
    nx: int
    ny: int
    nz: int
    Lx: float
    Ly: float
    Lz: float
    theta_ref0: float = 300.0        # base-state (isentropic) potential temperature [K]
    scheme: str = "weno5"            # SCALAR advection: "weno5" | "van_leer" | "upwind"
    momentum_scheme: str | None = None  # MOMENTUM advection: as `scheme`, plus "weno7"/
    #                                  "weno9" (less upwind diffusion) and "central" (non-
    #                                  dissipative, for sharp LES — WENO5's upwind k⁴-k⁶
    #                                  diffusion over-smooths the resolved eddies vs the
    #                                  spectral core; "central" needs a 2Δ de-noiser, see
    #                                  momentum_shapiro_coeff). None ⇒ FOLLOW `scheme`
    #                                  (bit-identical to before this field existed).
    moist: bool = False              # θ_ρ buoyancy from tracer slot 0 (q_v)
    n_tracers: int = 0               # water-tracer slots (0 ⇒ dry)
    poisson_tol: float = 1e-6
    poisson_atol: float = 1e-10
    poisson_maxiter: int = 200
    # --- boundary-layer physics (composable / lego) ---
    f_cor: float = 0.0               # Coriolis parameter f [1/s]
    ug: float = 0.0                  # geostrophic wind u_g [m/s]
    vg: float = 0.0                  # geostrophic wind v_g [m/s]
    sgs: str = "none"                # SGS closure: "none"|"smagorinsky"|"vreman"|"lasd"
    c_s: float = 0.16                # Smagorinsky constant
    c_vreman: float = 0.07           # Vreman constant (≈2.5 C_s²)
    cs_max: float = 1.0              # upper clip on the LASD dynamic C_s² (oracle mask)
    pr_sgs: float = 1.0              # SGS Prandtl number (K_h = ν_t/Pr)
    nu_floor: float = 0.0            # background eddy-viscosity floor [m²/s] — keeps a
    #                                  strongly-stable layer (where the SGS shuts off,
    #                                  ν_t→0) from going inviscid and growing the 2Δ
    #                                  grid-scale mode (SBL collapse). 0 ⇒ off.
    hyperdiff_coeff: float = 0.0     # OPT-IN horizontal biharmonic de-noiser [m⁴/s]:
    #                                  tendency −= coeff·∇⁴_h(f) on u,v,w,θ,tracers. The
    #                                  FD analogue of the spectral core's sharp k-cutoff
    #                                  — scale-selective (hits the 2Δ grid mode hard,
    #                                  resolved eddies lightly), for fine-res stable-BL /
    #                                  moist robustness. 0 ⇒ off (the core's default
    #                                  "no numerical hyperdiffusion, sustain turbulence").
    shapiro_coeff: float = 0.0       # OPT-IN per-step [1,2,1] horizontal low-pass on
    #                                  θ (+tracers), blend strength s∈[0,1]. A MULTIPLICATIVE
    #                                  filter (response cos²(kΔ/2)∈[0,1]) ⇒ NOT CFL-limited,
    #                                  unlike hyperdiff_coeff — de-noises the fine-res
    #                                  stable-BL 2Δ θ-mode (which biharmonic can't reach in
    #                                  f32 within its stability window). Scalars only ⇒ the
    #                                  projection is untouched (velocity stays div-free).
    #                                  0 ⇒ off (bit-identical). s≈0.05–0.5 typical.
    shapiro_order: int = 1           # Shapiro ORDER for the scalar θ/tracer de-noiser
    #                                  (as momentum_shapiro_order): order 1 = [1,2,1]; a HIGH
    #                                  order (8–16) is flat in the passband so a stronger
    #                                  coeff suppresses the fine-res stable-BL 2Δ θ-mode
    #                                  (needed for f32 at fine resolution) without eroding
    #                                  the resolved θ eddies.
    momentum_shapiro_coeff: float = 0.0  # OPT-IN per-step [1,2,1] horizontal low-pass on
    #                                  VELOCITY (u,v,w), blend s∈[0,1] — the CFL-unlimited
    #                                  2Δ de-noiser that makes NON-dissipative momentum
    #                                  (momentum_scheme="central") clean, mirroring the
    #                                  spectral core's sharp-filter pairing. The horizontal
    #                                  [1,2,1] convolution COMMUTES with the C-grid divergence
    #                                  (periodic x,y), so div(H·u)=H(div u)=0 ⇒ divergence-free
    #                                  is PRESERVED with NO re-projection. 0 ⇒ off (bit-identical).
    momentum_shapiro_order: int = 1  # Shapiro ORDER for the velocity de-noiser (response
    #                                  1−sin^(2·order)). order=1 ([1,2,1]) over-damps the
    #                                  resolved eddies under per-step use (its passband
    #                                  response <1 compounds); a HIGH order (8–16) is FLAT in
    #                                  the passband so only the 2Δ mode is removed. Use 8–16
    #                                  with momentum_shapiro_coeff for clean central momentum.
    surface: str = "free"            # surface BC: "free"|"flux"|"most_cooling"
    z0: float = 0.1                  # roughness length z0=z0h [m]
    sfc_theta_flux: float = 0.0      # prescribed kinematic heat flux ⟨w'θ'⟩₀ [K m/s]
    sfc_qv_flux: float = 0.0         # prescribed kinematic moisture flux ⟨w'q_v'⟩₀ [m/s]
    #                                  into tracer slot 0 (BOMEX LHF). 0 ⇒ off.


class PseudoIncompressibleForcing(NamedTuple):
    """Explicit (non-closure) large-scale forcing passed to :func:`step`.

    Passed as an argument (SegmentForcing doctrine: a closure would bake the
    values in at compile time and recompile when they change). All fields are
    optional; ``None`` ⇒ that forcing is off (static, Python-side ``if``)."""
    t_sfc: jax.Array | None = None       # surface temperature [K] (most_cooling)
    subsidence_w: jax.Array | None = None  # large-scale w_ls(z) (nz,) [m/s] — applied
    #                                        to θ AND q_v (tracer slot 0)
    dtheta_dt_ls: jax.Array | None = None  # large-scale dθ/dt (nz,) [K/s]
    dqv_dt_ls: jax.Array | None = None     # large-scale dq_v/dt (nz,) [1/s] (slot 0)
    ug_prof: jax.Array | None = None       # height-dependent geostrophic u_g(z) (nz,)
    vg_prof: jax.Array | None = None       # height-dependent geostrophic v_g(z) (nz,)
    #                                        override the scalar cfg.ug/vg (BOMEX shear)


class PseudoIncompressibleGrid(NamedTuple):
    cfg: PseudoIncompressibleConfig
    dx: float
    dy: float
    dz: float
    z_c: jax.Array                   # (nz,)
    z_f: jax.Array                   # (nz+1,)
    theta0: jax.Array                # (nz,) base-state θ
    rho0_theta0: jax.Array           # (nz,) base-state ρ0·θ0 (anelastic mass weight)


class PseudoIncompressibleState(NamedTuple):
    u: jax.Array                     # (ny, nx, nz)   x-faces (u[..,i]=u_{i+½}), C-grid
    v: jax.Array                     # (ny, nx, nz)   y-faces (v[..,j]=v_{j+½}), C-grid
    w: jax.Array                     # (ny, nx, nz+1) z-faces, walls 0
    theta: jax.Array                 # (ny, nx, nz)   potential temperature [K]
    pi_prev: jax.Array               # (ny, nx, nz)   previous π' (BiCGSTAB warm start)
    tracers: jax.Array | None = None  # (ny, nx, nz, n_tracers)


def make_grid(cfg: PseudoIncompressibleConfig, dtype=jnp.float64
              ) -> PseudoIncompressibleGrid:
    if cfg.scheme not in _adv._SCHEMES:
        raise ValueError(f"scheme must be one of {_adv._SCHEMES}, got {cfg.scheme!r}.")
    if cfg.momentum_scheme is not None and cfg.momentum_scheme not in _adv._SCHEMES:
        raise ValueError(f"momentum_scheme must be None or one of {_adv._SCHEMES}, "
                         f"got {cfg.momentum_scheme!r}.")
    if cfg.moist and cfg.n_tracers < 1:
        raise ValueError("cfg.moist=True requires n_tracers>=1 (slot 0 = q_v).")
    if cfg.sgs not in _SGS:
        raise ValueError(f"sgs must be one of {_SGS}, got {cfg.sgs!r}.")
    if cfg.surface not in _SURFACE:
        raise ValueError(f"surface must be one of {_SURFACE}, got {cfg.surface!r}.")
    if not 0.0 <= cfg.shapiro_coeff <= 1.0:
        # outside [0,1] the [1,2,1] blend stops being a convex low-pass (s>2 even
        # amplifies the 2Δ mode and breaks tracer positivity).
        raise ValueError(f"shapiro_coeff must be in [0,1], got {cfg.shapiro_coeff}.")
    if not 0.0 <= cfg.momentum_shapiro_coeff <= 1.0:
        raise ValueError(f"momentum_shapiro_coeff must be in [0,1], "
                         f"got {cfg.momentum_shapiro_coeff}.")
    if not isinstance(cfg.momentum_shapiro_order, int) or cfg.momentum_shapiro_order < 1:
        raise ValueError(f"momentum_shapiro_order must be an int >= 1, "
                         f"got {cfg.momentum_shapiro_order!r}.")
    if not isinstance(cfg.shapiro_order, int) or cfg.shapiro_order < 1:
        raise ValueError(f"shapiro_order must be an int >= 1, got {cfg.shapiro_order!r}.")
    nz = cfg.nz
    dx, dy, dz = cfg.Lx / cfg.nx, cfg.Ly / cfg.ny, cfg.Lz / nz
    z_c = (jnp.arange(nz, dtype=dtype) + 0.5) * dz
    z_f = jnp.arange(nz + 1, dtype=dtype) * dz
    # Hydrostatic base state (LEX setup_lex): isentropic θ0; dπ0/dz=−g/(Cp θ0).
    theta0 = jnp.full((nz,), cfg.theta_ref0, dtype=dtype)
    cp, rd, g, p00 = constants.c_pd, constants.R_d, constants.g, constants.p_ref
    cv = cp - rd
    # π0 at centres: integrate from a surface value π0(0)=1 − (g/(Cp θ0))·z_c (θ0 const).
    pi0 = 1.0 - (g / (cp * cfg.theta_ref0)) * z_c
    rho0_theta0 = pi0 ** (cv / rd) * p00 / rd
    return PseudoIncompressibleGrid(cfg=cfg, dx=dx, dy=dy, dz=dz, z_c=z_c, z_f=z_f,
                                    theta0=theta0, rho0_theta0=rho0_theta0)


# --------------------------------------------------------------------------- #
# Buoyancy + variable Poisson coefficient                                     #
# --------------------------------------------------------------------------- #
def _theta_rho(theta, tracers, cfg):
    """Density potential temperature θ_ρ = θ·(1+reps·q_v)/(1+q_v) (θ when dry)."""
    if cfg.moist and tracers is not None:
        qv = tracers[..., 0]
        return theta * (1.0 + _REPS * qv) / (1.0 + qv)
    return theta


def buoyancy_faces(theta, tracers, g: PseudoIncompressibleGrid):
    """Buoyancy ``b=g·(θ_ρ−θ_ρ0)/θ_ρ0`` interpolated to w-faces (walls 0)."""
    th_rho = _theta_rho(theta, tracers, g.cfg)
    th_rho0 = g.theta0[None, None, :]                 # dry base θ_ρ0 = θ0
    b_c = constants.g * (th_rho - th_rho0) / th_rho0   # centres (ny,nx,nz)
    b_f = _adv._centre_to_face_z(b_c)                  # (ny,nx,nz+1)
    return b_f.at[..., 0].set(0.0).at[..., -1].set(0.0)


def _rtt(theta, tracers, g: PseudoIncompressibleGrid):
    """Variable Poisson coefficient C = ρ0θ0·θ_ρ (the moist density weighting)."""
    return g.rho0_theta0[None, None, :] * _theta_rho(theta, tracers, g.cfg)


# --------------------------------------------------------------------------- #
# Pressure projection (fractional step)                                       #
# --------------------------------------------------------------------------- #
def _rho_weighted_divergence(u, v, w, g: PseudoIncompressibleGrid):
    """``∇·(ρ0θ0 u)`` at centres, C-grid COMPACT (consistent with the compact Poisson
    Laplacian ⇒ exact projection). ``u`` at x-faces, ``v`` at y-faces, ``w`` at z-faces;
    ρ0θ0=ρ0θ0(z) factors out horizontally and is face-interpolated for the vertical."""
    rt = g.rho0_theta0[None, None, :]
    du = (u - jnp.roll(u, 1, axis=_AX)) / g.dx            # (u_{i+½}−u_{i−½})/Δx
    dv = (v - jnp.roll(v, 1, axis=_AY)) / g.dy
    rt_f = _adv._centre_to_face_z(g.rho0_theta0[None, None, :])  # (1,1,nz+1)
    flux_w = rt_f * w
    dw = (flux_w[..., 1:] - flux_w[..., :-1]) / g.dz
    return rt * (du + dv) + dw


def precision_floored_poisson_tols(cfg, dtype):
    """Precision-aware BiCGSTAB tolerances ``(tol, atol)``.

    In float32 an ``atol=1e-10`` target is BELOW the achievable
    ~O(eps≈1.2e-7) residual, so the solver iterates past convergence and
    its ρ/ω recurrences underflow → breakdown → NaN (fatal for the
    non-dissipative ``central`` momentum, which has no numerical
    dissipation to damp the residual-driven divergence). Floor tol/atol
    at ~O(eps) so f32 stops before breakdown; the f64 floors (~1e-13)
    sit below the tight defaults ⇒ f64 is unchanged (bit-identical).
    Shared by the serial projection and the MPI ``project_mpi`` so both
    precisions behave identically across the decomposition.
    """
    eps = float(jnp.finfo(dtype).eps)
    return (max(cfg.poisson_tol, 8.0e2 * eps),
            max(cfg.poisson_atol, 8.0e1 * eps))


def project(u, v, w, theta, tracers, pi_prev, dt, g: PseudoIncompressibleGrid):
    """EXACT C-grid projection: ``∇·(ρ0θ0 u)=0`` via ``Cp ∇·(rtt ∇π') = ∇·(ρ0θ0 u*)/dt``
    then the C-grid pressure-gradient correction. Returns (u,v,w,π').

    The correction is built so ``∇·(ρ0θ0 u_corr) ≡ Cp ∇·(rtt ∇π')`` (the compact Poisson
    Laplacian) face-by-face, i.e. an exact ``D·G=L`` pair — so the post-correction
    ρ-weighted divergence is machine zero (to the BiCGSTAB tolerance). Faces: ρ0θ0·u_corr
    at the x-face = ``dt·Cp·rtt_{i+½}·∂π'/∂x``; since ρ0θ0 is z-only this is
    ``u_corr = dt·Cp·θ_ρ_{i+½}·∂π'/∂x`` (θ_ρ averaged to the face). The vertical
    coefficient ``Cp·rtt_{k+½}/ρ0θ0_{k+½}`` keeps the match under the z-variation of ρ0θ0.
    """
    cfg = g.cfg
    cp = constants.c_pd
    c = _rtt(theta, tracers, g)                           # rtt at centres
    rhs = _rho_weighted_divergence(u, v, w, g) / dt
    tol, atol = precision_floored_poisson_tols(cfg, rhs.dtype)
    pi, _info = _poisson.solve_pressure(
        rhs, c, g.dx, g.dy, g.dz, x0=pi_prev,
        tol=tol, atol=atol, maxiter=cfg.poisson_maxiter)
    th_rho = _theta_rho(theta, tracers, cfg)
    # horizontal: gradient + θ_ρ both at the i+½ / j+½ faces (C-grid).
    thr_xf = 0.5 * (th_rho + jnp.roll(th_rho, -1, axis=_AX))
    thr_yf = 0.5 * (th_rho + jnp.roll(th_rho, -1, axis=_AY))
    dpi_dx_f = (jnp.roll(pi, -1, axis=_AX) - pi) / g.dx
    dpi_dy_f = (jnp.roll(pi, -1, axis=_AY) - pi) / g.dy
    u_new = u - dt * cp * thr_xf * dpi_dx_f
    v_new = v - dt * cp * thr_yf * dpi_dy_f
    # vertical interior faces: coefficient Cp·rtt_{k+½}/ρ0θ0_{k+½} (exact-match weight).
    rtt_zf = 0.5 * (c[..., :-1] + c[..., 1:])                       # (ny,nx,nz-1)
    rho0t0 = g.rho0_theta0[None, None, :]
    rho0t0_zf = 0.5 * (rho0t0[..., :-1] + rho0t0[..., 1:])          # (1,1,nz-1)
    coef_w = cp * rtt_zf / rho0t0_zf
    dpi_dz_int = (pi[..., 1:] - pi[..., :-1]) / g.dz                # (ny,nx,nz-1)
    w_corr = jnp.pad(coef_w * dpi_dz_int, [(0, 0), (0, 0), (1, 1)])  # 0 at walls
    w_new = w - dt * w_corr
    w_new = w_new.at[..., 0].set(0.0).at[..., -1].set(0.0)
    return u_new, v_new, w_new, pi


# --------------------------------------------------------------------------- #
# Composable SGS closure + surface flux + large-scale forcing (lego)          #
# --------------------------------------------------------------------------- #
def _ddz_c(f, dz):
    """∂f/∂z for a CENTRE field ``(…,nz)``: centred interior, one-sided at walls."""
    inner = (f[..., 2:] - f[..., :-2]) / (2.0 * dz)
    bot = ((f[..., 1] - f[..., 0]) / dz)[..., None]
    top = ((f[..., -1] - f[..., -2]) / dz)[..., None]
    return jnp.concatenate([bot, inner, top], axis=_AZ)


def _ddx_c(f, dx):
    return (jnp.roll(f, -1, axis=_AX) - jnp.roll(f, 1, axis=_AX)) / (2.0 * dx)


def _ddy_c(f, dy):
    return (jnp.roll(f, -1, axis=_AY) - jnp.roll(f, 1, axis=_AY)) / (2.0 * dy)


def _hlap(f, g: PseudoIncompressibleGrid):
    """Horizontal 5-point Laplacian ∇²_h f on periodic x,y (roll). Vertical untouched,
    so it applies unchanged to centre fields (…,nz) and z-face w (…,nz+1)."""
    d2x = (jnp.roll(f, -1, axis=_AX) - 2.0 * f + jnp.roll(f, 1, axis=_AX)) / g.dx ** 2
    d2y = (jnp.roll(f, -1, axis=_AY) - 2.0 * f + jnp.roll(f, 1, axis=_AY)) / g.dy ** 2
    return d2x + d2y


def _hyperdiff(f, g: PseudoIncompressibleGrid):
    """Horizontal biharmonic de-noiser tendency −coeff·∇⁴_h f (FD analogue of the
    spectral core's sharp cutoff). coeff=0 ⇒ exactly zero (default path bit-identical)."""
    return -g.cfg.hyperdiff_coeff * _hlap(_hlap(f, g), g)


def _shapiro_h(f, order=1):
    """Separable order-2·``order`` Shapiro horizontal low-pass on periodic x,y (roll).

    Per-axis response ``1 − sin^(2·order)(kΔ/2) ∈ [0,1]`` — zero at the 2Δ Nyquist
    mode, ``1`` at ``k→0``. ``order=1`` is the classic [1,2,1]/4 (cos²); HIGHER order
    ⇒ a FLATTER passband (resolved scales barely eroded under repeated per-step use,
    which a single [1,2,1] would over-damp) with the same sharp 2Δ removal. A bounded
    filter (not a diffusion tendency) ⇒ NOT CFL-limited. It is a horizontal convolution
    (polynomial in x/y rolls), so it commutes with the C-grid divergence (div-free-
    preserving on velocity). Vertical untouched (centre (…,nz) and z-face (…,nz+1) alike)."""
    coeff = (-1.0) ** order / 4.0 ** order          # response 1 − sin^(2·order)
    out = f
    for axis in (_AX, _AY):
        d = out
        for _ in range(order):                      # (∂²)^order via the [1,−2,1] stencil
            d = jnp.roll(d, 1, axis=axis) - 2.0 * d + jnp.roll(d, -1, axis=axis)
        out = out - coeff * d
    return out


def _centre_velocities(u, v, w):
    """C-grid face velocities → cell centres (u,v from x/y faces, w from z faces)."""
    uc = 0.5 * (u + jnp.roll(u, 1, axis=_AX))
    vc = 0.5 * (v + jnp.roll(v, 1, axis=_AY))
    wc = _adv._face_to_centre_z(w)
    return uc, vc, wc


def _velocity_gradients(uc, vc, wc, g):
    """The nine resolved velocity gradients ``a_cd=∂u_c/∂x_d`` at cell CENTRES.

    SGS is evaluated at centres (the standard LES practice) on the centre-averaged
    velocities — a deliberate, disclosed simplification vs a corner-staggered C-grid
    stress; the momentum projection stays exact C-grid. The SGS term is a small
    dissipative correction so the centre evaluation is well within its own modelling error.
    """
    dx, dy, dz = g.dx, g.dy, g.dz
    return (_ddx_c(uc, dx), _ddy_c(uc, dy), _ddz_c(uc, dz),
            _ddx_c(vc, dx), _ddy_c(vc, dy), _ddz_c(vc, dz),
            _ddx_c(wc, dx), _ddy_c(wc, dy), _ddz_c(wc, dz))


def eddy_viscosity(uc, vc, wc, g: PseudoIncompressibleGrid):
    """Centred SGS eddy viscosity ν_t≥0 at cell centres. Composable via cfg.sgs:
    "none" (0), "smagorinsky" (Mason wall-damped |S|), "vreman" (shared core),
    "lasd" (Bou-Zeid 2005 scale-dependent dynamic C_s², shared ``lasd_core``)."""
    cfg = g.cfg
    if cfg.sgs == "none":
        return jnp.full_like(uc, cfg.nu_floor)   # shape from input (MPI-padded-safe)
    a11, a12, a13, a21, a22, a23, a31, a32, a33 = _velocity_gradients(uc, vc, wc, g)
    if cfg.sgs == "vreman":
        return vreman_nu_t(a11, a12, a13, a21, a22, a23, a31, a32, a33,
                           g.dx, g.dy, g.dz, cfg.c_vreman, nu_floor=cfg.nu_floor)
    # shared strain + |S| for the Smagorinsky family (smagorinsky, lasd)
    s11, s22, s33 = a11, a22, a33
    s12 = 0.5 * (a12 + a21); s13 = 0.5 * (a13 + a31); s23 = 0.5 * (a23 + a32)
    smod = jnp.sqrt(2.0 * (s11 ** 2 + s22 ** 2 + s33 ** 2
                           + 2.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)) + 1e-30)
    delta = (g.dx * g.dy * g.dz) ** (1.0 / 3.0)
    if cfg.sgs == "lasd":
        # Bou-Zeid scale-dependent dynamic C_s²(x,y,z); ν_t = C_s²·Δ²·|S|. Serial
        # only here (layout=None); the MPI path supplies the slab layout (follow-up).
        delta_z = jnp.full((cfg.nz,), delta, dtype=uc.dtype)
        cs2 = lasd_cs2(uc, vc, wc, s11, s22, s33, s12, s13, s23, smod,
                       delta_z, cs_max=cfg.cs_max, layout=None)
        return cs2 * delta ** 2 * smod + cfg.nu_floor
    # smagorinsky: ν_t = l_m²|S|; Mason l_m⁻²=(c_sΔ)⁻²+(κ(z+z0))⁻²
    kappa = constants.kappa_von_karman
    z = g.z_c[None, None, :]
    inv_l2 = 1.0 / (cfg.c_s * delta) ** 2 + 1.0 / (kappa * (z + cfg.z0)) ** 2
    return smod / inv_l2 + cfg.nu_floor


def _sgs_force(u, v, w, theta, g: PseudoIncompressibleGrid, sfc_flux, cd_surf,
              spd1_mean=None, nu_t=None):
    """SGS momentum force (faces) + θ diffusion (centres), with the SURFACE vertical
    flux replaced by the wall stress / prescribed heat flux into the first layer.

    Momentum force ``∂_j(2ν_t S_ij)`` and ``∂_j(K_h ∂_j θ)`` (K_h=ν_t/Pr) are formed at
    centres then the momentum components are averaged to the C-grid faces. The surface
    momentum sink ``−Cd⟨|u₁|⟩u₁`` and surface heat flux ``sfc_flux`` enter the first
    cell layer (flux convergence /Δz). Returns (Fu_face, Fv_face, Fw_face, Fθ, u_star)."""
    cfg = g.cfg
    dz = g.dz
    uc, vc, wc = _centre_velocities(u, v, w)
    if nu_t is None:
        nu_t = eddy_viscosity(uc, vc, wc, g)
    a11, a12, a13, a21, a22, a23, a31, a32, a33 = _velocity_gradients(uc, vc, wc, g)
    s12 = 0.5 * (a12 + a21); s13 = 0.5 * (a13 + a31); s23 = 0.5 * (a23 + a32)
    t11, t22, t33 = 2 * nu_t * a11, 2 * nu_t * a22, 2 * nu_t * a33
    t12, t13, t23 = 2 * nu_t * s12, 2 * nu_t * s13, 2 * nu_t * s23
    Fu_c = _ddx_c(t11, g.dx) + _ddy_c(t12, g.dy) + _ddz_c(t13, dz)
    Fv_c = _ddx_c(t12, g.dx) + _ddy_c(t22, g.dy) + _ddz_c(t23, dz)
    Fw_c = _ddx_c(t13, g.dx) + _ddy_c(t23, g.dy) + _ddz_c(t33, dz)
    # surface momentum drag into the first layer (Moeng wall model: planar-mean speed)
    u1, v1 = uc[..., 0], vc[..., 0]
    spd1 = jnp.sqrt(u1 ** 2 + v1 ** 2 + 1e-12)
    # PLANAR mean of the surface speed. Serial: local jnp.mean. MPI: the caller injects
    # the GLOBAL planar mean (computed from the unpadded interior) so the wall stress
    # matches the single-process value (drag ∝ ⟨|u₁|⟩ must span the whole plane).
    spd1_mean = jnp.mean(spd1) if spd1_mean is None else spd1_mean
    kappa = constants.kappa_von_karman
    Cd = ((kappa / jnp.log(g.z_c[0] / cfg.z0)) ** 2 if cd_surf is None else cd_surf)
    u_star = (Cd ** 0.5) * spd1_mean
    drag_u = jnp.zeros_like(Fu_c).at[..., 0].set(-Cd * spd1_mean * u1 / dz)
    drag_v = jnp.zeros_like(Fv_c).at[..., 0].set(-Cd * spd1_mean * v1 / dz)
    Fu_c = Fu_c + drag_u; Fv_c = Fv_c + drag_v
    # θ SGS diffusion + surface heat-flux convergence into the first layer
    kh = nu_t / cfg.pr_sgs
    Fth = (_ddx_c(kh * _ddx_c(theta, g.dx), g.dx)
           + _ddy_c(kh * _ddy_c(theta, g.dy), g.dy)
           + _ddz_c(kh * _ddz_c(theta, dz), dz))
    Fth = Fth.at[..., 0].add(sfc_flux / dz)
    # map momentum forces to the C-grid faces
    Fu = 0.5 * (Fu_c + jnp.roll(Fu_c, -1, axis=_AX))
    Fv = 0.5 * (Fv_c + jnp.roll(Fv_c, -1, axis=_AY))
    Fw = _adv._centre_to_face_z(Fw_c).at[..., 0].set(0.0).at[..., -1].set(0.0)
    return Fu, Fv, Fw, Fth, u_star


def _tracer_sgs(tracers, nu_t, g: PseudoIncompressibleGrid, sfc_qv_flux):
    """SGS diffusion ``∂_j(K_h ∂_j q)`` (K_h=ν_t/Pr) for each water tracer + the surface
    moisture flux ``sfc_qv_flux`` into slot 0 (q_v) first layer. Returns (ny,nx,nz,nt).

    Batched over the trailing tracer axis with ``jax.vmap`` (one fused kernel
    set for ALL slots, compile time independent of n_tracers) instead of a
    trace-time Python loop."""
    cfg = g.cfg; dz = g.dz
    kh = nu_t / cfg.pr_sgs

    def _sgs_one(q):
        return (_ddx_c(kh * _ddx_c(q, g.dx), g.dx)
                + _ddy_c(kh * _ddy_c(q, g.dy), g.dy)
                + _ddz_c(kh * _ddz_c(q, dz), dz))

    F = jax.vmap(_sgs_one, in_axes=-1, out_axes=-1)(tracers)
    # q_v (slot 0) gets the surface moisture flux into the first layer.
    return F.at[:, :, 0, 0].add(sfc_qv_flux / dz)


def _surface_state(u, v, theta, g: PseudoIncompressibleGrid, forcing, sfc_means=None):
    """Resolve the surface kinematic heat flux + drag coefficient for cfg.surface.
    ``sfc_means=(spd_mean, th1_mean)`` injects GLOBAL planar means (MPI); None ⇒ local."""
    cfg = g.cfg
    if cfg.surface == "free":
        return 0.0, None
    if cfg.surface == "flux":
        return cfg.sfc_theta_flux, None
    # most_cooling: coupled MOST (GABLS1 prescribed t_sfc), shared most_surface_flux
    if sfc_means is not None:
        spd_mean, th1_mean = sfc_means
    else:
        uc = 0.5 * (u + jnp.roll(u, 1, axis=_AX))
        vc = 0.5 * (v + jnp.roll(v, 1, axis=_AY))
        spd_mean = jnp.mean(jnp.sqrt(uc[..., 0] ** 2 + vc[..., 0] ** 2 + 1e-12))
        th1_mean = jnp.mean(theta[..., 0])
    _us, _ths, q0, cd_eff = most_surface_flux(
        spd_mean, th1_mean, forcing.t_sfc, g.z_c[0], cfg.z0, cfg.theta_ref0)
    return q0, cd_eff


def _large_scale_forcing(u, v, theta, tracers, g: PseudoIncompressibleGrid, forcing):
    """Coriolis/geostrophic momentum forcing + large-scale subsidence + dθ/dt_ls (θ) +
    subsidence + dq_v/dt_ls (q_v, slot 0). Returns (Fu_face, Fv_face, Fθ, Ftr|None)."""
    cfg = g.cfg
    Fu = jnp.zeros_like(u); Fv = jnp.zeros_like(v); Fth = jnp.zeros_like(theta)
    if cfg.f_cor != 0.0:
        uc = 0.5 * (u + jnp.roll(u, 1, axis=_AX))         # shapes from input (MPI-safe)
        vc = 0.5 * (v + jnp.roll(v, 1, axis=_AY))
        v_xf = 0.5 * (vc + jnp.roll(vc, -1, axis=_AX))    # v at x-face
        u_yf = 0.5 * (uc + jnp.roll(uc, -1, axis=_AY))    # u at y-face
        # height-dependent geostrophic wind (BOMEX shear) overrides the scalar default
        ug = cfg.ug if (forcing is None or forcing.ug_prof is None) else \
            forcing.ug_prof[None, None, :]
        vg = cfg.vg if (forcing is None or forcing.vg_prof is None) else \
            forcing.vg_prof[None, None, :]
        Fu = Fu + cfg.f_cor * (v_xf - vg)
        Fv = Fv - cfg.f_cor * (u_yf - ug)
    wls = None if forcing is None else forcing.subsidence_w
    if wls is not None:
        Fth = Fth - wls[None, None, :] * _ddz_c(theta, g.dz)
    if forcing is not None and forcing.dtheta_dt_ls is not None:
        Fth = Fth + forcing.dtheta_dt_ls[None, None, :]
    Ftr = None
    if tracers is not None and tracers.shape[-1] > 0:
        qv = tracers[..., 0]
        dqv = jnp.zeros_like(qv)
        if wls is not None:
            dqv = dqv - wls[None, None, :] * _ddz_c(qv, g.dz)
        if forcing is not None and forcing.dqv_dt_ls is not None:
            dqv = dqv + forcing.dqv_dt_ls[None, None, :]
        Ftr = jnp.zeros_like(tracers).at[..., 0].set(dqv)
    return Fu, Fv, Fth, Ftr


# --------------------------------------------------------------------------- #
# Tendencies (no pressure) + SSP-RK3 step                                     #
# --------------------------------------------------------------------------- #
def tendencies(u, v, w, theta, tracers, g: PseudoIncompressibleGrid, forcing=None,
               sfc_means=None):
    """Advective tendencies + buoyancy + composable SGS/surface/large-scale forcing,
    NO pressure gradient (applied by the projection). ``sfc_means=(spd_mean, th1_mean)``
    injects GLOBAL surface planar means for the MPI path; None ⇒ local (serial)."""
    cfg = g.cfg
    au, av, aw = _adv.advect_momentum(u, v, w, g.dx, g.dy, g.dz,
                                      cfg.momentum_scheme or cfg.scheme, vel_at_faces=True)
    aw = aw + buoyancy_faces(theta, tracers, g)
    ath = _adv.advect_scalar(theta, u, v, w, g.dx, g.dy, g.dz, cfg.scheme,
                             vel_at_faces=True)
    has_tracers = tracers is not None and tracers.shape[-1] > 0
    atr = None
    if has_tracers:
        # vmap over the trailing tracer axis: one batched advection kernel
        # for all slots (kernel count + compile time independent of n_tracers).
        atr = jax.vmap(
            lambda q: _adv.advect_scalar(q, u, v, w, g.dx, g.dy, g.dz,
                                         cfg.scheme, vel_at_faces=True),
            in_axes=-1, out_axes=-1,
        )(tracers)
    # eddy viscosity computed ONCE, shared by momentum + tracer SGS
    need_sgs = cfg.sgs != "none" or cfg.surface != "free" or has_tracers
    if need_sgs:
        uc, vc, wc = _centre_velocities(u, v, w)
        nu_t = eddy_viscosity(uc, vc, wc, g)
        sfc_flux, cd_surf = _surface_state(u, v, theta, g, forcing, sfc_means)
        if cfg.sgs != "none" or cfg.surface != "free":
            spd1_mean = None if sfc_means is None else sfc_means[0]
            Fu, Fv, Fw, Fth, _ustar = _sgs_force(u, v, w, theta, g, sfc_flux, cd_surf,
                                                 spd1_mean=spd1_mean, nu_t=nu_t)
            au = au + Fu; av = av + Fv; aw = aw + Fw; ath = ath + Fth
        if has_tracers:
            atr = atr + _tracer_sgs(tracers, nu_t, g, cfg.sfc_qv_flux)
    # Coriolis / geostrophic / large-scale forcing (θ + q_v)
    Lu, Lv, Lth, Ltr = _large_scale_forcing(u, v, theta, tracers, g, forcing)
    au = au + Lu; av = av + Lv; ath = ath + Lth
    if Ltr is not None:
        atr = atr + Ltr
    # Opt-in horizontal biharmonic de-noiser (coeff=0 ⇒ no-op, default path unchanged)
    if cfg.hyperdiff_coeff > 0.0:
        au = au + _hyperdiff(u, g); av = av + _hyperdiff(v, g)
        aw = aw + _hyperdiff(w, g); ath = ath + _hyperdiff(theta, g)
        if has_tracers:
            atr = atr + _hyperdiff(tracers, g)
    # re-zero w walls (covers buoyancy + any hyperdiff contribution; w=0 there anyway)
    aw = aw.at[..., 0].set(0.0).at[..., -1].set(0.0)
    return au, av, aw, ath, atr


def _euler_then_project(s_u, s_v, s_w, s_th, s_tr, theta_for_p, tr_for_p,
                        pi_prev, dt, g, forcing=None):
    """One forward-Euler advance of a (possibly RK-combined) state + projection."""
    au, av, aw, ath, atr = tendencies(s_u, s_v, s_w, s_th, s_tr, g, forcing)
    u_s = s_u + dt * au
    v_s = s_v + dt * av
    w_s = (s_w + dt * aw).at[..., 0].set(0.0).at[..., -1].set(0.0)
    th_n = s_th + dt * ath
    tr_n = None if s_tr is None else s_tr + dt * atr
    u_n, v_n, w_n, pi = project(u_s, v_s, w_s, theta_for_p, tr_for_p, pi_prev, dt, g)
    return u_n, v_n, w_n, th_n, tr_n, pi


def make_microphysics(g: PseudoIncompressibleGrid, micro_config, dt,
                      p_sfc=constants.p_ref, qv_prof=None):
    """Composable microphysics adapter (OPERATOR-SPLIT) — reuses the shared LES adapter
    ``spectral_les_moist.make_les_microphysics_fn`` so ANY scheme (kessler/morrison/
    thompson/…) swaps in via ``MicrophysicsConfig`` (dispatch raises on unknown). Returns
    ``micro(theta, tracers) -> (dtheta_dt, dtracers_dt, precip_sfc)`` on the LES grid.

    Apply forward-Euler OUTSIDE :func:`step` (keeps the dycore step scheme-agnostic;
    mirrors the CRM/spectral-LES microphysics cadence)::

        s = step(s, g, dt, forcing)
        dth, dtr, _ = micro(s.theta, s.tracers)
        s = s._replace(theta=s.theta + dt*dth, tracers=s.tracers + dt*dtr)

    The latent heating enters θ through the reference Exner; total water is conserved by
    the scheme. Requires ``cfg.moist`` + enough ``n_tracers`` for the chosen scheme."""
    if not g.cfg.moist or g.cfg.n_tracers < 1:
        raise ValueError(
            "make_microphysics requires cfg.moist=True and n_tracers>=1 (the standard "
            f"q_v,... slot layout); got moist={g.cfg.moist}, n_tracers={g.cfg.n_tracers}.")
    from legoesm.atmosphere.dynamics.les.spectral_les_moist import (
        make_anelastic_reference, make_les_microphysics_fn)
    ref = make_anelastic_reference(g.z_c, g.z_f, p_sfc, g.theta0, qv_prof,
                                   dtype=g.z_c.dtype)
    return make_les_microphysics_fn(micro_config, ref, g.dz, dt)


def step(state: PseudoIncompressibleState, g: PseudoIncompressibleGrid, dt,
         forcing: PseudoIncompressibleForcing | None = None):
    """One SSP-RK3 (Shu–Osher) step with the pressure projection applied per stage.

    Scalars (θ, tracers) advance with the same SSP-RK3 combination; momentum is
    projected divergence-free after each stage. ``forcing`` (explicit, non-closure)
    carries the composable surface temperature / large-scale tendencies; ``None`` ⇒ the
    config-level forcing only (Coriolis/SGS/prescribed flux). ``dt`` may be a Python
    float or a JAX scalar. The Poisson coefficient uses the stage potential temperature.
    """
    u0, v0, w0, th0, tr0 = state.u, state.v, state.w, state.theta, state.tracers
    pi0 = state.pi_prev

    # Stage 1: y1 = project(y0 + dt L(y0))
    u1, v1, w1, th1, tr1, pi1 = _euler_then_project(
        u0, v0, w0, th0, tr0, th0, tr0, pi0, dt, g, forcing)
    # Stage 2: y2 = 3/4 y0 + 1/4 (y1 + dt L(y1))
    eu, ev, ew, eth, etr, _ = _euler_then_project(
        u1, v1, w1, th1, tr1, th1, tr1, pi1, dt, g, forcing)
    a, b = 0.75, 0.25
    u2 = a * u0 + b * eu; v2 = a * v0 + b * ev
    w2 = (a * w0 + b * ew).at[..., 0].set(0.0).at[..., -1].set(0.0)
    th2 = a * th0 + b * eth
    tr2 = None if tr0 is None else a * tr0 + b * etr
    u2, v2, w2, pi2 = project(u2, v2, w2, th2, tr2, pi1, dt, g)
    # Stage 3: y3 = 1/3 y0 + 2/3 (y2 + dt L(y2))
    eu, ev, ew, eth, etr, _ = _euler_then_project(
        u2, v2, w2, th2, tr2, th2, tr2, pi2, dt, g, forcing)
    a, b = 1.0 / 3.0, 2.0 / 3.0
    u3 = a * u0 + b * eu; v3 = a * v0 + b * ev
    w3 = (a * w0 + b * ew).at[..., 0].set(0.0).at[..., -1].set(0.0)
    th3 = a * th0 + b * eth
    tr3 = None if tr0 is None else a * tr0 + b * etr
    u3, v3, w3, pi3 = project(u3, v3, w3, th3, tr3, pi2, dt, g)
    # CFL-unlimited [1,2,1] de-noiser on VELOCITY — removes the 2Δ grid noise that
    # NON-dissipative (central) momentum generates. The horizontal filter commutes
    # with the C-grid divergence (periodic x,y), so div(H·u)=H(div u)=0 ⇒ divergence-
    # free is PRESERVED with no re-projection (mirrors the spectral sharp-filter pairing).
    sm = g.cfg.momentum_shapiro_coeff
    if sm > 0.0:
        mo = g.cfg.momentum_shapiro_order
        u3 = u3 + sm * (_shapiro_h(u3, mo) - u3)
        v3 = v3 + sm * (_shapiro_h(v3, mo) - v3)
        w3 = w3 + sm * (_shapiro_h(w3, mo) - w3)
    # Per-step [1,2,1] de-noiser on the SCALARS only (θ, tracers) — kills the
    # fine-res stable-BL 2Δ θ-mode without touching the div-free velocity
    # (no re-projection needed). Static gate ⇒ default (s=0) is bit-identical.
    s = g.cfg.shapiro_coeff
    if s > 0.0:
        so = g.cfg.shapiro_order
        th3 = th3 + s * (_shapiro_h(th3, so) - th3)
        if tr3 is not None:
            tr3 = tr3 + s * (_shapiro_h(tr3, so) - tr3)
    return PseudoIncompressibleState(u=u3, v=v3, w=w3, theta=th3, pi_prev=pi3,
                                     tracers=tr3)
