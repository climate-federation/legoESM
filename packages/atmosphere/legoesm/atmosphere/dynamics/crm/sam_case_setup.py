"""Assemble a legoESM plane-CRM run from a SAM CASE directory (GATE/LBA/…).

Ties the SAM CASE reader (:mod:`legoesm.atmosphere.sam_case_forcing`) and the
D3 large-scale-forcing operator
(:mod:`legoesm.atmosphere.dynamics.plane_large_scale_forcing`) into the pieces a
plane-CRM driver needs:

* a :class:`HeightCoordinate` whose reference θ(z) **is** the case sounding (so
  the rest state ``θ'=0`` starts at the observed base state, exactly as SAM
  initialises from the ``snd`` file);
* a :class:`PlaneNonHydrostaticState` with the sounding ``u``, ``v``, ``q_v``
  and a small symmetry-breaking θ seed in the boundary layer;
* the large-scale-forcing ``physics_fn`` built from the ``lsf`` file;
* the surface SST + the Coriolis ``f`` implied by the case latitude.

GATE_IDEAL fidelity (from ``CASES/GATE_IDEAL/prm``): ``docoriolis=.false.``,
``donudging_uv=.true.`` with ``tauls=21600`` s, ``dolargescale=.true.``,
interactive surface fluxes over a fixed SST.  So the lsf ``u_ls``/``v_ls`` are
the wind-NUDGING targets (not a Coriolis reference), and Coriolis is off.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import make_rest_state
from legoesm.atmosphere.dynamics.plane_large_scale_forcing import (
    make_plane_ls_forcing_from_sam_case,
    make_plane_ls_forcing_physics,
)
from legoesm.atmosphere.sam_case_forcing import (
    extend_sounding_to_top,
    interp_sounding_to_levels,
    read_sam_rad,
    read_sam_sfc,
    read_sam_snd,
    surface_at_day,
)
from legoesm.grids.vertical import (
    HeightCoordinate, create_height_coordinate_from_z_half,
    create_stretched_height_coordinate,
)


def coriolis_f0(latitude_deg: float) -> float:
    """f-plane Coriolis parameter ``f = 2 Ω sin(φ)`` [1/s]."""
    return float(2.0 * constants.Omega * np.sin(np.deg2rad(latitude_deg)))


def _model_top_z(H: float, grd_file: str | None) -> float:
    """Geometric height of the model top: the SAM ``grd`` top scalar level when
    ``use_sam_grd`` (``grd_file`` given), else ``H`` (the stretched-grid top).

    Used to size the SND-TOP sounding extension so it always covers the model
    top (the grd top ≈30 km exceeds the H≈20 km stretched smoke top)."""
    if grd_file is not None:
        from legoesm.atmosphere.sam_case_forcing import read_sam_grd
        return float(read_sam_grd(grd_file).z_full_bottom_up[-1])
    return float(H)


def build_sam_case_height_coord(
    snd, nlev: int, H: float, p_sfc_pa: float, dz_sfc: float = 50.0,
    grd_file: str | None = None, allow_short_sounding: bool = False,
    dtype=jnp.float64,
) -> HeightCoordinate:
    """HeightCoordinate whose reference θ(z) is the SAM sounding.

    The geostrophic-reference winds ``u_geo0``/``v_geo0`` are set from the
    sounding wind on the model levels (used by the plane Coriolis term when a
    case enables it; harmless when Coriolis is off).

    Vertical grid (VGRID):

    * ``grd_file`` given ⇒ the EXACT SAM ``grd`` levels (``read_sam_grd`` +
      :func:`create_height_coordinate_from_z_half`) — the faithful grid (dz=50 m
      BL, ~100 m through the deep-convection layer, stretched aloft; ``nlev``/
      ``H``/``dz_sfc`` are ignored). SAM's GATE grd is 266 levels ⇒ CPU-heavy.
    * ``grd_file=None`` (default) ⇒ a GEOMETRIC stretch with ``dz_sfc=50 m``
      (the CPU-feasible smoke compromise: 50 m BL faithful, but the mid-
      troposphere coarsens faster than SAM's 100 m uniform). nlev≈64 keeps the
      stretch gentle; the old uniform dz=H/nlev (≈667 m) was 13× too coarse.
    """
    z_snd = jnp.asarray(snd.z, dtype=dtype)
    theta_snd = jnp.asarray(snd.theta, dtype=dtype)

    def theta_ref_fn(z):
        return jnp.interp(z, z_snd, theta_snd)

    if grd_file is not None:
        from legoesm.atmosphere.sam_case_forcing import read_sam_grd
        grd = read_sam_grd(grd_file)
        hc = create_height_coordinate_from_z_half(
            jnp.asarray(grd.z_half, dtype=dtype),
            theta_ref_fn=theta_ref_fn, p_sfc=p_sfc_pa,
        )
    else:
        hc = create_stretched_height_coordinate(
            nlev, H=H, dz_sfc=dz_sfc, p_sfc=p_sfc_pa, theta_ref_fn=theta_ref_fn,
        )
        # codex iter-51 D (HIGH): too few levels for (dz_sfc, H) forces an
        # aggressively stretched grid (large dz aloft) — an accuracy + stability
        # risk. SAM's GATE grd uses ~64 levels; warn so callers bump nlev.
        _dz = np.asarray(hc.dz)
        _ratio = float(_dz.max() / _dz.min())
        if _ratio > 30.0:
            import warnings
            warnings.warn(
                f"build_sam_case_height_coord: stretch ratio dz_top/dz_sfc="
                f"{_ratio:.0f} (>30) at nlev={nlev}, H={H:.0f}, "
                f"dz_sfc={dz_sfc:.0f}. Use more levels (SAM GATE grd ≈64) or "
                f"grd_file= for the exact SAM grid.",
                stacklevel=2,
            )
    # SND-TOP coverage guard (codex iter-57b, HIGH): the constructed grid top
    # MUST be covered by the sounding, else theta_ref_fn's jnp.interp silently
    # CLAMPS θ constant aloft (the dry-neutral bug) — and a θ_ic≈θ_ref check
    # would still pass because BOTH clamp identically. The real cases pre-extend
    # via extend_sounding_to_top(snd, _model_top_z(...)); this asserts the EXACT
    # constructed grid (not just the +margin estimate) is covered. Synthetic-
    # sounding unit tests opt out with allow_short_sounding=True (legacy clamp).
    z_model_top = float(np.asarray(hc.z_full).max())
    z_snd_top = float(np.asarray(snd.z).max())
    if not allow_short_sounding and z_snd_top < z_model_top - 1.0:
        raise ValueError(
            f"build_sam_case_height_coord: sounding top {z_snd_top:.0f} m does "
            f"not cover the constructed model top {z_model_top:.0f} m — θ_ref "
            "would clamp constant (dry-neutral) aloft. Pre-extend with "
            "extend_sounding_to_top(snd, model_top), or pass "
            "allow_short_sounding=True for a synthetic/stub sounding.")
    levs = interp_sounding_to_levels(snd, np.asarray(hc.z_full))
    return hc._replace(
        u_geo0=jnp.asarray(levs["u"], dtype=dtype),
        v_geo0=jnp.asarray(levs["v"], dtype=dtype),
    )


def _smooth_k1_theta_seed(ny, nx, nlev, n_seed_lev, amp, dtype):
    """Smooth k=1 cosine θ' seed in the lowest ``n_seed_lev`` levels.

    A RESOLVED (lowest-wavenumber) symmetry breaker — white noise has full
    power at the grid scale where Smagorinsky/hyperdiff are weakest at t=0 and
    NaNs the semi-implicit acoustic core (same rationale as the RCEMIP plane
    IC).  Zero horizontal mean so the IC adds no domain-mean θ.
    """
    ix = jnp.arange(nx, dtype=dtype)
    iy = jnp.arange(ny, dtype=dtype)
    pattern = (jnp.cos(2 * jnp.pi * iy / ny)[:, None]
               * jnp.cos(2 * jnp.pi * ix / nx)[None, :])
    n_seed_lev = min(n_seed_lev, nlev)
    seed = amp * pattern[:, :, None] * jnp.ones((1, 1, n_seed_lev), dtype=dtype)
    seed = seed - jnp.mean(seed, axis=(0, 1), keepdims=True)
    theta_p = jnp.zeros((ny, nx, nlev), dtype=dtype)
    return theta_p.at[..., -n_seed_lev:].set(seed)  # bottom = last entries


def band_limited_seed_pattern(ny, nx, k_max=6, rng_seed=0):
    """A horizontal BAND-LIMITED random field (modes 1≤|k|≤k_max), unit std,
    zero mean. White noise has 2Δx grid-scale power that NaNs the rest-state IC
    (Smag K=0 at t=0); a single smooth k=1 wave seeds ONE domain-scale plume
    that never breaks into cells. A band of RESOLVED wavenumbers (k=1..k_max,
    well below Nyquist) gives a MULTI-CELL pattern WITHOUT grid-scale power — the
    SAM-like broadband boundary-layer-noise analogue (cf SAM's C4 random θ'
    perturbation in the lowest levels)."""
    rng = np.random.default_rng(rng_seed)
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2)
    band = ((kmag >= 1.0) & (kmag <= float(k_max))).astype(np.float64)
    spec = band * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=(ny, nx)))
    field = np.real(np.fft.ifft2(spec))
    field = field - field.mean()
    std = field.std()
    return field / std if std > 0 else field


def _random_band_theta_seed(ny, nx, nlev, n_seed_lev, amp, k_max, rng_seed,
                            dtype):
    """Band-limited RANDOM θ' seed (SAM-like C4 broadband BL noise) in the lowest
    ``n_seed_lev`` levels — multi-cell triggering, zero horizontal mean."""
    n_seed_lev = min(n_seed_lev, nlev)
    pattern = jnp.asarray(
        band_limited_seed_pattern(ny, nx, k_max, rng_seed), dtype)
    seed = amp * pattern[:, :, None] * jnp.ones((1, 1, n_seed_lev), dtype=dtype)
    seed = seed - jnp.mean(seed, axis=(0, 1), keepdims=True)
    theta_p = jnp.zeros((ny, nx, nlev), dtype=dtype)
    return theta_p.at[..., -n_seed_lev:].set(seed)


def build_sam_case_initial_state(
    snd, grid, height_coord, *, n_tracers: int = 10,
    seed_amp: float = 0.1, n_seed_lev: int = 4,
    seed_kind: str = "smooth_k1", seed_kmax: int = 6, rng_seed: int = 0,
    dtype=jnp.float64,
):
    """Plane state from the SAM sounding: u, v, q_v + θ seed + hydrostatic ρ'.

    ``θ'`` is ``θ_snd − θ_ref`` (≈0, since θ_ref IS the sounding) plus the
    boundary-layer seed; ``ρ'`` is the DRY warm-bubble linearisation
    ``−ρ₀·θ'/θ_ref`` so the IC PRESSURE perturbation is zero (no acoustic shock).

    Dry reference + moist tracer (codex iter-33 C/D): the plane core uses a DRY
    reference state (``θ_ref`` dry, dry EOS ``p=EOS(ρ,θ)``) and carries the full
    ``q_v`` as a tracer; moisture enters the dynamics through the SEPARATE moist
    buoyancy term ``moisture_buoyancy_w_half`` (vapour-virtual + condensate
    loading), which by design reproduces SAM's ``buoyancy.f90`` — INCLUDING the
    ``g·ε_v·⟨q_v⟩`` unbalanced-mean-buoyancy that drives the moist circulation.
    So ``ρ'`` is correctly DRY here (putting moisture in ``ρ'`` would double-count
    against ``moisture_buoyancy_w_half`` and the dry EOS); this matches the
    validated RCEMIP plane IC.
    """
    rest = make_rest_state(grid, height_coord, dtype=dtype)
    ny, nx, nlev = rest.theta_prime.data.shape
    levs = interp_sounding_to_levels(snd, np.asarray(height_coord.z_full))
    u = jnp.broadcast_to(jnp.asarray(levs["u"], dtype).reshape(1, 1, nlev),
                         (ny, nx, nlev))
    v = jnp.broadcast_to(jnp.asarray(levs["v"], dtype).reshape(1, 1, nlev),
                         (ny, nx, nlev))
    q_v = jnp.asarray(levs["q_v"], dtype)
    tracers = jnp.zeros((ny, nx, nlev, n_tracers), dtype=dtype)
    tracers = tracers.at[..., 0].set(
        jnp.broadcast_to(q_v.reshape(1, 1, nlev), (ny, nx, nlev)))

    theta_ref = jnp.asarray(height_coord.theta_ref, dtype=dtype)
    theta_base = jnp.asarray(levs["theta"], dtype) - theta_ref          # ≈0
    if seed_kind == "band_random":
        seed_field = _random_band_theta_seed(
            ny, nx, nlev, n_seed_lev, seed_amp, seed_kmax, rng_seed, dtype)
    elif seed_kind == "smooth_k1":
        seed_field = _smooth_k1_theta_seed(
            ny, nx, nlev, n_seed_lev, seed_amp, dtype)
    else:
        raise ValueError(
            f"seed_kind must be 'smooth_k1' or 'band_random', got {seed_kind!r}")
    theta_p = seed_field + theta_base.reshape(1, 1, nlev)
    rho_0 = jnp.asarray(height_coord.rho_ref, dtype=dtype)
    rho_p = -rho_0 * theta_p / theta_ref            # hydrostatic IC
    assert u.shape == (ny, nx, nlev), (u.shape, (ny, nx, nlev))
    assert tracers.shape == (ny, nx, nlev, n_tracers), tracers.shape
    assert theta_p.shape == (ny, nx, nlev), theta_p.shape
    return rest._replace(
        u=rest.u.replace(data=u),
        v=rest.v.replace(data=v),
        theta_prime=rest.theta_prime.replace(data=theta_p),
        rho_prime=rest.rho_prime.replace(data=rho_p),
        tracers=rest.tracers.replace(data=tracers),
    )


class SAMCaseSetup(NamedTuple):
    """Assembled components for a SAM CASE plane-CRM run."""
    height_coord: HeightCoordinate
    initial_state: object
    ls_forcing_physics: object
    sst: float
    p_sfc_pa: float
    f0: float
    coriolis: bool


def build_gate_ideal_setup(
    case_dir, grid, *, nlev: int, H: float,
    n_tracers: int = 10, seed_amp: float = 0.1, use_sam_grd: bool = False,
    dtype=jnp.float64,
) -> SAMCaseSetup:
    """Assemble the GATE_IDEAL case for the plane CRM (per its ``prm``).

    Reads ``snd``/``lsf``/``sfc`` from ``case_dir`` and returns the height
    coordinate, initial state, large-scale-forcing physics_fn (subsidence +
    advective tendencies + DOMAIN-MEAN wind nudging at τ=21600 s), the fixed
    SST, and Coriolis metadata.  GATE_IDEAL runs ``docoriolis=.false.`` so
    ``coriolis=False`` (the f0 is reported for reference only, from φ=8.5°N).

    ``use_sam_grd=True`` uses the EXACT SAM ``grd`` vertical levels (266, the
    faithful grid — CPU-heavy; ``nlev``/``H`` ignored); default False uses the
    geometric stretch (the CPU-feasible smoke compromise).
    """
    case_dir = Path(case_dir)
    grd_file = str(case_dir / "grd") if use_sam_grd else None
    # SND-TOP (iter-57): extend the sounding above its top (isothermal-to-warming
    # US-std-atm stratosphere) so θ_ref + IC share a physical (stable, not
    # dry-neutral) extrapolation that COVERS the model top (grd top or H).
    snd = extend_sounding_to_top(
        read_sam_snd(case_dir / "snd"), _model_top_z(H, grd_file))
    sfc = read_sam_sfc(case_dir / "sfc")
    p_sfc_pa = float(snd.pres0) * 100.0          # mb → Pa
    hc = build_sam_case_height_coord(
        snd, nlev, H, p_sfc_pa, grd_file=grd_file, dtype=dtype)
    state = build_sam_case_initial_state(
        snd, grid, hc, n_tracers=n_tracers, seed_amp=seed_amp, dtype=dtype)
    # GATE_IDEAL prm: dolargescale + donudging_uv (tauls=21600 s), no T/q nudge.
    forcing = make_plane_ls_forcing_from_sam_case(
        hc, case_dir / "lsf",
        include_subsidence=True, include_advective=True,
        nudge_winds=True, tau_nudge=21600.0,
    )
    sfc0 = surface_at_day(sfc, day=0.0)
    return SAMCaseSetup(
        height_coord=hc, initial_state=state, ls_forcing_physics=forcing,
        sst=sfc0["sst"], p_sfc_pa=p_sfc_pa,
        f0=coriolis_f0(8.5), coriolis=False,
    )


def apply_prescribed_surface_fluxes_plane(state, shflx, lhflx, height_coord, dt,
                                          cd_momentum=0.0, gust=1.0):
    """Forward-Euler apply of PRESCRIBED surface sensible + latent heat fluxes
    (+ optional interactive momentum drag) to the lowest model level (SAM
    ``SFC_FLX_FXD``, LBA-style).

    ``shflx``/``lhflx`` [W/m²] (scalars, the current-time values from the case
    ``sfc`` file) enter the lowest cell as a kinematic flux divergence:

        dθ'/dt|_sfc = shflx / (ρ_sfc·c_p·dz_sfc·Π_sfc)
        dq_v/dt|_sfc = lhflx / (ρ_sfc·L_v·dz_sfc)

    where Π_sfc is the surface Exner factor (T→θ). The column-integrated moisture
    source = ρ·dz·dq_v = LE/L_v exactly (and likewise heat = H/c_p).

    INTERACTIVE momentum drag (``cd_momentum`` > 0, SAM ``SFC_TAU_FXD=.false.``):
    a bulk-aerodynamic surface stress ``τ = ρ·C_d·|U|·U`` decelerating the lowest
    wind, ``du/dt|_sfc = −C_d·|U|·u/dz_sfc`` (likewise v), with a gustiness floor
    ``|U|=√(u²+v²+gust²)`` so the free-convective surface drag does not vanish at
    zero mean wind.

    Applied OUTSIDE ``model.step`` (like the gated radiation) so the host-side
    time-varying flux series needs no JIT recompilation. The scalar flux goes
    into the LOWEST CELL (column integral conserved; the Smagorinsky SGS then
    mixes it upward — an approximation to SAM's surface-layer SGS bottom-BC, fine
    at LBA's small dz/dt). Returns the updated state.
    """
    rho_sfc = float(height_coord.rho_ref[-1])
    exner_sfc = float(height_coord.exner_ref[-1])
    dz_sfc = float(height_coord.dz[-1])
    dtheta = shflx / (rho_sfc * constants.c_pd * dz_sfc * exner_sfc)
    dqv = lhflx / (rho_sfc * constants.L_v * dz_sfc)
    theta_p = state.theta_prime.data.at[..., -1].add(dt * dtheta)
    tracers = state.tracers.data.at[..., -1, 0].add(dt * dqv)
    new = state._replace(
        theta_prime=state.theta_prime.replace(data=theta_p),
        tracers=state.tracers.replace(data=tracers),
    )
    if cd_momentum > 0.0 and state.v is not None:
        u = state.u.data
        v = state.v.data
        speed = jnp.sqrt(u[..., -1] ** 2 + v[..., -1] ** 2 + gust ** 2)
        drag = cd_momentum * speed / dz_sfc                # 1/s
        u_new = u.at[..., -1].add(-dt * drag * u[..., -1])
        v_new = v.at[..., -1].add(-dt * drag * v[..., -1])
        new = new._replace(
            u=new.u.replace(data=u_new), v=new.v.replace(data=v_new))
    return new


def apply_prescribed_radiative_cooling_plane(state, dTdt_rad, height_coord, dt):
    """Forward-Euler apply of a PRESCRIBED radiative heating profile (SAM
    ``doradforcing``, LBA) to θ' at ALL levels.

    ``dTdt_rad`` is the ``(nlev,)`` prescribed radiative dT/dt [K/s] on the model
    levels (typically ≈−1.4 K/day cooling in the troposphere). Converted to a θ
    tendency via the reference Exner (dθ' = dT/Π_ref since θ_ref is fixed) and
    integrated forward-Euler — used INSTEAD of interactive RRTM for LBA. Applied
    OUTSIDE ``model.step`` so a time-varying profile needs no recompilation.
    """
    exner = jnp.asarray(height_coord.exner_ref)            # (nlev,)
    nlev = exner.shape[0]
    dtheta = jnp.asarray(dTdt_rad) / exner                 # K/s on θ'
    theta_p = state.theta_prime.data + dt * dtheta.reshape(1, 1, nlev)
    return state._replace(
        theta_prime=state.theta_prime.replace(data=theta_p))


class LBACaseSetup(NamedTuple):
    """Assembled components for the LBA land-diurnal plane-CRM run."""
    height_coord: HeightCoordinate
    initial_state: object
    ls_forcing_physics: object      # wind nudging only (LBA has no lsf)
    sfc: object                     # SAMSurface: time-varying H/LE flux series
    rad: object                     # SAMRadForcing: prescribed dT/dt|_rad series
    p_sfc_pa: float
    latitude: float


def build_lba_setup(
    case_dir, grid, *, nlev: int, H: float,
    n_tracers: int = 11, seed_amp: float = 0.1, tau_nudge: float = 7200.0,
    seed_kind: str = "band_random", latitude: float = -10.8,
    use_sam_grd: bool = False, dtype=jnp.float64,
) -> LBACaseSetup:
    """Assemble the LBA land-diurnal case for the plane CRM (per its ``prm``).

    LBA (`LAND=.true.`, `dolargescale=.false.`, `SFC_FLX_FXD=.true.`,
    `docoriolis=.false.`, `donudging_uv` τ=7200 s): convection is driven by the
    PRESCRIBED diurnal surface sensible+latent heat fluxes (peaking ≈270/554
    W/m² at local noon), NOT large-scale forcing. Reads ``snd`` (initial state) +
    ``sfc`` (the H/LE flux time series); the only "forcing" physics is wind
    nudging toward the sounding wind. The run driver applies the time-varying
    surface fluxes via :func:`apply_prescribed_surface_fluxes_plane`.
    """
    case_dir = Path(case_dir)
    grd_file = str(case_dir / "grd") if use_sam_grd else None
    # SND-TOP (iter-57): extend the sounding above its top (US-std-atm
    # stratosphere) so θ_ref + IC share a physical extrapolation covering the
    # model top (grd top or H).
    snd = extend_sounding_to_top(
        read_sam_snd(case_dir / "snd"), _model_top_z(H, grd_file))
    sfc = read_sam_sfc(case_dir / "sfc")
    # LBA prm doradforcing=.true. ⇒ PRESCRIBED radiative cooling (rad file), NOT
    # interactive RRTM (dolongwave=doshortwave=.false.).
    rad = read_sam_rad(case_dir / "rad")
    p_sfc_pa = float(snd.pres0) * 100.0
    hc = build_sam_case_height_coord(
        snd, nlev, H, p_sfc_pa, grd_file=grd_file, dtype=dtype)
    # codex iter-57/57b: interp_rad_to_levels CLAMPS the prescribed cooling above
    # the rad-file top; if the model extends higher AND the rad cooling is nonzero
    # NEAR the top, that value spuriously cools the stratosphere. LBA's rad tail
    # is 0 (correct) — guard a future nonzero-top file. Check the last few levels
    # (|·|, all times), not only the very top, to catch an unresolved tail.
    rad_z = np.asarray(rad.z)
    z_rad_top = float(rad_z.max())
    if float(np.asarray(hc.z_full).max()) > z_rad_top + 1.0:
        ntail = min(3, rad_z.shape[-1])
        top_cool = float(np.max(np.abs(np.asarray(rad.dTdt_rad)[..., -ntail:])))
        if top_cool > 1.0e-7:        # >~0.01 K/day in the top few rad levels
            raise ValueError(
                f"build_lba_setup: model top exceeds the rad-file top "
                f"{z_rad_top:.0f} m but the top {ntail} rad levels cool at "
                f"{top_cool:.2e} K/s (≠0) — interp would clamp that into the "
                "stratosphere. Extend the rad profile or lower H.")
    state = build_sam_case_initial_state(
        snd, grid, hc, n_tracers=n_tracers, seed_amp=seed_amp,
        seed_kind=seed_kind, dtype=dtype)
    # LBA: no lsf — the only large-scale "forcing" is DOMAIN-MEAN wind nudging
    # toward the sounding wind (hc.u_geo0/v_geo0), donudging_uv τ=7200 s.
    forcing = make_plane_ls_forcing_physics(
        hc, u_nudge=hc.u_geo0, v_nudge=hc.v_geo0, tau_nudge=tau_nudge)
    return LBACaseSetup(
        height_coord=hc, initial_state=state, ls_forcing_physics=forcing,
        sfc=sfc, rad=rad, p_sfc_pa=p_sfc_pa, latitude=latitude,
    )
