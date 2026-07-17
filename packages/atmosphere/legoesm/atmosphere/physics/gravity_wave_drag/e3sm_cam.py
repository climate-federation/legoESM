"""Faithful E3SM/CAM gravity-wave drag (``gw_drag_prof`` spectral solver).

This is a differentiable JAX re-implementation of the core E3SM EAM
gravity-wave module, faithful term-by-term to the Fortran oracle:

* ``gw_prof``       — background interface density ``rhoi`` and
  Brunt-Vaisala frequency ``ni`` (+ midpoint ``nm``).
  (``components/eam/src/physics/cam/gw_common.F90``)
* ``gw_oro_src``    — McFarlane (1987) orographic c=0 source: depth-averaged
  source region, ``sghmax`` Froude cap, ``tauoro`` launch stress.
  (``components/eam/src/physics/cam/gw_oro.F90``)
* ``gw_cm_src``     — frontal source: uniform Gaussian spectrum launched
  where the frontogenesis function exceeds a threshold.
  (``components/eam/src/physics/cam/gw_front.F90``)
* ``gw_drag_prof``  — the spectral solver: bottom-up stress profile with
  Lindzen saturation + WKB damping, then top-down stress-divergence
  tendency with the stability limiters and re-derived stress.
  (``components/eam/src/physics/cam/gw_common.F90``)

Faithfulness scope & departures
-------------------------------
"Faithful term-by-term" is pinned to **E3SM-3.0.1** (``gw_common.F90`` /
``gw_oro.F90`` / ``gw_front.F90`` at the line numbers cited inline), pinned by
the committed gfortran oracle regression anchors in
``tests/atmosphere/hydrostatic/unit/test_gwd_e3sm_cam.py`` (anchors truncated
to 8 significant figures, tested at ``rtol=1e-7``) — NOT by a live diff against
a vendored source tree (the Fortran is path-referenced, not committed). Known,
deliberate departures / version choices (canaries in
``test_e3sm_cam_gwd_faithful.py``):

* **Heating frame (version choice).** The spectral thermal term defaults to the
  E3SM-3.0.1 *ground-relative* form ``dttke = sum_l c_l*gwut_l``
  (``gw_common.F90:727``). This is the wave-energy-flux-divergence term, NOT the
  irreversible heating: for ``U>c>0`` (``gwut<0``) it is negative, so the
  shipped spectral path can locally cool. Newer CAM/EAM trunk uses the
  *intrinsic-frequency* form ``sum_l (c_l - ubm)*gwut_l`` (the irreversible
  dissipative conversion, non-negative for the ``sign(c-ubm)`` tendency); the
  two differ by ``-ubm*gwut``. Opt in with ``config.dttke_use_intrinsic``
  (default ``False`` = the pinned 3.0.1 oracle).
* **Standalone eddy diffusion (default OFF).** ``config.do_eddy_diffusion``
  defaults to ``False``. E3SM exports the GW eddy diffusivity (the ``EKGWSPEC``
  diagnostic) and defers the u/v/T eddy diffusion to the host
  ``vertical_diffusion`` scheme. When enabled here the module applies the
  E3SM-faithful dry-static-energy diffusion heating ``dttdf`` AND, so the GW
  momentum eddy flux is not dropped standalone, the u/v eddy diffusion that
  E3SM defers to the host (``dttdf`` and the u/v diffusion are applied only in
  this branch, not otherwise).
* **Driver-level orographic heating + landfrac (E3SM gw_tend, gw_drag.F90:
  902-915).** E3SM applies the oro tendencies OUTSIDE gw_drag_prof: it scales
  ``utgw *= cam_in%landfrac`` (:904-906, zeroing oro drag over ocean) and heats
  with the DISCRETE-step KE closure ``ptend%s += -(ptend%u*(u+0.5*dt*ptend%u)
  + ...)`` (:908-913).  Both are available here: pass ``land_frac_col`` for
  the landfrac scaling (``None`` default = no scaling), and set
  ``config.use_discrete_ke_heating=True`` for the discrete closure (default
  ``False`` = the continuous-rate identity, which over-heats by
  ``0.5*dt*(du²+dv²)/c_pd`` per step).  STRUCTURAL DEPARTURE kept: E3SM's
  closure runs once over the ACCUMULATED ptend of all GW sources ("includes
  spectrum"); our per-source calls close each source independently.  (The
  alternative ``use_gw_energy_fix`` branch, :916-937, deposits a
  column-UNIFORM dE — code default ``.false.``, phys_control.F90:174 — and is
  not implemented.)
* **Newtonian alpha profile (default OFF).** ``config.use_newtonian_profile``
  defaults to ``False``; E3SM uses a Newtonian-cooling vertical ``alpha(z)``
  profile in the spectral saturation / WKB damping (with an orographic floor).
  Enabling it changes the drag.
* **Beres (2004) convective source: TABLE not bundled.** ``source="convective"``
  RUNS, but on a clearly-labelled analytic STAND-IN spectrum
  (``build_stand_in_mfcc``, explicitly NOT bit-faithful to Beres); the real
  offline ``mfcc`` lookup table (``newmfspectra*.nc``) is not vendored. Pass a
  real ``mfcc_table`` (``beres.use_stand_in_table=False``) for faithfulness.
* **Smoothing.** The genuine Fortran kinks (critical level, saturation cap,
  tendency limiters) are kept as-is — no extra sigmoids — plus a few defensive
  clamps (e.g. ``mi>=0``) that bind only for pathological inputs (see below).

Conservation (see ``__physics_contract__`` energy note): ``conserves=["none"]``
as the static INTERSECTION over all selectable sources (a contract must hold
for every config a user can select). The orographic (c=0, DEFAULT) path IS
energy-conserving in-atmosphere — the resolved mean-flow KE it removes is
returned as heat by construction. But the frontal / convective sources launch
NONSTATIONARY spectral components from an EXTERNAL, unbudgeted reservoir
(momentum not conserved), and their default ground-relative thermal term is not
the irreversible ``(c-ubm)*gwut`` conversion — so no single conserved quantity
holds across every source. ``do_energy_conservation=True`` forces the discrete
air-column momentum+dse finite-step residual to zero (a corrective
redistribution below source, not a physical source / boundary-flux accounting).

Index convention (matches the rest of legoESM column physics): array axis
``k = 0`` is the **model top**, ``k = nlev-1`` is the **surface**; interface
arrays have ``nlev+1`` points with index ``0`` the top interface and
``nlev`` the surface interface.  This is the E3SM ``1..pver`` /
``0..pver`` convention shifted to 0-based Python indexing.

Differentiability notes
-----------------------
The E3SM solver is piecewise-smooth.  The kinks are the same ones the Fortran
has — the critical-level test ``sign(u-c)`` changing between interfaces (a
physical discontinuity), the ``min(taudmp, tausat)`` saturation cap, and the
``min`` tendency limiters — plus a few defensive clamps (e.g. ``mi>=0``) that
bind only for pathological inputs.  All are kept exactly (via ``jnp.where`` /
``jnp.minimum`` / ``jnp.maximum``); their sub-gradients are correct on each
side and the kinks are measure-zero, so ``jax.grad`` is well-defined almost
everywhere and never NaN.  No hard Python control flow on traced values is
used; both vertical sweeps are ``lax.scan``.

The Beres (2004) convective source (``gw_convect.F90``) source-SPECTRUM table
(the offline ``mfcc`` lookup, ``newmfspectra*.nc``) is not vendored;
``source="convective"`` runs on a clearly-labelled analytic stand-in
(``build_stand_in_mfcc``) unless a real ``mfcc_table`` is supplied.  See module
``REPORT`` for the gap.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    E3SMBeresConfig,
    E3SMCAMConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput

# Machine-checked scheme contract (see tests/test_physics_contracts.py). Applies
# to the top-level driver ``e3sm_cam_gwd``; the ``gw_*`` functions above/below
# are its faithful E3SM sub-steps (source, stress solver, ediff, fixer).
__physics_contract__ = {
    "summary": (
        "Faithful differentiable E3SM/CAM gravity-wave drag (gw_drag_prof "
        "spectral solver): orographic (McFarlane c=0), frontal (CM), or "
        "convective (Beres) source launched, propagated with Lindzen "
        "saturation + WKB damping, deposited as wind tendencies + heating."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "rho": "kg/m^3", "lat": "rad", "dt": "s",
        "h_topo_col": "m (optional subgrid orographic stddev, sgh)",
        "frontgf_col": "K^2 m^-2 s^-1 (optional frontogenesis function, frontal source)",
        "netdt_col": "K/s (optional convective heating rate, Beres source)",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "eps_gwd": "W/m^2",
    },
    "sign_convention": (
        "z up; k=0 model top, k=nlev-1 surface. EACH wave's tendency is signed "
        "sign(c - ubm) -- it drives the flow toward that wave's phase speed c "
        "(deceleration for c<U, acceleration for c>U), and its limiter caps the "
        "magnitude without changing sign; the SUMMED multi-wave tendency need "
        "not have a single sign. For the orographic (single c=0 wave) path "
        "dT_dt = -(u*du+v*dv)/c_pd >= 0 and eps_gwd >= 0 (resolved KE->heat). "
        "For the spectral path dT_dt is the ground-relative dttke term, which "
        "is SIGNED (can cool where U>c>0). eps_gwd = -integral rho*(u*du+v*dv)*dz "
        "is the mean-flow KE removal rate (positive for the orographic path; "
        "SIGNED for spectra, negative where the flow is accelerated toward c)."
    ),
    # Conserved quantity: NONE — the static INTERSECTION over all selectable
    # sources (a contract must hold for every config a user can select). This is
    # NOT because the default runtime path fails to conserve; it is because some
    # selectable source does. Nuance (pinned in test_e3sm_cam_gwd_faithful.py):
    #  - Orographic (c=0, the DEFAULT source) IS energy-conserving in-atmosphere:
    #    a stationary mountain exchanges momentum without mechanical work, and the
    #    code returns the resolved mean-flow KE it removes as heat BY CONSTRUCTION
    #    (dT_dt=-(u*du+v*dv)/c_pd, so c_pd*sum(rho*dT*dz)==eps_gwd definitionally).
    #    Momentum is a surface sink (mountain drag), never conserved.
    #  - Frontal / convective sources launch NONSTATIONARY spectral components
    #    whose momentum AND energy come from an EXTERNAL, unbudgeted reservoir
    #    (the front / convection) -> a one-way source, so momentum is not
    #    conserved. Their DEFAULT thermal term is the GROUND-RELATIVE
    #    dttke=sum_l c_l*gwut_l, which is the wave-energy-flux-divergence term,
    #    NOT the irreversible conversion sum_l (c_l-ubm)*gwut_l (they differ by
    #    ubm*gwut, pinned by test_frontal_dttke_intrinsic_switch) and can be
    #    signed. So neither a resolved-KE->heat closure nor a total-energy law
    #    holds across these sources -> a static ["energy"] claim is FALSE ->
    #    conserves=["none"].
    #  - config.dttke_use_intrinsic=True switches to the irreversible
    #    (c-ubm)*gwut conversion; config.do_energy_conservation=True (C.-C. Chen
    #    fixer, default OFF) forces the discrete air-column momentum+dse
    #    finite-step residual to zero (a corrective below-source redistribution,
    #    NOT a physical accounting of the wave/frontal source or boundary
    #    fluxes). Neither is the shipped default.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "McFarlane (1987) / Lindzen (1981) / Beres (2004); E3SM EAM "
        "gw_common.F90 (gw_prof, gw_drag_prof, momentum_energy_conservation), "
        "gw_oro.F90, gw_front.F90, gw_convect.F90"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_gwd_e3sm_cam.py: rest state -> "
        "zero tendency; drag decelerates a westerly jet; "
        "test_gwd_damping_factor_never_amplifies_stress (mi>=0, stress "
        "monotone); do_energy_conservation closes the column energy budget"
    ),
}
# Default critical-Froude tuning + orographic tendency cap (scheme defaults).
_DCA_DEFAULT = 0.1
_TNDMAX_ORO_PER_DAY = 500.0



# ---------------------------------------------------------------------------
# gw_prof: background profiles
# ---------------------------------------------------------------------------

def gw_prof(
    t: jax.Array,
    pmid: jax.Array,
    pint: jax.Array,
    cpair: float,
    rair: float,
    gravit: float,
    n2min: float,
):
    """Background interface density and Brunt-Vaisala frequency.

    Faithful to ``gw_prof`` (gw_common.F90).

    Parameters
    ----------
    t : (ncol, nlev)      midpoint temperature [K] (k=0 top, k=nlev-1 sfc).
    pmid : (ncol, nlev)   midpoint pressure [Pa].
    pint : (ncol, nlev+1) interface pressure [Pa] (k=0 top .. k=nlev sfc).
    cpair, rair, gravit, n2min : scalars.

    Returns
    -------
    rhoi : (ncol, nlev+1) interface density [kg/m^3].
    ti   : (ncol, nlev+1) interface temperature [K].
    nm   : (ncol, nlev)   midpoint Brunt-Vaisala frequency [1/s].
    ni   : (ncol, nlev+1) interface Brunt-Vaisala frequency [1/s].
    """
    ncol, nlev = t.shape

    # Top interface (k=0): isothermal-above assumption -> ti = t[0].
    ti_top = t[:, 0]
    rhoi_top = pint[:, 0] / (rair * ti_top)
    ni_top = jnp.sqrt(gravit * gravit / (cpair * ti_top))

    # Interior interfaces k = 1 .. nlev-1 (between midpoints k-1 and k).
    # ti[k] = 0.5*(t[k-1] + t[k]); dtdp = (t[k]-t[k-1])/(pmid[k]-pmid[k-1]).
    # (E3SM: ti(1:pver-1)=midpoint_interp(t); dtdp=(t(k+1)-t(k))/(pmid(k+1)-pmid(k)))
    t_lo = t[:, :-1]   # k-1  (upper)   -> E3SM t(k)
    t_hi = t[:, 1:]    # k    (lower)   -> E3SM t(k+1)
    pm_lo = pmid[:, :-1]
    pm_hi = pmid[:, 1:]
    ti_int = 0.5 * (t_lo + t_hi)                   # (ncol, nlev-1)
    rhoi_int = pint[:, 1:-1] / (rair * ti_int)
    dtdp = (t_hi - t_lo) / (pm_hi - pm_lo)
    n2 = gravit * gravit / ti_int * (1.0 / cpair - rhoi_int * dtdp)
    ni_int = jnp.sqrt(jnp.maximum(n2min, n2))      # (ncol, nlev-1)

    # Bottom interface (k=nlev): ti = t[nlev-1]; ni = ni[nlev-1] (interior top).
    ti_bot = t[:, -1]
    rhoi_bot = pint[:, -1] / (rair * ti_bot)
    ni_bot = ni_int[:, -1]                          # E3SM ni(pver)=ni(pver-1)

    ti = jnp.concatenate([ti_top[:, None], ti_int, ti_bot[:, None]], axis=1)
    rhoi = jnp.concatenate(
        [rhoi_top[:, None], rhoi_int, rhoi_bot[:, None]], axis=1
    )
    ni = jnp.concatenate([ni_top[:, None], ni_int, ni_bot[:, None]], axis=1)

    # Midpoint nm = midpoint_interp(ni): nm[k] = 0.5*(ni[k]+ni[k+1]).
    nm = 0.5 * (ni[:, :-1] + ni[:, 1:])             # (ncol, nlev)

    return rhoi, ti, nm, ni


# ---------------------------------------------------------------------------
# gw_oro_src: McFarlane orographic source
# ---------------------------------------------------------------------------

def gw_oro_src(
    u: jax.Array,
    v: jax.Array,
    t: jax.Array,
    sgh: jax.Array,
    pmid: jax.Array,
    pint: jax.Array,
    dpm: jax.Array,
    zm: jax.Array,
    nm: jax.Array,
    rair: float,
    kwv: float,
    fcrit2: float,
    oro_min_h: float,
    oro_min_wind: float,
):
    """McFarlane (1987) orographic source (single c=0 wave).

    Faithful to ``gw_oro_src`` (gw_oro.F90).  Returns the launched stress
    profile (ncol, nlev+1), source/tend levels, projection unit vectors,
    and the projected winds (ubm, ubi).  The depth-averaged source region
    selection uses a smooth mask so the result stays differentiable; the
    selection threshold ``hdsp > sqrt(zm[k]*zm[k+1])`` is the same as
    E3SM (a sharp inequality there, but its sub-gradient is benign).

    Index: k=0 top, k=nlev-1 surface.
    """
    ncol, nlev = u.shape
    hdsp = 2.0 * sgh                                    # (ncol,)
    inv_rt = 1.0 / (rair * t)
    rho_mid = pmid * inv_rt                             # pmid/(rair*t)

    # --- E3SM source-region selection (gw_oro.F90) ----------------------
    # 1-based E3SM: start with k=pver always included, src_level=pver-1.
    # Loop kk=pver-1..pver/2: include layer kk (and set src_level=kk-1) if
    # hdsp > sqrt(zm(kk)*zm(kk+1)).  In 0-based top-down indexing the surface
    # midpoint is index nlev-1; E3SM midpoint kk -> index kk-1; the pair
    # zm(kk)*zm(kk+1) -> zm[kk-1]*zm[kk].  The bottom-half loop kk=pver-1..pver/2
    # spans 0-based indices nlev-2 .. nlev//2-1.
    k_idx = jnp.arange(nlev)
    # geometric-mean height at the lower interface of midpoint index i:
    # gm[i] = sqrt(zm[i-1]*zm[i]) for i>=1 (E3SM pair zm(kk)*zm(kk+1) at kk-1).
    gm = jnp.sqrt(jnp.abs(zm[:, :-1] * zm[:, 1:]))     # (ncol, nlev-1): index i->pair(i,i+1)
    # For 0-based midpoint index i in [nlev//2-1, nlev-2], E3SM tests
    # hdsp > sqrt(zm[i]*zm[i+1]) == gm[i].
    lo = (nlev // 2) - 1
    in_loop = (k_idx >= lo) & (k_idx <= nlev - 2)      # (nlev,)
    gm_full = jnp.concatenate([gm, gm[:, -1:]], axis=1)  # pad: index nlev-1 unused
    penetrates = hdsp[:, None] > gm_full               # (ncol, nlev)
    # include[i]: surface (nlev-1) always; loop levels if penetrates.
    include = jnp.where(
        k_idx[None, :] == (nlev - 1),
        True,
        in_loop[None, :] & penetrates,
    )                                                   # (ncol, nlev) bool
    w = include.astype(u.dtype) * dpm                   # dp-weights

    # src_level (interface index 0..nlev): smallest included midpoint index
    # minus... E3SM sets src_level=kk-1 (interface above the topmost included
    # midpoint).  Topmost included midpoint index = min index where include.
    big = nlev
    top_inc = jnp.min(
        jnp.where(include, k_idx[None, :], big), axis=1
    )                                                   # (ncol,) midpoint idx
    # interface above that midpoint = same index (top-down: interface i sits
    # above midpoint i).  E3SM: src_level = (topmost kk) - 1 (1-based) ->
    # 0-based interface index = top_inc.
    src_level = top_inc.astype(jnp.int32)               # interface idx

    # E3SM dpsrc = pint(pver) - pint(src_level): the continuous pressure
    # interval from the surface interface to the source-top interface
    # (gw_oro.F90:139).  For monotone columns this equals sum(included dpm)
    # but the pressure-interval form is exactly E3SM (codex iter-1 #9).
    pint_at_src = jnp.take_along_axis(
        pint, src_level[:, None], axis=1
    )[:, 0]                                              # (ncol,)
    dpsrc = pint[:, nlev] - pint_at_src
    dpsrc = jnp.where(dpsrc > 0.0, dpsrc, 1.0)
    rsrc = jnp.sum(w * rho_mid, axis=1) / dpsrc
    usrc = jnp.sum(w * u, axis=1) / dpsrc
    vsrc = jnp.sum(w * v, axis=1) / dpsrc
    nsrc = jnp.sum(w * nm, axis=1) / dpsrc

    mag = jnp.sqrt(usrc * usrc + vsrc * vsrc)
    safe = mag > 0.0
    xv = jnp.where(safe, usrc / jnp.where(safe, mag, 1.0), 0.0)
    yv = jnp.where(safe, vsrc / jnp.where(safe, mag, 1.0), 0.0)

    # Project winds onto source direction.
    ubm = u * xv[:, None] + v * yv[:, None]             # (ncol, nlev)
    ubi = _interface_proj(ubm, mag)                     # (ncol, nlev+1)

    ubi_sfc = ubi[:, -1]                                # = mag
    # McFarlane c=0 launch stress (gw_oro.F90).
    oroko2 = 0.5 * kwv
    sghmax = fcrit2 * (ubi_sfc / jnp.where(nsrc > 0.0, nsrc, 1.0)) ** 2
    tauoro = (
        oroko2 * jnp.minimum(hdsp ** 2, sghmax) * rsrc * nsrc * ubi_sfc
    )
    active = (ubi_sfc > oro_min_wind) & (hdsp > oro_min_h)
    tauoro = jnp.where(active, tauoro, 0.0)             # (ncol,)
    # Where orographic term is zero, E3SM sets src_level = pver (no waves).
    src_level = jnp.where(active, src_level, nlev).astype(jnp.int32)

    # Launch stress fills interfaces k = src_level .. nlev (E3SM:
    # tau(:,0,k)=tauoro for src_level<=k).
    iface_idx = jnp.arange(nlev + 1)
    fill = (iface_idx[None, :] >= src_level[:, None]).astype(u.dtype)
    tau0 = (tauoro[:, None] * fill)[:, None, :]         # (ncol, 1, nlev+1)

    # E3SM: tend_level = pver (tendencies allowed all the way to bottom).
    tend_level = jnp.full((ncol,), nlev, dtype=jnp.int32)

    c = jnp.zeros((ncol, 1), dtype=u.dtype)            # single c=0 wave
    return tau0, src_level, tend_level, xv, yv, c, ubm, ubi


def _interface_proj(ubm: jax.Array, mag_sfc: jax.Array) -> jax.Array:
    """Interface projection of the midpoint projected wind (E3SM convention).

    ubi[0] = ubm[0] (top); ubi[k] = 0.5*(ubm[k-1]+ubm[k]) for interior;
    ubi[nlev] = mag_sfc (source magnitude at surface).
    """
    ncol, nlev = ubm.shape
    ubi_top = ubm[:, 0:1]
    ubi_int = 0.5 * (ubm[:, :-1] + ubm[:, 1:])         # (ncol, nlev-1)
    ubi_bot = mag_sfc[:, None]
    return jnp.concatenate([ubi_top, ubi_int, ubi_bot], axis=1)


# ---------------------------------------------------------------------------
# gw_cm_src: frontal source
# ---------------------------------------------------------------------------

def _front_fav(pgwv: int, dc: float, c0: float, taubgnd: float, dtype,
               dca: float = _DCA_DEFAULT):
    """Average Gaussian over each phase-speed bin (E3SM gw_front_init).

    ``dca`` is the sub-bin c-grid spacing [m/s] for the quadrature
    (E3SM ``gw_front.F90`` ``dca``; default 0.1).

    Returns ``fav`` of length ``2*pgwv+1`` (index 0 = wave -pgwv).
    """
    if pgwv == 0:
        return jnp.array([0.0], dtype=dtype)  # wavenumber 0 prohibited
    # cref(l) = l*dc for l = -pgwv..pgwv
    ls = jnp.arange(-pgwv, pgwv + 1)
    cref = ls.astype(dtype) * dc
    # Fortran nint = round half away from zero (Python round is banker's
    # rounding -> differs on exact .5 ties; codex iter-1 #8).
    n_sub = int(math.floor(dc / dca + 0.5)) - 1
    cmn = cref - 0.5 * dc
    cmx = cref + 0.5 * dc
    fav = 0.5 * dca * (jnp.exp(-(cmn / c0) ** 2) + jnp.exp(-(cmx / c0) ** 2))
    if n_sub > 0:
        ns = jnp.arange(1, n_sub + 1).astype(dtype)
        # sum over interior sub-intervals
        contrib = dca * jnp.exp(-(((cmn[:, None] + ns[None, :] * dca)) / c0) ** 2)
        fav = fav + jnp.sum(contrib, axis=1)
    fav = fav / dc
    fav = taubgnd * fav
    # prohibit wavenumber 0
    fav = fav.at[pgwv].set(0.0)
    return fav


def gw_cm_src(
    u: jax.Array,
    v: jax.Array,
    frontgf: jax.Array,
    pgwv: int,
    dc: float,
    c0: float,
    taubgnd: float,
    frontgfc: float,
    kbot: int,
    kfront: int,
    dca: float = _DCA_DEFAULT,
):
    """Frontal source (E3SM gw_cm_src).

    Launches a uniform Gaussian spectrum at LAUNCH interface ``kbot`` where
    the frontogenesis function ``frontgf`` exceeds ``frontgfc`` at the
    TRIGGER level ``kfront`` (E3SM tests the trigger at ``kfront`` ~600 hPa
    but launches at ``kbot`` ~500 hPa — they differ; codex iter-1 #2).
    Index k=0 top.

    Returns the launched stress (ncol, 2*pgwv+1, nlev+1), src/tend levels,
    projection vectors, projected winds, and phase speeds c (ncol, 2*pgwv+1).
    """
    ncol, nlev = u.shape
    # Spectrum has ``nwav = 2*pgwv + 1`` phase-speed bins; the per-wave axis
    # is built directly from ``pgwv`` (see the ``(ncol, nwav)`` shape comments).
    # kbot / kfront are 0-d traced integers (jit-safe).  All level access uses
    # dynamic gathers / a one-hot launch mask so nothing requires a Python int
    # (codex iter-2 #1 — int(argmin) on traced pressure broke under jit).
    kbot_i = jnp.asarray(kbot, dtype=jnp.int32)
    kfront_i = jnp.asarray(kfront, dtype=jnp.int32)
    klo_i = jnp.maximum(kbot_i - 1, 0)

    # Source-region winds = average of midpoints straddling kbot interface.
    # E3SM: usrc = 0.5*(u(:,kbot+1)+u(:,kbot)). With 0-based top-down, the
    # midpoints adjacent to interface kbot are (kbot-1, kbot).
    gather = lambda a, idx: jnp.take(a, idx, axis=1)   # (ncol,) dynamic gather
    usrc = 0.5 * (gather(u, klo_i) + gather(u, kbot_i))
    vsrc = 0.5 * (gather(v, klo_i) + gather(v, kbot_i))
    mag = jnp.sqrt(usrc * usrc + vsrc * vsrc)
    safe = mag > 0.0
    xv = jnp.where(safe, usrc / jnp.where(safe, mag, 1.0), 0.0)
    yv = jnp.where(safe, vsrc / jnp.where(safe, mag, 1.0), 0.0)

    ubm = u * xv[:, None] + v * yv[:, None]
    ubi = _interface_proj(ubm, mag)

    fav = _front_fav(pgwv, dc, c0, taubgnd, u.dtype, dca)   # (nwav,)
    # E3SM gw_front.F90:169 — launch_wave = frontgf(:,kfront) > frontgfc.
    launch = gather(frontgf, kfront_i) > frontgfc      # (ncol,) tested at kfront
    tau_launch = jnp.where(launch[:, None], fav[None, :], 0.0)  # (ncol, nwav)
    # Place launched stress at interface kbot via a one-hot mask (jit-safe).
    iface_onehot = (jnp.arange(nlev + 1) == kbot_i).astype(u.dtype)  # (nlev+1,)
    tau0 = tau_launch[:, :, None] * iface_onehot[None, None, :]

    # Phase speeds: cref + |ubi[kbot]| (E3SM gw_cm_src).
    ls = jnp.arange(-pgwv, pgwv + 1).astype(u.dtype)
    cref = ls * dc
    ubi_kbot = jnp.take(jnp.abs(ubi), kbot_i, axis=1)  # (ncol,)
    c = cref[None, :] + ubi_kbot[:, None]

    src_level = jnp.full((ncol,), kbot_i, dtype=jnp.int32)
    tend_level = jnp.full((ncol,), kbot_i, dtype=jnp.int32)
    return tau0, src_level, tend_level, xv, yv, c, ubm, ubi


# ---------------------------------------------------------------------------
# gw_beres_src: Beres (2004) convective source
# ---------------------------------------------------------------------------

def build_stand_in_mfcc(pgwv: int, dc: float, beres: E3SMBeresConfig, dtype):
    """Documented analytic STAND-IN ``mfcc`` mean-flux lookup table.

    *** NOT bit-faithful to Beres (2004). ***  The real E3SM source spectrum
    is an offline-generated lookup table ``mfcc(maxh, -maxuh:maxuh, -pgwv:pgwv)``
    stored in the ``gw_drag_file`` netcdf (``newmfspectra*.nc``, default
    namelist name ``Beres04_file``; variable name ``mfcc``; E3SM expects
    ``maxh = 20`` heating-depth bins [1..20 km], ``2*maxuh+1 = 81`` mean-wind
    bins [-40..40 m/s], and ``2*ngwv_file+1 = 81`` phase-speed bins with
    ``ngwv_file = 40``, sliced to ``-pgwv:pgwv`` at load via
    ``start=[1,1,ngwv_file-pgwv+1]``).  That table is NOT bundled here.

    This stand-in reproduces the *algorithm* with a clearly-labelled,
    physically-plausible placeholder so the convective scheme RUNS and is
    testable: a phase-speed Gaussian (width ``mfcc_c0``, peak ``mfcc_peak``)
    that grows monotonically with heating depth (factor
    ``1 + mfcc_hdepth_growth*(h-1)``) and is independent of mean wind ``uh``
    (the real table's uh-dependence is a Doppler shift the algorithm applies
    separately via the ground-relative cshift).  To recover bit-faithfulness,
    set ``beres.use_stand_in_table=False`` and pass the real ``mfcc_table``
    to ``gw_beres_src`` / ``e3sm_cam_gwd``.

    Returns
    -------
    mfcc : (maxh, 2*maxuh+1, 2*pgwv+1)
        ``mfcc[h-1, j, :]`` is the source spectrum for heating-depth index
        ``h`` (1-based -> ``h-1``) and mean-wind index ``uh`` (``j = uh +
        maxuh``).  Indexed exactly as E3SM ``mfcc(NINT(hdepth), NINT(uh), :)``
        with the 1-based ``hdepth`` / ``-maxuh:maxuh`` ``uh`` shifted to
        0-based Python.
    """
    maxh = beres.maxh
    maxuh = beres.maxuh
    # Phase-speed bin centres cref(l) = l*dc, l=-pgwv..pgwv (the per-wave axis
    # has length ``nwav = 2*pgwv + 1``; see the ``(nwav,)`` shape comments).
    ls = jnp.arange(-pgwv, pgwv + 1, dtype=dtype)
    cref = ls * dc                                          # (nwav,)
    base_spec = beres.mfcc_peak * jnp.exp(-(cref / beres.mfcc_c0) ** 2)
    base_spec = base_spec.at[pgwv].set(0.0)                 # prohibit c=0 wave
    h_idx = jnp.arange(1, maxh + 1, dtype=dtype)            # 1..maxh (km)
    h_growth = 1.0 + beres.mfcc_hdepth_growth * (h_idx - 1.0)  # (maxh,)
    # (maxh, nwav): depth-scaled spectrum.
    spec_h = h_growth[:, None] * base_spec[None, :]
    # uh dependence: amplitude factor 1 + mfcc_uh_slope*uh, where uh ranges
    # over the table columns [-maxuh, maxuh].  Default slope 0 -> uh-independent
    # (back-compatible); a non-zero slope makes the stand-in exercise the
    # uh table-column lookup (used by the oracle to validate uh_idx/uh_col).
    uh_vals = jnp.arange(-maxuh, maxuh + 1, dtype=dtype)    # (2*maxuh+1,)
    uh_factor = 1.0 + beres.mfcc_uh_slope * uh_vals         # (2*maxuh+1,)
    # mfcc[h, j, l] = spec_h[h, l] * uh_factor[j]
    mfcc = (spec_h[:, None, :] * uh_factor[None, :, None]).astype(dtype)
    return mfcc


def gw_beres_src(
    u: jax.Array,
    v: jax.Array,
    netdt: jax.Array,
    zm: jax.Array,
    lat: jax.Array,
    k700: int,
    mfcc: jax.Array,
    pgwv: int,
    dc: float,
    beres: E3SMBeresConfig,
):
    """Beres (2004) convective gravity-wave source (``gw_beres_src``).

    Faithful to ``components/eam/src/physics/cam/gw_convect.F90``.  Builds
    the launched phase-speed spectrum from the deep-convective heating
    profile: the heating depth ``hdepth`` (first continuously-positive
    ``netdt`` range from the surface, capped at ``z_heat_max``) and the
    maximum heating rate ``q0`` index the mean-flux lookup ``mfcc(hdepth,
    uh, c)``; the spectrum is Doppler-shifted to be ground-relative by the
    cell speed ``CS``, scaled by ``q0^2/AL``, and critical-level filtered
    (zeroed for ``Umin <= c <= Umax``).

    Index: k=0 top, k=nlev-1 surface; interfaces 0..nlev.

    Parameters
    ----------
    u, v : (ncol, nlev)   midpoint winds [m/s].
    netdt : (ncol, nlev)  convective heating rate [K/s] (>0 where heating).
    zm : (ncol, nlev)     midpoint heights [m].
    lat : (ncol,)         latitudes [rad].
    k700 : int            0-based midpoint index nearest 700 hPa (source-wind
                          level).  Traced int32 ok (consumed via jnp.take).
    mfcc : (maxh, 2*maxuh+1, 2*pgwv+1)  mean-flux lookup table.
    pgwv, dc, beres : spectral half-width, bin width, Beres sub-config.

    Returns
    -------
    tau0    : (ncol, nwav, nlev+1) launched stress (set at maxi interface).
    src_level, tend_level : (ncol,) int interface indices (= maxi).
    xv, yv  : (ncol,) source unit vectors.
    c       : (ncol, nwav) phase speeds (= cref).
    ubm, ubi: projected winds (ncol,nlev)/(ncol,nlev+1).
    hdepth  : (ncol,) heating depth [km] (diagnostic).
    maxq0   : (ncol,) max heating rate [K/day] (diagnostic).
    """
    ncol, nlev = u.shape
    nwav = 2 * pgwv + 1
    maxuh = beres.maxuh
    maxh = beres.maxh
    dtype = u.dtype
    k700_i = jnp.asarray(k700, dtype=jnp.int32)

    # --- Source-wind direction from the 700 hPa winds (E3SM uses u(:,k700)).
    u700 = jnp.take(u, k700_i, axis=1)                     # (ncol,)
    v700 = jnp.take(v, k700_i, axis=1)
    mag700 = jnp.sqrt(u700 * u700 + v700 * v700)
    safe = mag700 > 0.0
    xv = jnp.where(safe, u700 / jnp.where(safe, mag700, 1.0), 0.0)
    yv = jnp.where(safe, v700 / jnp.where(safe, mag700, 1.0), 0.0)

    ubm = u * xv[:, None] + v * yv[:, None]                # (ncol, nlev)
    ubi = _interface_proj(ubm, mag700)                     # (ncol, nlev+1)

    # --- Heating depth.  E3SM scans k=pver..1 (surface upward); mini = first
    # level (from bottom) with netdt>0, maxi = first level above mini where
    # heating stops; either bounded by z>=z_heat_max.  In 0-based top-down
    # indexing the surface midpoint is nlev-1; "from the bottom" = decreasing
    # index.  We reproduce the same first-continuously-positive-range logic
    # vectorially (no Python control flow on traced values).
    # Precondition (codex beres-1 #4): E3SM's scan assumes the heating run
    # either terminates (a non-heating level above) or hits the z_heat_max cap
    # before the model top.  If the model top is BELOW z_heat_max and a positive
    # heating run reaches the top without terminating, E3SM never sets maxi and
    # the column launches no waves; our `found = mini_set & maxi_set` mask
    # reproduces that (such a column gets hdepth=0, zero source) — this is the
    # same E3SM precondition, not a JAX-specific behaviour.
    heat = netdt > 0.0                                     # (ncol, nlev) bool
    # Bottom level of the heating range (mini): the lowest (largest-index)
    # interior level that is both heating and in-window AND whose run of
    # heating reaches the surface-most heating cell continuously.  E3SM finds
    # the first netdt>0 from the bottom, OR the first z>=z_heat_max.  Build a
    # bottom-up scan via lax.scan over k = nlev-1 .. 0.
    big = nlev

    def _depth_step(carry, k):
        # carry: (mini, maxi, mini_set, maxi_set) per column, int/bool arrays.
        mini, maxi, mini_set, maxi_set = carry
        k = k.astype(jnp.int32)
        z_k = zm[:, k]
        heat_k = heat[:, k]
        out_window = z_k >= beres.z_heat_max
        # Phase 1: locate mini (bottom of heating range).
        # mini not yet set:
        set_mini_now = (~mini_set) & (out_window | heat_k)
        new_mini = jnp.where(set_mini_now, k, mini)
        # if out_window triggers it, E3SM also sets maxi=k immediately.
        set_maxi_via_window = set_mini_now & out_window
        # Phase 2: mini set, maxi not yet set -> first non-heating (or window).
        in_phase2 = mini_set & (~maxi_set)
        set_maxi_now = in_phase2 & (out_window | (~heat_k))
        new_maxi = jnp.where(
            set_maxi_via_window, k,
            jnp.where(set_maxi_now, k, maxi),
        )
        new_mini_set = mini_set | set_mini_now
        new_maxi_set = maxi_set | set_maxi_via_window | set_maxi_now
        return (new_mini, new_maxi, new_mini_set, new_maxi_set), None

    z0 = (
        jnp.full((ncol,), big, dtype=jnp.int32),   # mini
        jnp.full((ncol,), big, dtype=jnp.int32),   # maxi
        jnp.zeros((ncol,), dtype=bool),            # mini_set
        jnp.zeros((ncol,), dtype=bool),            # maxi_set
    )
    (mini, maxi, mini_set, maxi_set), _ = jax.lax.scan(
        _depth_step, z0, jnp.arange(nlev - 1, -1, -1, dtype=jnp.int32)
    )
    # E3SM leaves mini/maxi = 0 when no heating found -> hdepth = 0.  Our
    # "unset" sentinel is `big`; map unset columns to a benign in-range level
    # so the gathers below are valid, but force hdepth = 0 there.
    found = mini_set & maxi_set
    mini_g = jnp.where(found, mini, nlev - 1).astype(jnp.int32)
    maxi_g = jnp.where(found, maxi, nlev - 1).astype(jnp.int32)

    zm_mini = jnp.take_along_axis(zm, mini_g[:, None], axis=1)[:, 0]
    zm_maxi = jnp.take_along_axis(zm, maxi_g[:, None], axis=1)[:, 0]
    # E3SM hdepth = (zm(maxi) - zm(mini))/1000.  In our 0-based top-down
    # indexing maxi_g is the TOP of the heating range (smaller index, higher
    # altitude) and mini_g the BOTTOM (larger index, lower altitude), so
    # zm[maxi_g] > zm[mini_g] and hdepth > 0.
    hdepth = jnp.where(found, (zm_maxi - zm_mini) / 1000.0, 0.0)  # km
    hdepth = jnp.minimum(hdepth, float(maxh))
    hdepth = hdepth * beres.hdepth_scaling_factor

    # --- Heating-range mean wind uh and max heating q0.  E3SM averages over
    # the inclusive index range [maxi, mini] (maxi above, mini below).
    k_idx = jnp.arange(nlev)[None, :]                      # (1, nlev)
    in_range = (
        (k_idx >= maxi_g[:, None]) & (k_idx <= mini_g[:, None]) & found[:, None]
    )                                                      # (ncol, nlev) bool
    n_in = jnp.maximum(jnp.sum(in_range, axis=1), 1)       # = mini-maxi+1
    q0 = jnp.max(jnp.where(in_range, netdt, -jnp.inf), axis=1)
    q0 = jnp.where(found, q0, 0.0)
    maxq0 = q0 * 24.0 * 3600.0  # coeff-ok: hours/day x s/hr -> K/day diagnostic
    q0 = q0 * beres.cf                                     # convert to source amp

    # Cell speed CS (integer, ground-relative shift).
    ubm700 = jnp.take(ubm, k700_i, axis=1)                 # (ncol,)
    cs_f = jnp.sign(ubm700) * jnp.maximum(
        jnp.abs(ubm700) - beres.storm_speed_min, 0.0
    )
    cs = jnp.trunc(cs_f).astype(jnp.int32)                 # int(sign(...))

    uh = jnp.sum(
        jnp.where(in_range, ubm, 0.0), axis=1
    ) / n_in.astype(dtype)
    uh = uh - cs.astype(dtype)
    uh = jnp.clip(uh, -float(maxuh), float(maxuh))

    # Speeds for critical-level filtering (min/max projected wind in range).
    umin = jnp.where(
        in_range, ubm, jnp.asarray(pgwv * dc, dtype)
    )
    umin = jnp.where(found[:, None], umin, jnp.asarray(pgwv * dc, dtype))
    umin = jnp.min(umin, axis=1)
    umax = jnp.where(
        in_range, ubm, jnp.asarray(-pgwv * dc, dtype)
    )
    umax = jnp.where(found[:, None], umax, jnp.asarray(-pgwv * dc, dtype))
    umax = jnp.max(umax, axis=1)

    # --- Table lookup.  E3SM: tau0 = mfcc(NINT(hdepth), NINT(uh), :).
    # Round-half-away-from-zero like Fortran NINT.  NOTE (codex beres-1 #2):
    # E3SM caps hdepth at maxh BEFORE applying hdepth_scaling_factor and then
    # indexes mfcc(NINT(hdepth),...) WITHOUT a second cap; with a scaling
    # factor > 1 this could push NINT(hdepth) past maxh and read out of bounds
    # (E3SM relies on the namelist keeping it in range).  We instead clip the
    # table ROW index to [1, maxh] (a deliberate, documented safety guard) so
    # the gather is always valid and jit/grad-safe.  For the default
    # hdepth_scaling_factor == 1 this clip is a no-op and the lookup is
    # bit-identical to E3SM.
    h_idx = _nint(hdepth)                                  # 1..maxh (km)
    h_idx = jnp.clip(h_idx, 1, maxh)
    h_row = (h_idx - 1).astype(jnp.int32)                  # 0-based table row
    uh_idx = _nint(uh)
    uh_idx = jnp.clip(uh_idx, -maxuh, maxuh)
    uh_col = (uh_idx + maxuh).astype(jnp.int32)            # 0-based table col
    # gather per column: mfcc[h_row, uh_col, :] -> (ncol, nwav)
    tau0 = mfcc[h_row, uh_col, :]                          # (ncol, nwav)

    # Doppler shift so the spectrum is ground-relative: cshift by -nint(CS/dc).
    shift = -_nint(cs.astype(dtype) / dc).astype(jnp.int32)  # (ncol,)
    tau0 = _cshift_rows(tau0, shift)

    # Adjust magnitude by q0^2/AL.
    tau0 = tau0 * (q0 * q0 / beres.al)[:, None]

    # Critical-level filtering: zero tau0 for Umini <= l <= Umaxi.
    umini = jnp.maximum(_nint(umin / dc), -pgwv).astype(jnp.int32)  # (ncol,)
    umaxi = jnp.minimum(_nint(umax / dc), pgwv).astype(jnp.int32)
    ls = jnp.arange(-pgwv, pgwv + 1)[None, :]              # (1, nwav) wave index
    crit_mask = (
        (umaxi[:, None] > umini[:, None])
        & (ls >= umini[:, None])
        & (ls <= umaxi[:, None])
    )                                                      # (ncol, nwav)
    tau0 = jnp.where(crit_mask, 0.0, tau0)

    # Only launch where hdepth >= hdepth_min and |lat| < pi/2 (E3SM gate).
    launch = (hdepth >= beres.hdepth_min_km) & (jnp.abs(lat) < (0.5 * math.pi))
    tau_launch = jnp.where(launch[:, None], tau0, 0.0)     # (ncol, nwav)

    # Place launched stress at interface maxi via one-hot (jit-safe), and set
    # src/tend levels = maxi.  E3SM uses the 1-based midpoint index ``maxi``
    # directly as an INTERFACE index in tau(:,:,maxi) / src_level=maxi.  In our
    # 0-based top-down convention E3SM's interface index ``maxi`` is the
    # interface BELOW our midpoint ``maxi_g`` (= maxi-1), i.e. our interface
    # index ``maxi_g + 1``.  Non-launching columns keep this valid index but
    # carry zero tau.
    src_iface = (maxi_g + 1).astype(jnp.int32)             # (ncol,)
    iface_idx = jnp.arange(nlev + 1)[None, :]              # (1, nlev+1)
    iface_onehot = (iface_idx == src_iface[:, None]).astype(dtype)
    tau0_full = tau_launch[:, :, None] * iface_onehot[:, None, :]

    src_level = src_iface
    tend_level = src_iface
    # Phase speeds: just the reference speeds (E3SM c = spread(cref,...)).
    cref = (jnp.arange(-pgwv, pgwv + 1, dtype=dtype) * dc)[None, :]
    c = jnp.broadcast_to(cref, (ncol, nwav))
    return tau0_full, src_level, tend_level, xv, yv, c, ubm, ubi, hdepth, maxq0


def _nint(x: jax.Array) -> jax.Array:
    """Fortran NINT: round half away from zero (int32)."""
    return jnp.sign(x) * jnp.floor(jnp.abs(x) + 0.5)


def _cshift_rows(a: jax.Array, shift: jax.Array) -> jax.Array:
    """Per-row circular shift matching Fortran ``cshift(a, shift)``.

    Fortran ``cshift(array, SHIFT)`` moves element ``i`` to ``i-SHIFT``
    (positive SHIFT = leftward/toward lower index, wrapping).  ``a`` is
    (ncol, n); ``shift`` is (ncol,) int.  Implemented with a gather so it is
    jit/vmap/grad-safe.
    """
    ncol, n = a.shape
    cols = jnp.arange(n)[None, :]                          # (1, n)
    src = jnp.mod(cols + shift[:, None], n)                # (ncol, n)
    return jnp.take_along_axis(a, src.astype(jnp.int32), axis=1)


# ---------------------------------------------------------------------------
# gw_drag_prof: the spectral solver (momentum path)
# ---------------------------------------------------------------------------

def gw_drag_prof(
    tau_launch: jax.Array,
    c: jax.Array,
    src_level: jax.Array,
    tend_level: jax.Array,
    t: jax.Array,
    ti: jax.Array,
    piln: jax.Array,
    rhoi: jax.Array,
    nm: jax.Array,
    ni: jax.Array,
    ubm: jax.Array,
    ubi: jax.Array,
    xv: jax.Array,
    yv: jax.Array,
    dpm: jax.Array,
    rdpm: jax.Array,
    lat: jax.Array,
    effgw: float,
    dt: float,
    cfg: E3SMCAMConfig,
    orographic_only: bool,
    do_taper: bool = False,
    alpha_iface: jax.Array | None = None,
):
    """Stress-profile + tendency solver, faithful to ``gw_drag_prof``.

    Index: k=0 top (= ktop), k=nlev-1 surface; interfaces 0..nlev.

    Parameters
    ----------
    tau_launch : (ncol, nwav, nlev+1) launched stress (source level set).
    c          : (ncol, nwav) phase speeds.
    src_level, tend_level : (ncol,) int interface indices.
    t, nm, ubm, dpm, rdpm : (ncol, nlev) midpoint fields.
    ti, piln, rhoi, ni, ubi : (ncol, nlev+1) interface fields.
    xv, yv : (ncol,) source unit vectors.
    alpha_iface : (ncol, nlev+1) or None
        Height-dependent Newtonian-cooling coefficient [1/s] at interfaces
        (E3SM ``alpha(0:pver)``).  ``None`` -> uniform ``cfg.alpha_newtonian``
        at every interface (the default clean-oracle behaviour).

    Returns
    -------
    tau   : (ncol, nwav, nlev+1) final stress profile.
    utgw  : (ncol, nlev) zonal wind tendency [m/s^2].
    vtgw  : (ncol, nlev) meridional wind tendency [m/s^2].
    gwut  : (ncol, nlev, nwav) per-wave tendency (for ediff).
    """
    ncol, nwav, _ = tau_launch.shape
    nlev = ubm.shape[1]
    # E3SM keys the orographic-only code paths (the analytic (u-c)^3 tendency
    # cap and the deferred efficiency/taper application) on the global
    # ``orographic_only`` flag, NOT on the wave count.  Pass it explicitly so
    # a single-wave non-orographic call is not silently routed through the
    # orographic branch (codex iter-1 finding #7).
    is_oro = orographic_only

    effkwv = cfg.kwv * cfg.fcrit2
    rog = constants.R_d / constants.g
    gravit = constants.g
    kwv = cfg.kwv
    # E3SM: tndmax = 400/86400 for spectral (multi-source), 500/86400 when
    # orographic_only (gw_common.F90:156).  (codex iter-1 finding #4)
    tndmax_per_day = _TNDMAX_ORO_PER_DAY if orographic_only else cfg.tndmax_per_day
    tndmax = tndmax_per_day / 86400.0
    # Newtonian cooling: a height profile (E3SM alpha(k)) when supplied, else
    # a single uniform value at every interface.  alpha_arr is (ncol, nlev+1).
    if alpha_iface is None:
        alpha_arr = jnp.full((ncol, nlev + 1), cfg.alpha_newtonian, dtype=t.dtype)
    else:
        alpha_arr = alpha_iface

    # Polar taper applied to the tendency (E3SM ptaper = cos(lat) when
    # do_taper, else 1).  (codex iter-1 finding #3)
    ptaper = jnp.cos(lat) if do_taper else jnp.ones((ncol,))

    # interface index array 0..nlev; src_level/tend_level are interface idx.
    # E3SM scans k = maxval(src_level)-1 .. ktop (=0), i.e. from the interface
    # just above the source upward to the top. With src_level=nlev (surface
    # interface) the first computed interface is nlev-1.

    # ---- Bottom-up stress profile via lax.scan over interface index. ----
    # carry: tau at interface k+1 (below), shape (ncol, nwav).
    # We iterate k from nlev-1 down to 0.
    # scan index j = 0..nlev-1 maps to k = nlev-1-j.

    c_col = c                                   # (ncol, nwav)
    # source-active mask per column at interface k: src_level > k means the
    # interface is ABOVE the source (E3SM `where (src_level > k)`).  At and
    # below the source (k >= src_level) the stress stays at the launched
    # value (the launched stress is uniform there, so it carries through).
    tau_launched_iface = tau_launch              # (ncol, nwav, nlev+1)

    def stress_step(carry, j):
        tau_below = carry                        # (ncol, nwav) at interface k+1
        k = nlev - 1 - j                         # interface index 0..nlev-1
        active = (src_level > k)[:, None]        # (ncol,1)

        ubi_k = ubi[:, k][:, None]               # (ncol,1)
        ubi_kp1 = ubi[:, k + 1][:, None]
        ubmc = ubi_k - c_col                     # (ncol,nwav)  (u-c) at k
        ubmc_below = ubi_kp1 - c_col             # (u-c) at k+1

        same_sign = (ubmc * ubmc_below) > 0.0
        tausat = jnp.abs(
            effkwv * rhoi[:, k][:, None] * ubmc ** 3
            / (2.0 * ni[:, k][:, None])
        )
        tausat = jnp.where(same_sign, tausat, 0.0)
        tausat = jnp.where(tausat <= cfg.taumin, 0.0, tausat)

        # diffusivity: d = max(dback, dscal*dsat) over waves.  alpha is the
        # Newtonian-cooling coefficient at interface k (E3SM alpha(k)).
        alpha_k = alpha_arr[:, k][:, None]       # (ncol,1)
        ni_k = ni[:, k][:, None]
        ti_k = ti[:, k][:, None]
        dsat = (ubmc / ni_k) ** 2 * (
            effkwv * ubmc ** 2 / (rog * ti_k * ni_k) - alpha_k
        )
        dscal = jnp.minimum(1.0, tau_below / (tausat + cfg.taumin))
        d_waves = jnp.where(active, dscal * dsat, -jnp.inf)
        d = jnp.maximum(cfg.dback, jnp.max(d_waves, axis=1, keepdims=True))

        # stress this interface: min(taudmp, tausat).
        # E3SM: wrk = -2*mi*rog*t(k+1)*(piln(k+1)-piln(k)) where t(k+1) is the
        # midpoint below interface k and (piln(k+1)-piln(k)) spans the layer
        # below interface k.  In 0-based top-down: midpoint below interface k
        # is t[k]; the layer below interface k spans interfaces k..k+1, so the
        # piln difference is piln[k+1]-piln[k].
        ubmc2 = jnp.maximum(ubmc ** 2, cfg.ubmc2mn)
        mi = ni_k / (2.0 * kwv * ubmc2) * (alpha_k + ni_k ** 2 / ubmc2 * d)
        # Sign convention: ``mi`` is Im(m), the wave-damping rate; it must be
        # >= 0 so the damping factor exp(-2*mi*...) NEVER amplifies stress
        # upward.  DEFENSIVE floor: ``mi ∝ (alpha + ni^2/ubmc2 * d)`` and in
        # the orographic path ``d = max(dback, dscal*dsat)`` with ``dscal<=1``,
        # so the negative ``-alpha`` inside ``dsat`` is scaled down before being
        # added back to ``+alpha`` and ``mi`` stays >= 0 for every reachable
        # ``(alpha, dback)`` (an empirical sweep confirmed the floor does not
        # bind here).  It guards a HYPOTHETICAL inverted-diffusivity config
        # where ``d`` is forced so negative that ``ni^2/ubmc2 * d < -alpha``,
        # which would make ``exp(-2*mi*...) > 1`` (anti-dissipative); the unit
        # test ``test_gwd_damping_factor_never_amplifies_stress`` exercises that
        # forced regime.  Floor at 0 so the column is always a momentum sink.
        mi = jnp.maximum(mi, 0.0)
        wrk = -2.0 * mi * rog * t[:, k][:, None] * (
            piln[:, k + 1][:, None] - piln[:, k][:, None]
        )
        taudmp = tau_below * jnp.exp(wrk)
        taudmp = jnp.where(taudmp <= cfg.taumin, 0.0, taudmp)
        tau_computed = jnp.minimum(taudmp, tausat)
        # In the source region keep the launched value at interface k.
        tau_k = jnp.where(active, tau_computed, tau_launched_iface[:, :, k])

        return tau_k, tau_k

    # carry init = launched stress at the surface interface (k=nlev).
    tau_src = tau_launched_iface[:, :, nlev]         # (ncol, nwav)
    _, tau_stack = jax.lax.scan(
        stress_step, tau_src, jnp.arange(nlev)
    )
    # tau_stack[j] = tau at interface k=nlev-1-j -> reorder to (ncol,nwav,nlev)
    tau_interior = jnp.transpose(tau_stack[::-1], (1, 2, 0))  # interfaces 0..nlev-1
    tau = jnp.concatenate(
        [tau_interior, tau_src[:, :, None]], axis=2
    )                                                # (ncol, nwav, nlev+1)

    # ---- Top-down tendency via lax.scan over midpoint level. ----
    # E3SM loops k = ktop+1 .. maxval(tend_level). midpoint k uses interfaces
    # (k-1, k) i.e. tau[k] - tau[k-1]. In 0-based top-down: midpoint level m
    # (0..nlev-1) is bounded by interfaces m (above) and m+1 (below).
    # E3SM ubtl = g*(tau(k)-tau(k-1))*rdpm(k): interface k (below) minus
    # interface k-1 (above). For midpoint m -> tau[m+1]-tau[m].
    nm_m = nm                                        # (ncol, nlev)
    ubm_m = ubm

    def tend_step(carry, m):
        tau_state = carry                            # (ncol, nwav, nlev+1) mutable stress
        tau_above = tau_state[:, :, m]               # interface above midpoint m
        tau_below = tau_state[:, :, m + 1]           # interface below midpoint m
        ubtl_raw = gravit * (tau_below - tau_above) * rdpm[:, m][:, None]

        cmu = c_col - ubm_m[:, m][:, None]           # (ncol, nwav)
        if is_oro:
            ubtlsat = effkwv * jnp.abs(cmu) ** 3 / (
                2.0 * rog * t[:, m][:, None] * nm_m[:, m][:, None]
            )
            ubtl = jnp.minimum(ubtl_raw, ubtlsat)
        else:
            ubtl = ubtl_raw
        ubtl = jnp.minimum(ubtl, cfg.umcfac * jnp.abs(cmu) / dt)
        ubtl = jnp.minimum(ubtl, tndmax)

        within = (m < tend_level)[:, None]           # midpoint m below tend
        signed = jnp.sign(cmu) * ubtl
        ptap = ptaper[:, None]                        # (ncol,1)
        if is_oro:
            # E3SM: gwut = sign(ubtl,c-ubm)*effgw*ptaper, but ubt accumulates
            # the UN-tapered, UN-efficiency'd signed tendency; effgw & ptaper
            # are applied once to utgw/vtgw below.
            gwut = jnp.where(within, signed * effgw * ptap, 0.0)
            ubt = jnp.sum(jnp.where(within, signed, 0.0), axis=1)
        else:
            # E3SM: gwut already carries effgw*ptaper; ubt sums gwut.
            gwut = jnp.where(within, signed * effgw * ptap, 0.0)
            ubt = jnp.sum(gwut, axis=1)

        # re-derive stress on the interface below from the bounded tendency.
        tau_below_new = jnp.where(
            within,
            tau_above + ubtl * dpm[:, m][:, None] / gravit,
            tau_below,
        )
        tau_state = tau_state.at[:, :, m + 1].set(tau_below_new)

        if is_oro:
            utgw_m = ubt * xv * effgw * ptaper
            vtgw_m = ubt * yv * effgw * ptaper
        else:
            utgw_m = ubt * xv
            vtgw_m = ubt * yv
        return tau_state, (utgw_m, vtgw_m, gwut)

    tau_state0 = tau
    tau_final, (utgw, vtgw, gwut) = jax.lax.scan(
        tend_step, tau_state0, jnp.arange(nlev)
    )
    # scan stacks along axis 0 = midpoint level -> transpose to (ncol, nlev).
    utgw = utgw.T                                    # (ncol, nlev)
    vtgw = vtgw.T
    gwut = jnp.transpose(gwut, (1, 0, 2))            # (ncol, nlev, nwav)

    return tau_final, utgw, vtgw, gwut


# ---------------------------------------------------------------------------
# Newtonian-cooling height profile (gw_drag.F90)
# ---------------------------------------------------------------------------

# E3SM pre-calculated Newtonian-cooling coefficients (1/day) and the pressure
# levels (hPa) at which they were tabulated (gw_drag.F90:242-281).  This is a
# published DATA table (the radiative-damping profile), not a tunable, so it
# lives here as a named constant rather than in a *Config (akin to the PFT
# lookup tables exempted in surface_params.py).
_ALPHA0_PER_DAY = (
    1.896007, 1.196965, 0.7251356, 0.6397463, 0.5777858, 0.5712274,
    0.6836302, 0.6678557, 0.5683219, 0.4754283, 0.3960519, 0.332022,
    0.2497581, 0.168667, 0.1323903, 0.1257139, 0.1069889, 0.09873954,
    0.09215571, 0.09398635, 0.1061087, 0.1294598, 0.1544743, 0.1648226,
    0.1687332, 0.1691513, 0.1664987, 0.159048, 0.149292, 0.1351563,
    0.1174998, 0.09913579, 0.08300615, 0.0707, 0.0615588, 0.0542623,
    0.0478562, 0.04132157, 0.03454087, 0.02296682, 0.006723819, 0.02164464,
    0.05756261, 0.003844868, 0.02929285, 0.006627098, 0.04558291, 0.02042176,
    0.00000000, 0.005880283, 0.00689498, 0.01343466, 0.00000000, 0.03415992,
    0.02855049, 0.01688839, 0.0272628, 0.02772121, 0.02135626, 0.04863235,
    0.04568304, 0.00000000, 0.009604108, 0.00000000, 0.00000000, 0.00000000,
)
_PALPH_HPA = (
    5.11075e-6, 9.8269e-6, 1.620185e-5, 2.671225e-5, 4.4041e-5, 7.261275e-5,
    1.19719e-4, 1.9738e-4, 3.254225e-4, 5.365325e-4, 8.846025e-4, 0.001458458,
    0.002404575, 0.00397825, 0.006556825, 0.01081382, 0.017898, 0.02955775,
    0.04873075, 0.07991075, 0.1282732, 0.19812, 0.292025, 0.4101675, 0.55347,
    0.73048, 0.9559475, 1.244795, 1.61285, 2.079325, 2.667425, 3.404875,
    4.324575, 5.4654, 6.87285, 8.599725, 10.70705, 13.26475, 16.35175,
    20.05675, 24.479, 29.728, 35.92325, 43.19375, 51.6775, 61.5205, 72.8745,
    85.65715, 100.5147, 118.2503, 139.1154, 163.6621, 192.5399, 226.5132,
    266.4812, 313.5013, 368.818, 433.8952, 510.4553, 600.5242, 696.7963,
    787.7021, 867.1607, 929.6489, 970.5548, 992.5561,
)
# Minimum Newtonian-cooling coefficient (E3SM floor, 1/s); ~10-day damping.
_ALPHA_MIN_PER_S = 1.0e-6


def newtonian_alpha_profile(pint: jax.Array) -> jax.Array:
    """E3SM height-dependent Newtonian-cooling coefficient at interfaces.

    Faithful to gw_drag.F90:315-342: convert ``alpha0`` to 1/s, floor at
    ``1e-6``, convert ``palph`` hPa->Pa, then linearly interpolate (in
    pressure) onto the column interface pressures.  Returns ``alpha`` of
    shape ``(ncol, nlev+1)`` [1/s].

    The interpolation is differentiable (``jnp.interp``); the table itself
    is the published radiative-damping profile.
    """
    palph_pa = jnp.asarray(_PALPH_HPA, dtype=pint.dtype) * 1.0e2  # hPa -> Pa
    alpha0_s = jnp.maximum(
        jnp.asarray(_ALPHA0_PER_DAY, dtype=pint.dtype) / 86400.0,
        _ALPHA_MIN_PER_S,
    )
    # jnp.interp requires increasing xp; palph_pa is already increasing.
    # E3SM lininterp clamps outside the table range to the endpoint values,
    # which is exactly jnp.interp's default left/right behaviour.
    flat = pint.reshape(-1)
    alpha_flat = jnp.interp(flat, palph_pa, alpha0_s)
    return alpha_flat.reshape(pint.shape)


# ---------------------------------------------------------------------------
# gw_ediff + gw_diff_tend: GW-induced eddy diffusion (gw_diffusion.F90)
# ---------------------------------------------------------------------------

def gw_ediff(
    gwut: jax.Array,
    ubm: jax.Array,
    nm: jax.Array,
    c: jax.Array,
    rhoi_kludge: jax.Array,
    pmid: jax.Array,
    rdpm: jax.Array,
    tend_level: jax.Array,
    dt: float,
    gravit: float,
    rair: float,
    ktop: int,
    kbot: int,
    prndl: float,
    egwd_max: float,
):
    """GW-induced effective eddy diffusivity at interfaces (``gw_ediff``).

    Faithful to ``gw_diffusion.F90``: the per-wave diffusivity
    ``prndl*0.5*gwut*(c-ubm)/nm^2`` summed over waves at midpoints,
    interpolated to interfaces (zero at top/bottom), capped at ``egwd_max``,
    and zeroed at/below ``tend_level``.  Returns ``egwdffi`` (ncol, nlev+1).

    Index: k=0 top, k=nlev-1 surface; interfaces 0..nlev.  E3SM operates on
    interfaces ``ktop+1 .. kbot`` (0-based here ktop=0, kbot=kbotbg).

    Notes
    -----
    The E3SM diffusivity is signed by ``gwut*(c-ubm)`` and CAN be negative;
    E3SM only caps it from ABOVE (``min(150, egwdffi)``) and never floors it
    at zero, so we keep the same (a negative diffusivity = up-gradient flux
    is a known feature of the linear GW saturation theory there).
    """
    ncol, nlev = ubm.shape
    # ``gwut`` carries the per-wave axis (length ``nwav = gwut.shape[2]``); the
    # sum over waves below collapses it (see the ``(ncol, nlev, nwav)`` shapes).
    # egwdffm at midpoints: sum over waves.
    cmu = c[:, None, :] - ubm[:, :, None]                  # (ncol, nlev, nwav)
    egwdffm = prndl * 0.5 * jnp.sum(
        gwut * cmu, axis=2
    ) / (nm ** 2)                                          # (ncol, nlev)
    # E3SM accumulates egwdffm over 1-based midpoints k = ktop+1 .. kbot.  A
    # 1-based midpoint k maps to our 0-based midpoint k-1, so E3SM midpoints
    # ktop+1 .. kbot -> our midpoints ktop .. kbot-1.  (kbot is an INTERFACE
    # index, kbotbg.)
    k_idx = jnp.arange(nlev)[None, :]
    in_band = (k_idx >= ktop) & (k_idx <= (kbot - 1))
    egwdffm = jnp.where(in_band, egwdffm, 0.0)

    # Interpolate to interfaces: egwdffi[k] = 0.5*(egwdffm[k-1]+egwdffm[k]) for
    # interior; zero at top and bottom interfaces.  E3SM:
    # egwdffi(:,ktop+1:kbot-1) = midpoint_interp(egwdffm(:,ktop+1:kbot)).
    # In 0-based: interface k sits below midpoint k-1 and above midpoint k.
    egwdffi = jnp.zeros((ncol, nlev + 1), dtype=ubm.dtype)
    interior = 0.5 * (egwdffm[:, :-1] + egwdffm[:, 1:])    # (ncol, nlev-1) at iface 1..nlev-1
    # interface index i (1..nlev-1) <- 0.5*(egwdffm[i-1]+egwdffm[i])
    egwdffi = egwdffi.at[:, 1:nlev].set(interior)
    # Mask to the E3SM band [ktop+1, kbot-1] interfaces.
    iface_idx = jnp.arange(nlev + 1)[None, :]
    iface_band = (iface_idx >= (ktop + 1)) & (iface_idx <= (kbot - 1))
    egwdffi = jnp.where(iface_band, egwdffi, 0.0)

    # Cap from above only (E3SM min(150, egwdffi)).
    egwdffi = jnp.minimum(egwd_max, egwdffi)

    # Zero at/below tend_level (E3SM: where k >= tend_level -> 0).
    below_tend = iface_idx >= tend_level[:, None]
    egwdffi = jnp.where(below_tend, 0.0, egwdffi)
    return egwdffi


def gw_diff_tridiag_coeffs(
    egwdffi: jax.Array,
    rhoi_kludge: jax.Array,
    pmid: jax.Array,
    rdpm: jax.Array,
    dt: float,
    gravit: float,
    ktop: int,
    kbot: int,
):
    """Build the implicit-diffusion tridiagonal coefficients (``vd_lu_decomp``).

    Faithful to ``vdiff_lu_solver.vd_lu_decomp`` (no surface drag, no fixed
    upper BC, ``cpairv`` absent): the off-diagonals are

        ca[k] = kv[k+1]*tmpi[k+1]*rdpm[k]      (super-diagonal of row k)
        cc[k] = kv[k]  *tmpi[k]  *rdpm[k]      (sub-diagonal of row k)

    with ``tmpi[k] = dt*(g*rhoi[k])^2/(pmid[k]-pmid[k-1])`` at interior
    interfaces and the implicit-diffusion matrix
    ``-ca*q[k+1] + (1+ca+cc)*q[k] - cc*q[k-1] = q_old[k]``.

    Returns ``(a, b, cc)`` for the Thomas solver ``a*x[k-1]+b*x[k]+c*x[k+1]=d``
    where ``a=-cc``, ``b=1+ca+cc``, ``c=-ca``.  Shapes (ncol, nlev), but the
    system is only solved on rows ``ktop .. kbot`` (other rows are identity).

    Index: k=0 top, kv/rhoi/tmpi on interfaces 0..nlev (here egwdffi/rhoi are
    (ncol, nlev+1)).  E3SM midpoint k uses interface k (above) and k+1 (below)
    in the 1-based decomp; mapped to 0-based below.
    """
    ncol, nlev = pmid.shape
    # tmpi at interfaces: dt*(g*rhoi)^2/(pmid[k]-pmid[k-1]) at interior
    # interfaces (E3SM ktop+2..kbot+1 in 1-based -> our interior interfaces).
    # tmpi[i] uses midpoints i-1 (above) and i (below): pmid[i]-pmid[i-1].
    dpmid = pmid[:, 1:] - pmid[:, :-1]                     # (ncol, nlev-1) iface 1..nlev-1
    # E3SM divides by pmid(k)-pmid(k-1) directly.  We floor the magnitude at
    # 1 Pa (codex ediff-1 #3): a SAFETY floor only — on any physical
    # monotone-pressure grid the midpoint spacing is >> 1 Pa, so this is a
    # no-op and the coefficients are bit-faithful to E3SM; it only guards a
    # pathological/degenerate column from a divide-by-zero (same pattern as the
    # dpm/piln floors used elsewhere in this module).
    dpmid = jnp.where(jnp.abs(dpmid) < 1.0, 1.0, dpmid)
    tmpi_int = dt * (gravit * rhoi_kludge[:, 1:nlev]) ** 2 / dpmid  # iface 1..nlev-1
    tmpi = jnp.zeros((ncol, nlev + 1), dtype=pmid.dtype)
    tmpi = tmpi.at[:, 1:nlev].set(tmpi_int)

    # Off-diagonals at midpoint row k:
    #   ca[k] = kv[k+1]*tmpi[k+1]*rdpm[k]   (couples to k+1, below)
    #   cc[k] = kv[k]  *tmpi[k]  *rdpm[k]   (couples to k-1, above)
    # kv on interfaces 0..nlev; midpoint row k is between iface k (above) and
    # iface k+1 (below) in 0-based top-down.
    kv = egwdffi                                           # (ncol, nlev+1)
    ca = kv[:, 1:] * tmpi[:, 1:] * rdpm                    # (ncol, nlev) k+1 iface
    cc = kv[:, :-1] * tmpi[:, :-1] * rdpm                  # (ncol, nlev) k iface

    # Only rows ktop..kbot participate; outside -> identity (ca=cc=0).
    k_idx = jnp.arange(nlev)[None, :]
    active = (k_idx >= ktop) & (k_idx <= kbot)
    ca = jnp.where(active, ca, 0.0)
    cc = jnp.where(active, cc, 0.0)
    # Top row of the active band has no upper coupling beyond the band; bottom
    # row no lower coupling.  E3SM zeroes ca at nbot and the band edges are
    # handled by zero kv at the top/bottom interfaces (egwdffi already zero
    # there), so ca/cc vanish at the band boundary automatically.

    a = -cc                                                # sub-diagonal
    b = 1.0 + ca + cc                                      # diagonal
    c = -ca                                                # super-diagonal
    # Identity rows outside the band: a=c=0, b=1 already (ca=cc=0 -> b=1).
    return a, b, c


def gw_diff_tend(
    q: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    dt: float,
):
    """Implicit GW eddy-diffusion tendency of a scalar ``q`` (``gw_diff_tend``).

    Solves the tridiagonal implicit-diffusion system ``M q_new = q`` (zero-flux
    BCs) via the shared Thomas solver and returns ``dq = (q_new - q)/dt``.
    Faithful to ``gw_diffusion.F90`` / ``vd_lu_solve``.

    Pivot note (codex ediff-1 #4)
    -----------------------------
    The GW effective diffusivity is signed (E3SM preserves negative
    egwdffi), so the off-diagonals ``ca, cc`` can be negative.  The diagonal
    ``b = 1 + ca + cc``, however, is the implicit-Euler diagonal and stays
    positive for any physical timestep because ``ca, cc`` are
    ``dt*(g*rho)^2*kv*rdpm/dp`` terms whose negative excursions are tiny
    relative to 1 (validated against the LU oracle to ~5e-16).  The shared
    ``thomas_solve`` additionally clamps a near-zero pivot to ``+_TINY`` for
    JAX safety; this differs from the Fortran LU only in the (non-physical)
    event of a singular pivot, which the E3SM system does not reach either.
    """
    from legoesm.timestepping.tridiagonal import thomas_solve
    q_new = thomas_solve(a, b, c, q)
    return (q_new - q) / dt


# ---------------------------------------------------------------------------
# momentum_energy_conservation: C.-C. Chen column fixer (gw_common.F90)
# ---------------------------------------------------------------------------

def gw_taucd_net(
    tau: jax.Array,
    c: jax.Array,
    ubi: jax.Array,
    tend_level: jax.Array,
):
    """Net signed Reynolds stress at the source/tend interface (``taucd`` sum).

    Faithful to the ``taucd`` block of ``gw_drag_prof`` (gw_common.F90:559-602)
    reduced to the only quantity the momentum fixer consumes:

        tau_net = sum_l sign(tau(l, tend_level), c(l) - ubi(tend_level))

    E3SM splits this into forward/backward (``tauf``/``taub``) and projects
    onto the four cardinal directions, but the momentum fixer only ever uses
    ``taucd_east + taucd_west = tau_net*xv`` and
    ``taucd_north + taucd_south = tau_net*yv`` (the cardinal split cancels in
    the sum), so we compute the scalar ``tau_net`` directly.

    Returns ``tau_net`` (ncol,): the total stress penetrating below the source
    to be redistributed as a uniform body force.
    """
    ncol, nwav, _ = tau.shape
    # tau at the tend_level interface, per wave.
    tau_tend = jnp.take_along_axis(
        tau, tend_level[:, None, None], axis=2
    )[:, :, 0]                                             # (ncol, nwav)
    ubi_tend = jnp.take_along_axis(
        ubi, tend_level[:, None], axis=1
    )[:, 0]                                                # (ncol,)
    # E3SM accumulates taub (c < ubi_tend) and tauf (c > ubi_tend) and the
    # fixer uses taub+tauf; a wave with c == ubi_tend contributes to NEITHER
    # (dropped).  So tau_net = sum_{c != ubi_tend} sign(tau, c-ubi_tend)
    # = sum [ +|tau| if c>ubi_tend, -|tau| if c<ubi_tend, 0 if c==ubi_tend ].
    cmu = c - ubi_tend[:, None]                            # (ncol, nwav)
    sgn = jnp.where(cmu > 0.0, 1.0, jnp.where(cmu < 0.0, -1.0, 0.0))
    tausg = jnp.abs(tau_tend) * sgn                        # (ncol, nwav)
    tau_net = jnp.sum(tausg, axis=1)                       # (ncol,)
    return tau_net


def momentum_energy_conservation(
    tend_level: jax.Array,
    dt: float,
    pdel: jax.Array,
    u: jax.Array,
    v: jax.Array,
    dudt: jax.Array,
    dvdt: jax.Array,
    dsdt: jax.Array,
    pint: jax.Array,
    gravit: float,
    tau_net: jax.Array,
    xv: jax.Array,
    yv: jax.Array,
):
    """C.-C. Chen column momentum & energy fixer (``momentum_energy_conservation``).

    Faithful to ``gw_common.F90:249``.  Two corrections, applied below the
    source level (E3SM: 1-based midpoints ``k > tend_level`` -> our 0-based
    midpoints ``m >= tend_level``):

    1. **Momentum** (codex ediff-1 #2): the net stress that penetrated below
       the source, ``tau_net`` (from :func:`gw_taucd_net`), is deposited as a
       uniform body force ``-(tau_net*xv)/dz`` / ``-(tau_net*yv)/dz`` where
       ``dz = sum_{m>=tend_level} pdel/gravit`` is the mass below the source.
       (E3SM uses ``taucd_east+taucd_west = tau_net*xv`` etc.)
    2. **Energy**: the net column total-energy change (computed AFTER the
       momentum correction) is removed uniformly from ``dsdt`` below source so
       the column total-energy budget self-closes.

    Returns the corrected ``(dudt, dvdt, dsdt)``.

    Index convention (codex ediff-1 #1): ``tend_level`` is an INTERFACE index.
    The below-source region is our 0-based midpoints ``m >= tend_level`` and
    the below-source pressure interval is ``pint[nlev] - pint[tend_level]``.
    """
    ncol, nlev = u.shape
    k_idx = jnp.arange(nlev)[None, :]
    below = k_idx >= tend_level[:, None]                   # (ncol, nlev) bool

    # --- 1. Momentum: uniform body force from the penetrating stress. ---
    # dz = mass below source = sum_{m>=tend_level} pdel/gravit.
    dz = jnp.sum(jnp.where(below, pdel, 0.0), axis=1) / gravit   # (ncol,)
    dz_safe = jnp.where(dz > 0.0, dz, 1.0)
    ut_dz = -(tau_net * xv) / dz_safe                      # (ncol,)
    vt_dz = -(tau_net * yv) / dz_safe
    # Only apply where there IS mass below source (dz>0); else no-op.
    has_below = (dz > 0.0)[:, None]
    add_u = jnp.where(below & has_below, ut_dz[:, None], 0.0)
    add_v = jnp.where(below & has_below, vt_dz[:, None], 0.0)
    dudt = dudt + add_u
    dvdt = dvdt + add_v

    # --- 2. Energy: remove the net column total-energy change below source. ---
    dE_int = jnp.sum(
        pdel * (
            dsdt
            + dudt * (u + 0.5 * dt * dudt)
            + dvdt * (v + 0.5 * dt * dvdt)
        ),
        axis=1,
    )                                                      # (ncol,)
    # Below-source pressure interval: E3SM pint(pver+1)-pint(tend_level+1) in
    # 1-based pint -> our pint[nlev]-pint[tend_level].
    pint_tend = jnp.take_along_axis(
        pint, tend_level[:, None], axis=1
    )[:, 0]
    denom = pint[:, nlev] - pint_tend
    denom = jnp.where(jnp.abs(denom) > 0.0, denom, 1.0)
    dE = dE_int / denom                                    # (ncol,)
    dsdt = jnp.where(below, dsdt - dE[:, None], dsdt)
    return dudt, dvdt, dsdt


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def e3sm_cam_gwd(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    rho: jax.Array,
    lat: jax.Array,
    dt: float,
    config: E3SMCAMConfig,
    h_topo_col: jax.Array | None = None,
    frontgf_col: jax.Array | None = None,
    netdt_col: jax.Array | None = None,
    mfcc_table: jax.Array | None = None,
    land_frac_col: jax.Array | None = None,
) -> GWDOutput:
    """Faithful E3SM/CAM gravity-wave drag (gw_drag_prof solver).

    Parameters
    ----------
    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
        Standard GWD backend signature. Column arrays (ncol, nlev);
        ``p_half``/``z_half`` are (ncol, nlev+1).  Index 0 = model top.
    h_topo_col : (ncol,) or None
        Per-column subgrid orographic stddev [m] (``sgh``) for the
        orographic source.  ``None`` -> ``config.orographic.sgh_default``.
    frontgf_col : (ncol, nlev) or None
        Frontogenesis function for the frontal source.  Required when
        ``config.source == "frontal"``.
    netdt_col : (ncol, nlev) or None
        Convective heating rate [K/s] for the Beres source.  Required when
        ``config.source == "convective"``; ``None`` -> zero heating (no
        convective waves).
    mfcc_table : array or None
        The real offline E3SM ``mfcc`` mean-flux lookup table for the Beres
        source, shape ``(maxh, 2*maxuh+1, 2*pgwv+1)`` (see
        ``E3SMBeresConfig`` / ``build_stand_in_mfcc`` for the layout).  When
        ``None`` and ``config.beres.use_stand_in_table`` is ``True`` the
        documented analytic stand-in spectrum is used (NOT bit-faithful to
        Beres-2004); when ``None`` and ``use_stand_in_table`` is ``False`` a
        ``ValueError`` is raised.
    land_frac_col : (ncol,) or None
        Per-column land fraction in [0, 1] for the E3SM driver-level
        orographic scaling ``utgw *= landfrac`` (gw_drag.F90:904-906 — the
        oro drag is zeroed over ocean BEFORE the heating closure, so ocean
        columns get neither drag nor heat).  Applied ONLY when
        ``config.source == "orographic"`` — E3SM scales the oro tendencies
        only; spectral (frontal/convective) sources ignore this argument,
        matching the oracle.  ``None`` -> no scaling (legacy behaviour,
        bit-identical).

    Returns
    -------
    GWDOutput
    """
    ncol, nlev = u.shape
    cpair = constants.c_pd
    rair = constants.R_d
    gravit = constants.g

    pmid = p_full
    pint = p_half
    dpm = jnp.abs(pint[:, 1:] - pint[:, :-1])         # (ncol, nlev)
    dpm = jnp.maximum(dpm, 1.0)
    rdpm = 1.0 / dpm
    piln = jnp.log(jnp.maximum(pint, 1.0))            # (ncol, nlev+1)
    zm = z_full

    rhoi, ti, nm, ni = gw_prof(
        T, pmid, pint, cpair, rair, gravit, config.n2min
    )

    if config.source == "orographic":
        if h_topo_col is None:
            sgh = jnp.full((ncol,), config.orographic.sgh_default, dtype=u.dtype)
        else:
            sgh = jnp.clip(jnp.asarray(h_topo_col), 0.0, None).astype(u.dtype)
        tau0, src_level, tend_level, xv, yv, c, ubm, ubi = gw_oro_src(
            u, v, T, sgh, pmid, pint, dpm, zm, nm,
            rair, config.kwv, config.fcrit2,
            config.orographic.oro_min_h, config.orographic.oro_min_wind,
        )
        orographic_only = True
        do_taper = False
    elif config.source == "frontal":
        if frontgf_col is None:
            frontgf_col = jnp.zeros((ncol, nlev), dtype=u.dtype)
        # E3SM picks kbot (launch, ~500 hPa) and kfront (trigger, ~600 hPa)
        # from the reference pressure grid — NOT a fixed fraction of nlev
        # (codex iter-1 #1/#2).  We locate the level whose column-mean
        # pressure is closest to the configured threshold.  The index stays a
        # 0-d TRACED integer (jnp.argmin, NOT int(...)) so the kernel is
        # jit-safe (codex iter-2 #1); gw_cm_src consumes it via dynamic
        # gathers / a one-hot launch mask.  The level choice itself is not
        # differentiated (argmin), matching E3SM's static init-time selection.
        pmean = jnp.mean(pmid, axis=0)                  # (nlev,)
        kbot = jnp.clip(
            jnp.argmin(jnp.abs(pmean - config.frontal.launch_p)), 1, nlev - 1
        ).astype(jnp.int32)
        kfront = jnp.clip(
            jnp.argmin(jnp.abs(pmean - config.frontal.front_p)), 0, nlev - 1
        ).astype(jnp.int32)
        tau0, src_level, tend_level, xv, yv, c, ubm, ubi = gw_cm_src(
            u, v, frontgf_col, config.pgwv, config.dc,
            config.frontal.c0, config.frontal.taubgnd,
            config.frontal.frontgfc, kbot, kfront,
            config.frontal.front_spectrum_dc_resolution,
        )
        orographic_only = False
        # E3SM tapers the frontal (CM) source by cos(lat) on structured grids.
        do_taper = True
    elif config.source == "convective":
        if netdt_col is None:
            netdt_col = jnp.zeros((ncol, nlev), dtype=u.dtype)
        # Build / select the mfcc lookup table.
        if mfcc_table is None:
            if not config.beres.use_stand_in_table:
                raise ValueError(
                    "E3SM Beres convective source: mfcc_table is None and "
                    "config.beres.use_stand_in_table is False.  Either supply "
                    "the offline E3SM mfcc table (newmfspectra*.nc, variable "
                    "'mfcc') or set use_stand_in_table=True to run with the "
                    "documented analytic stand-in spectrum."
                )
            mfcc = build_stand_in_mfcc(
                config.pgwv, config.dc, config.beres, u.dtype
            )
        else:
            mfcc = jnp.asarray(mfcc_table, dtype=u.dtype)
        # k700: 0-based midpoint index nearest the source-wind pressure
        # (~700 hPa; config.beres.source_wind_p).  Traced int, jit-safe.
        pmean = jnp.mean(pmid, axis=0)                      # (nlev,)
        k700 = jnp.clip(
            jnp.argmin(jnp.abs(pmean - config.beres.source_wind_p)), 0, nlev - 1
        ).astype(jnp.int32)
        (tau0, src_level, tend_level, xv, yv, c, ubm, ubi,
         _hdepth, _maxq0) = gw_beres_src(
            u, v, netdt_col, zm, lat, k700, mfcc,
            config.pgwv, config.dc, config.beres,
        )
        orographic_only = False
        # E3SM does NOT latitude-taper the Beres source (do_latitude_taper=F).
        do_taper = False
    else:
        raise ValueError(
            f"Unknown E3SM GWD source: {config.source!r}. "
            "Choose 'orographic', 'frontal', or 'convective'."
        )

    # Newtonian-cooling profile (spectral path only; orographic uses uniform).
    if config.use_newtonian_profile and config.source != "orographic":
        alpha_iface = newtonian_alpha_profile(pint)
    else:
        alpha_iface = None

    tau, utgw, vtgw, gwut = gw_drag_prof(
        tau0, c, src_level, tend_level, T, ti, piln, rhoi, nm, ni,
        ubm, ubi, xv, yv, dpm, rdpm, lat, config.effgw, dt, config,
        orographic_only=orographic_only, do_taper=do_taper,
        alpha_iface=alpha_iface,
    )

    du_dt = utgw
    dv_dt = vtgw

    # E3SM driver-level orographic land-fraction scaling (gw_drag.F90:904-906):
    # ``utgw(:,k) = utgw(:,k) * cam_in%landfrac`` — the oro drag is zeroed
    # over ocean.  ORDER MATTERS: E3SM scales the momentum tendencies BEFORE
    # the heating closure at :908-913 (the closure reads the scaled ptend%u),
    # so ocean columns get neither drag nor heat.  Oro source only; E3SM does
    # NOT landfrac-scale the spectral (frontal/Beres) tendencies.
    if land_frac_col is not None and config.source == "orographic":
        lfrac = jnp.clip(
            jnp.asarray(land_frac_col), 0.0, 1.0
        ).astype(u.dtype)[:, None]
        du_dt = du_dt * lfrac
        dv_dt = dv_dt * lfrac

    # Temperature tendency (thermal deposition).
    #
    # Orographic (single c=0 wave): a stationary wave does no mechanical work,
    # so the kinetic energy lost by the mean flow is deposited locally as heat.
    # Two closures (config.use_discrete_ke_heating):
    #   False (default): the CONTINUOUS-rate identity
    #     dT/dt = -(u*du + v*dv)/c_pd (exact as dt -> 0).
    #   True: the E3SM DISCRETE-step closure (gw_drag.F90:908-913, default
    #     no-energy-fix branch; ttgw = 0 for the oro ngwv=0 call per
    #     gw_common.F90:739):
    #     dT/dt = -(du*(u + 0.5*dt*du) + dv*(v + 0.5*dt*dv))/c_pd,
    #     which returns EXACTLY the discrete resolved-KE change
    #     -[KE(u+dt*du) - KE(u)]/dt as heat, closing the discrete column
    #     energy budget the way E3SM's does.  The continuous form over-heats
    #     by 0.5*dt*(du^2+dv^2)/c_pd per step (O(10%) of local heating in
    #     tndmax-limited layers at dt=1800 s).  STRUCTURAL DEPARTURE kept:
    #     E3SM applies this closure once over the ACCUMULATED ptend of all
    #     GW sources ("includes spectrum", gw_drag.F90:898-899); our
    #     per-source calls close each source independently, so the
    #     spectral-x-oro cross terms of |du_total|^2 are not represented.
    #
    # Spectral (frontal / convective): dttke is the GROUND-RELATIVE
    # wave-energy-flux-divergence term, NOT the irreversible (c-ubm)*gwut
    # heating -- it is signed (cools where U>c>0) and is not the mean-flow KE
    # removal rate eps_gwd. The E3SM-3.0.1 oracle (gw_common.F90:727,
    #   dttke(:,k) = dttke(:,k) + c(:,l) * gwut(:,k,l))
    # uses ``sum_l c_l * gwut_l`` — faithfully reproduced below.  When
    # ``config.do_eddy_diffusion`` is on, E3SM ALSO adds the dse-diffusion
    # heating ``dttdf`` (ttgw = dttke + dttdf, gw_common.F90:731); the GW eddy
    # diffusion of u, v, dse runs through the implicit tridiagonal solver.
    #
    # NOTE (cross-version): newer CAM/EAM trunk uses the intrinsic-frequency
    # form ``sum_l (c_l - ubm) * gwut_l`` instead (codex iter-2 #2).  We match
    # the pinned E3SM-3.0.1 oracle, not the trunk; the two differ by a
    # ``-ubm*gwut`` term.  Set ``config.dttke_use_intrinsic`` to switch.
    if config.source == "orographic":
        if config.use_discrete_ke_heating:
            dT_dt = -(
                du_dt * (u + 0.5 * dt * du_dt)
                + dv_dt * (v + 0.5 * dt * dv_dt)
            ) / cpair
        else:
            dT_dt = -(u * du_dt + v * dv_dt) / cpair
    else:
        if config.dttke_use_intrinsic:
            ceff = c[:, None, :] - ubm[:, :, None]      # (ncol, nlev, nwav)
        else:
            ceff = c[:, None, :]
        dttke = jnp.sum(ceff * gwut, axis=2)            # (ncol, nlev)
        dT_dt = dttke / cpair

        # ---- GW-induced eddy diffusion (gw_ediff + gw_diff_tend) ----
        # Spectral path only.  Provenance (codex final-3): in E3SM, inside
        # gw_drag_prof the effective diffusivity egwdffi is used by gw_diff_tend
        # ONLY for the dry-static-energy heating term ``dttdf`` (ttgw =
        # dttke + dttdf, gw_common.F90:721-731) and for constituents (qtgw); the
        # diffusion of MOMENTUM (u, v) and temperature by egwdffi is applied
        # DOWNSTREAM by the host vertical_diffusion scheme (egwdffi is exported
        # as the EKGWSPEC diagnostic and summed into the model's eddy
        # diffusivity).  For a self-contained GWD module we apply BOTH here: the
        # E3SM-faithful dse heating ``dttdf`` (added to dT_dt), AND the
        # u/v eddy diffusion that E3SM defers to vertical_diffusion (so the GW
        # momentum eddy flux is not silently dropped when this module is used
        # standalone).  The implicit tridiagonal solve reuses the shared
        # thomas_solve and is validated against the E3SM LU oracle to ~5e-16.
        if config.do_eddy_diffusion:
            ktop = 0
            # E3SM kbotbg = the bottom of the background-wave region (~500 hPa).
            # Keep the index a 0-d TRACED int32 (jnp.argmin, NOT int(...)) so the
            # ediff path is jit-safe; gw_ediff / the coeff builder consume it via
            # comparison masks (no static slicing on kbot).
            pmean_b = jnp.mean(pmid, axis=0)
            kbot = jnp.clip(
                jnp.argmin(jnp.abs(pmean_b - config.ediff_kbot_p)), 1, nlev - 1
            ).astype(jnp.int32)
            # Recalculate rhoi "kludge" exactly as E3SM (preserve answers).
            rhoi_kludge = jnp.zeros((ncol, nlev + 1), dtype=u.dtype)
            rhoi_kludge = rhoi_kludge.at[:, 0].set(
                pint[:, 0] / (rair * T[:, 0])
            )
            rhoi_kludge = rhoi_kludge.at[:, 1:nlev].set(
                pint[:, 1:nlev] * 2.0 / (rair * (T[:, 1:] + T[:, :-1]))
            )
            rhoi_kludge = rhoi_kludge.at[:, nlev].set(
                pint[:, nlev] / (rair * T[:, nlev - 1])
            )
            egwdffi = gw_ediff(
                gwut, ubm, nm, c, rhoi_kludge, pmid, rdpm, tend_level,
                dt, gravit, rair, ktop, kbot, config.prndl, config.egwd_max,
            )
            a, b, cc = gw_diff_tridiag_coeffs(
                egwdffi, rhoi_kludge, pmid, rdpm, dt, gravit, ktop, kbot,
            )
            # Diffuse u, v through the GW eddy diffusivity (constituent path).
            du_diff = gw_diff_tend(u, a, b, cc, dt)
            dv_diff = gw_diff_tend(v, a, b, cc, dt)
            du_dt = du_dt + du_diff
            dv_dt = dv_dt + dv_diff
            # Dry static energy s = cpair*T + g*z; diffuse it and convert the
            # dse tendency to a temperature tendency (dttdf).
            dse = cpair * T + gravit * z_full
            dttdf = gw_diff_tend(dse, a, b, cc, dt)
            dT_dt = dT_dt + dttdf / cpair

        # ---- Column momentum & energy conservation (C.-C. Chen fixer) ----
        if config.do_energy_conservation:
            # Net penetrating stress at the source/tend interface (taucd sum),
            # redistributed as a uniform below-source body force.
            tau_net = gw_taucd_net(tau, c, ubi, tend_level)
            # dsdt = cpair * dT_dt (dry-static-energy tendency).
            dsdt = cpair * dT_dt
            du_dt, dv_dt, dsdt = momentum_energy_conservation(
                tend_level, dt, dpm, u, v, du_dt, dv_dt, dsdt, pint, gravit,
                tau_net, xv, yv,
            )
            dT_dt = dsdt / cpair

    # Column mean-flow KE removal rate = -integral rho*(u*du+v*dv)*dz. Positive
    # (dissipative) for the orographic path; for a spectral source it is SIGNED
    # (negative where the spectrum accelerates the resolved flow toward c).
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)

    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
